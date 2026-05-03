import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half
from transformers.utils import ModelOutput, logging

from .configuration_pt import PtConfig


logger = logging.get_logger(__name__)


class RopeApplier:
    def __init__(self, cos, sin, unsqueeze_dim=1) -> None:
        self.cos = cos.unsqueeze(unsqueeze_dim)
        self.sin = sin.unsqueeze(unsqueeze_dim)

    def apply(self, qkv):
        return (qkv * self.cos) + (rotate_half(qkv) * self.sin)

    def apply_o(self, o):
        return (o * self.cos) - (rotate_half(o) * self.sin)


class SquaredSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32).pow(2)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps) * hidden_states.shape[self.dim]
        return hidden_states.to(input_dtype)


class AbsNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = F.relu(hidden_states.to(torch.float32))
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps) * hidden_states.shape[self.dim]
        return hidden_states.to(input_dtype)


class Softmax(nn.Softmax):
    def __init__(self, dim=-1, eps=None):
        super().__init__(dim=dim)


POTENTIAL2ACT = {
    "exp": Softmax,
    "abs": AbsNormalization,
    "square": SquaredSoftmax,
}


def _invalid_dependency_positions(dependency_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if dependency_mask is None:
        return None
    return dependency_mask < 0


class PtHeadSelection(nn.Module):
    """Multi-channel PT dependency-head update with selectable normalizers."""

    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.num_channels = config.num_channels
        self.ternary_rank = config.ternary_rank

        self.ternary_factor_u = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)

        if config.channel_gate_type == "static":
            self.channel_gate_logits = nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter("channel_gate_logits", None)

        if config.channel_gate_type == "token":
            self.token_channel_gate = nn.Linear(self.dim_z, self.num_channels, bias=True)
        else:
            self.token_channel_gate = None

        self._init_ternary()

    def _init_ternary(self):
        std = 1.0 / math.sqrt(self.config.dim_z)
        nn.init.normal_(self.ternary_factor_u, mean=0.0, std=std)
        nn.init.normal_(self.ternary_factor_v, mean=0.0, std=std)
        self.reset_channel_gate_parameters()

    def reset_channel_gate_parameters(self):
        if self.channel_gate_logits is not None:
            nn.init.zeros_(self.channel_gate_logits)
        if self.token_channel_gate is not None:
            nn.init.zeros_(self.token_channel_gate.weight)
            nn.init.zeros_(self.token_channel_gate.bias)

    def _length_bias(self, scores: torch.Tensor, dependency_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if dependency_mask is None:
            valid_counts = torch.full(
                (scores.shape[0], 1, scores.shape[2], 1),
                scores.shape[-1],
                dtype=torch.float32,
                device=scores.device,
            )
        else:
            invalid = _invalid_dependency_positions(dependency_mask)
            valid_counts = (~invalid).sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
        return -torch.log(valid_counts).to(scores.dtype)

    def _normalize_heads(self, scores: torch.Tensor, dependency_mask: Optional[torch.Tensor]) -> torch.Tensor:
        method = self.config.head_selection_method
        if method in {"softmax", "sel_softmax"}:
            return F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

        if method == "sigmoid":
            return torch.sigmoid(scores).to(scores.dtype)

        if method == "sigmoid_with_b":
            return torch.sigmoid(scores + self._length_bias(scores, dependency_mask)).to(scores.dtype)

        if method == "normalized_sigmoid":
            weights = torch.sigmoid(scores)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(self.config.normalized_sigmoid_eps)
            return (weights / denom).to(scores.dtype)

        if method == "linear":
            weights = scores
            invalid = _invalid_dependency_positions(dependency_mask)
            if invalid is not None:
                weights = weights.masked_fill(invalid, 0.0)
            return weights.to(scores.dtype)

        raise ValueError(f"Unsupported PT head_selection_method={method!r}")

    def _apply_pairwise_channel_softmax(
        self,
        qh: torch.Tensor,
        dependency_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.pairwise_channel_softmax:
            return qh
        qh = F.softmax(qh, dim=1, dtype=torch.float32).to(qh.dtype)
        invalid = _invalid_dependency_positions(dependency_mask)
        if invalid is not None:
            qh = qh.masked_fill(invalid, 0.0)
        return qh

    def _channel_gate(
        self,
        qz: torch.Tensor,
        raw_scores: torch.Tensor,
        dependency_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        gate_type = self.config.channel_gate_type
        if gate_type == "none":
            return None

        temperature = self.config.channel_gate_temperature
        scale = float(self.num_channels)

        if gate_type == "static":
            gate = F.softmax(self.channel_gate_logits / temperature, dim=-1, dtype=torch.float32) * scale
            return gate.to(qz.dtype).view(1, self.num_channels, 1, 1)

        if gate_type == "token":
            gate_logits = self.token_channel_gate(qz) / temperature
            gate = F.softmax(gate_logits, dim=-1, dtype=torch.float32) * scale
            return gate.to(qz.dtype).transpose(1, 2).unsqueeze(-1)

        if gate_type == "message":
            invalid = _invalid_dependency_positions(dependency_mask)
            gate_scores = raw_scores
            if invalid is not None:
                gate_scores = gate_scores.masked_fill(invalid, 0.0)
                valid_counts = (~invalid).sum(dim=-1).clamp_min(1)
                gate_logits = gate_scores.sum(dim=-1) / valid_counts
            else:
                gate_logits = gate_scores.mean(dim=-1)
            gate = F.softmax(gate_logits / temperature, dim=1, dtype=torch.float32) * scale
            return gate.to(qz.dtype).unsqueeze(-1)

        raise ValueError(f"Unsupported PT channel_gate_type={gate_type!r}")

    def _transpose_dependencies_for_message(self, qh: torch.Tensor) -> torch.Tensor:
        transposed_qh = qh.transpose(2, 3)
        if not self.config.is_causal:
            return transposed_qh

        # PT has an incoming-message term. In CLM, future queries must not send
        # information back to earlier target positions through this transpose.
        seq_len = qh.shape[-1]
        target_positions = torch.arange(seq_len, device=qh.device).view(1, 1, seq_len, 1)
        source_positions = torch.arange(seq_len, device=qh.device).view(1, 1, 1, seq_len)
        causal_message_mask = source_positions <= target_positions
        return transposed_qh.masked_fill(~causal_message_mask, 0.0)

    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dependencies: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = qz.size()

        qz_u = F.linear(qz, self.ternary_factor_u) * self.config.ternary_factor_scaling
        qz_v = F.linear(qz, self.ternary_factor_v) * self.config.ternary_factor_scaling

        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)

        cos, sin = position_embeddings
        rope_applier = RopeApplier(cos, sin)
        qz_uo = rope_applier.apply_o(qz_u)
        qz_u = rope_applier.apply(qz_u)
        qz_v = rope_applier.apply(qz_v)

        raw_message_f = torch.matmul(qz_u, qz_v.transpose(2, 3)) / self.ternary_rank
        message_f = raw_message_f

        expected_size = (bsz, self.num_channels, seq_len, seq_len)
        if message_f.size() != expected_size:
            raise ValueError(f"Dependency scores should be of size {expected_size}, got {message_f.size()}")

        if dependency_mask is not None:
            if dependency_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Dependency mask should be of size {(bsz, 1, seq_len, seq_len)}, got {dependency_mask.size()}"
                )
            message_f = message_f + dependency_mask

        scores = message_f / self.config.regularize_h
        qh = self._normalize_heads(scores, dependency_mask)
        qh = self._apply_pairwise_channel_softmax(qh, dependency_mask)

        channel_gate = self._channel_gate(qz, raw_message_f, dependency_mask)
        if channel_gate is not None:
            qh = qh * channel_gate

        qh = self.dropout(qh)

        qh_v1 = torch.matmul(qh, qz_v)
        qh_v2 = torch.matmul(self._transpose_dependencies_for_message(qh), qz_uo)

        qh_v1 = rope_applier.apply_o(qh_v1)
        qh_v2 = rope_applier.apply(qh_v2)

        expected_message_size = (bsz, self.num_channels, seq_len, self.ternary_rank)
        if qh_v1.size() != expected_message_size:
            raise ValueError(f"`qh_v1` should be of size {expected_message_size}, got {qh_v1.size()}")
        if qh_v2.size() != expected_message_size:
            raise ValueError(f"`qh_v2` should be of size {expected_message_size}, got {qh_v2.size()}")

        qh_v1 = qh_v1.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        qh_v2 = qh_v2.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.num_channels * self.ternary_rank)

        message_g = (
            torch.matmul(qh_v1, self.ternary_factor_u) + torch.matmul(qh_v2, self.ternary_factor_v)
        ) * self.config.ternary_factor_scaling

        if not output_dependencies:
            qh = None

        return message_g, qh


class PtTopicModeling(nn.Module):
    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.dim_g = config.dim_g
        self.binary_factor = nn.Parameter(torch.empty(self.dim_g, self.dim_z))
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)

        std = 1.0 / math.sqrt(self.config.dim_z)
        nn.init.normal_(self.binary_factor, mean=0.0, std=std)

    def forward(self, qz: torch.Tensor):
        qg = F.linear(qz, self.binary_factor) * self.config.binary_factor_scaling
        qg = self.act(qg / self.config.regularize_g)
        return qg @ self.binary_factor * self.config.binary_factor_scaling


class PtEncoderIterator(nn.Module):
    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.head_selection = PtHeadSelection(config=config)
        self.topic_modeling = PtTopicModeling(config)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_dependencies: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        old_qz = qz
        qz = self.norm(qz)

        m1, qh = self.head_selection(
            qz=qz,
            dependency_mask=dependency_mask,
            position_ids=position_ids,
            output_dependencies=output_dependencies,
            position_embeddings=position_embeddings,
        )
        m2 = self.topic_modeling(qz)

        qz = (m1 + m2 + unary_potentials) / self.config.regularize_z
        qz = (qz + old_qz) * 0.5

        outputs = (qz,)
        if output_dependencies:
            outputs += (qh,)
        return outputs


@dataclass
class PtModelOutput(ModelOutput):
    last_qz: torch.FloatTensor = None
    all_qzs: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_qhs: Optional[Tuple[torch.FloatTensor, ...]] = None


class PtPreTrainedModel(PreTrainedModel):
    config_class = PtConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PtEncoderIterator"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 1.0 / math.sqrt(self.config.dim_z)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PtModel(PtPreTrainedModel):
    config_class = PtConfig

    def __init__(self, config: PtConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.unary_factors = nn.Embedding(config.vocab_size, config.dim_z, self.padding_idx)
        self.iterator = PtEncoderIterator(config)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        config_copy = PtConfig.from_dict(config.to_dict())
        config_copy.head_dim = config.ternary_rank
        config_copy.hidden_size = config.dim_z
        config_copy.num_attention_heads = config.num_channels
        self.rotary_emb = LlamaRotaryEmbedding(config=config_copy)

        self.gradient_checkpointing = False
        self.post_init()
        self.iterator.head_selection.reset_channel_gate_parameters()

    def get_input_embeddings(self):
        return self.unary_factors

    def set_input_embeddings(self, value):
        self.unary_factors = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unary_potentials: Optional[torch.FloatTensor] = None,
        output_dependencies: Optional[bool] = None,
        output_qzs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PtModelOutput]:
        output_dependencies = output_dependencies if output_dependencies is not None else self.config.output_attentions
        output_qzs = output_qzs if output_qzs is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (unary_potentials is None):
            raise ValueError("Specify exactly one of input_ids or unary_potentials/inputs_embeds.")

        if unary_potentials is None:
            unary_potentials = self.unary_factors(input_ids)

        bsz, seq_length, _ = unary_potentials.size()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else unary_potentials.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        dependency_mask = self._update_dependency_mask(dependency_mask, unary_potentials)

        qz = unary_potentials
        position_embeddings = self.rotary_emb(qz, position_ids)

        all_qzs = () if output_qzs else None
        all_qhs = () if output_dependencies else None

        for _ in range(self.config.num_iterations):
            if output_qzs:
                all_qzs += (qz,)

            iter_outputs = self.iterator(
                unary_potentials,
                qz,
                dependency_mask=dependency_mask,
                position_ids=position_ids,
                output_dependencies=output_dependencies,
                position_embeddings=position_embeddings,
            )
            qz = iter_outputs[0]

            if output_dependencies:
                all_qhs += (iter_outputs[1],)

        if output_qzs:
            all_qzs += (qz,)

        qz = self.norm(qz)

        if not return_dict:
            return tuple(v for v in [qz, all_qzs, all_qhs] if v is not None)
        return PtModelOutput(last_qz=qz, all_qzs=all_qzs, all_qhs=all_qhs)

    def _update_dependency_mask(self, dependency_mask: Optional[torch.Tensor], unary_potentials: torch.Tensor):
        bsz, seq_length, _ = unary_potentials.shape

        if dependency_mask is None:
            dependency_mask = torch.ones(
                (bsz, seq_length),
                dtype=torch.long,
                device=unary_potentials.device,
            )

        if dependency_mask.dim() == 4:
            mask_4d = dependency_mask.to(dtype=unary_potentials.dtype, device=unary_potentials.device)
        elif dependency_mask.dim() == 2:
            attn_mask_converter = AttentionMaskConverter(is_causal=self.config.is_causal)
            mask_4d = attn_mask_converter.to_4d(
                dependency_mask,
                seq_length,
                dtype=unary_potentials.dtype,
                key_value_length=seq_length,
            )
        else:
            raise ValueError(
                f"PT dependency_mask expects shape [batch, seq] or [batch, 1, seq, seq], got {dependency_mask.shape}"
            )

        if self.config.mask_self:
            diag_mask = torch.eye(seq_length, dtype=torch.bool, device=mask_4d.device).unsqueeze(0).unsqueeze(0)
            mask_4d = mask_4d.masked_fill(diag_mask, torch.finfo(mask_4d.dtype).min)

        return mask_4d


class PtForCausalLM(PtPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config: PtConfig):
        super().__init__(config)
        self.model = PtModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.vocab_size = config.vocab_size

        self.post_init()
        self.model.iterator.head_selection.reset_channel_gate_parameters()
        self.cls.predictions.decoder.weight.data.normal_(mean=0.0, std=1.0 / config.dim_z)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            dependency_mask=attention_mask,
            position_ids=position_ids,
            unary_potentials=inputs_embeds,
            output_dependencies=output_attentions,
            output_qzs=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] * self.config.classifier_amplifier
        logits = self.cls(sequence_output).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.all_qzs,
            attentions=outputs.all_qhs,
        )
