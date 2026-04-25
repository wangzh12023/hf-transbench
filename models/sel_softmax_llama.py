import torch
import torch.nn.functional as F

from .attention_utils import apply_causal_mask, finalize_attention, prepare_qkv
from .configuration_llama import SelSoftmaxLlamaConfig
from .custom_llama_base import CustomAttentionLlamaForCausalLM, CustomAttentionLlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention


class SelSoftmaxLlamaAttention(LlamaAttention):
    config_class = SelSoftmaxLlamaConfig

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape, query_states, key_states, value_states = prepare_qkv(
            self,
            hidden_states,
            position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_scores = apply_causal_mask(attn_scores, attention_mask, key_states)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = finalize_attention(self, input_shape, attn_weights, value_states)
        if not kwargs.get("output_attentions", False):
            attn_weights = None
        return attn_output, attn_weights


class SelSoftmaxLlamaModel(CustomAttentionLlamaModel):
    config_class = SelSoftmaxLlamaConfig
    attention_cls = SelSoftmaxLlamaAttention


class SelSoftmaxLlamaForCausalLM(CustomAttentionLlamaForCausalLM):
    config_class = SelSoftmaxLlamaConfig
    model_cls = SelSoftmaxLlamaModel
