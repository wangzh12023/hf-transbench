import torch
import torch.nn.functional as F

from .attention_utils import apply_causal_mask, finalize_attention, prepare_qkv
from .configuration_llama import HeadSoftmaxLlamaConfig
from .custom_llama_base import CustomAttentionLlamaForCausalLM, CustomAttentionLlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention


class HeadSoftmaxLlamaAttention(LlamaAttention):
    config_class = HeadSoftmaxLlamaConfig

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

        bsz, _, q_len, k_len = attn_scores.shape
        num_kv_heads = self.config.num_key_value_heads
        group_size = self.num_key_value_groups

        attn_weights = attn_scores.view(bsz, num_kv_heads, group_size, q_len, k_len)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32)
        attn_weights = attn_weights.view(bsz, num_kv_heads * group_size, q_len, k_len).to(query_states.dtype)

        attn_output = finalize_attention(self, input_shape, attn_weights, value_states)
        if not kwargs.get("output_attentions", False):
            attn_weights = None
        return attn_output, attn_weights


class HeadSoftmaxLlamaModel(CustomAttentionLlamaModel):
    config_class = HeadSoftmaxLlamaConfig
    attention_cls = HeadSoftmaxLlamaAttention


class HeadSoftmaxLlamaForCausalLM(CustomAttentionLlamaForCausalLM):
    config_class = HeadSoftmaxLlamaConfig
    model_cls = HeadSoftmaxLlamaModel
