import torch

from .attention_utils import apply_causal_mask, finalize_attention, prepare_qkv
from .configuration_llama import MySigmoidLlamaConfig
from .custom_llama_base import CustomAttentionLlamaForCausalLM, CustomAttentionLlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention


class MySigmoidLlamaAttention(LlamaAttention):
    config_class = MySigmoidLlamaConfig

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
        attn_weights = torch.sigmoid(attn_scores).to(query_states.dtype)

        attn_output = finalize_attention(self, input_shape, attn_weights, value_states)
        if not kwargs.get("output_attentions", False):
            attn_weights = None
        return attn_output, attn_weights


class MySigmoidLlamaModel(CustomAttentionLlamaModel):
    config_class = MySigmoidLlamaConfig
    attention_cls = MySigmoidLlamaAttention


class MySigmoidLlamaForCausalLM(CustomAttentionLlamaForCausalLM):
    config_class = MySigmoidLlamaConfig
    model_cls = MySigmoidLlamaModel
