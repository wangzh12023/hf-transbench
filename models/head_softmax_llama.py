import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_utils import apply_causal_mask, finalize_attention, prepare_qkv
from .configuration_llama import HeadSoftmaxLlamaConfig
from .custom_llama_base import CustomAttentionLlamaForCausalLM, CustomAttentionLlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention


class HeadSoftmaxLlamaAttention(LlamaAttention):
    config_class = HeadSoftmaxLlamaConfig

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.head_gate_logits = nn.Parameter(
            torch.zeros(self.config.num_key_value_heads, self.num_key_value_groups)
        )

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

        # 1. 保留标准 token-wise softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

        bsz, _, q_len, k_len = attn_weights.shape
        num_kv_heads = self.config.num_key_value_heads
        group_size = self.num_key_value_groups

        # 2. 在 GQA group 内做 head/channel gate
        # shape: [num_kv_heads, group_size]
        head_gate = F.softmax(self.head_gate_logits, dim=-1, dtype=torch.float32)
        
        # 保持初始化尺度不变：初始 softmax = 1 / group_size，因此乘 group_size 后为 1
        head_gate = head_gate * group_size

        # 3. 应用到 attention weights
        attn_weights = attn_weights.view(bsz, num_kv_heads, group_size, q_len, k_len)
        attn_weights = attn_weights * head_gate.view(1, num_kv_heads, group_size, 1, 1)
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
