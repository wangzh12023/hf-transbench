from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


def prepare_qkv(
    module: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)
    return input_shape, query_states, key_states, value_states


def apply_causal_mask(attn_scores: torch.Tensor, attention_mask: Optional[torch.Tensor], key_states: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return attn_scores

    if attention_mask.dim() != 4:
        raise ValueError(
            "Custom attention expects a 4D causal mask. "
            "Use --attn_implementation eager for custom attention variants."
        )

    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    return attn_scores + causal_mask


def finalize_attention(
    module: LlamaAttention,
    input_shape: tuple[int, ...],
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    attn_weights = torch.nn.functional.dropout(
        attn_weights,
        p=0.0 if not module.training else module.attention_dropout,
        training=module.training,
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    return module.o_proj(attn_output)