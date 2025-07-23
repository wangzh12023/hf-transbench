import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Callable


from transformers.cache_utils import Cache
from .configuration_llama import MySigmoidWithBLlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,

    LlamaAttention,

    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,

)
from torch.nn import CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention


from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class MySigmoidWithBLlamaAttention(LlamaAttention):
    config_class = MySigmoidWithBLlamaConfig
    def __init__(self, config: MySigmoidWithBLlamaConfig,layer_idx: int):
        super().__init__(config, layer_idx) 
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Cache]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # —— Use sigmoid instead of softmax ——
        # manually compute attention output using sigmoid
        # shape: [batch, num_heads, q_len, k_len]
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        # Apply sigmoid with bias b = -log(n) where n is sequence length
        seq_len = attn_scores.size(-1) 
        bias = -torch.log(torch.tensor(seq_len, dtype=attn_scores.dtype, device=attn_scores.device))
        attn_probs = torch.sigmoid(attn_scores + bias)
        attn_probs = F.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs, past_key_value 


class MySigmoidWithBLlamaModel(LlamaModel):
    config_class = MySigmoidWithBLlamaConfig

    def __init__(self, config: MySigmoidWithBLlamaConfig):
        super().__init__(config)
        # Replace every block’s self-attention
        for idx, block in enumerate(self.layers):
            block.self_attn = MySigmoidWithBLlamaAttention(config, layer_idx=idx)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return output

class MySigmoidWithBLlamaForCausalLM(LlamaForCausalLM):
    config_class = MySigmoidWithBLlamaConfig

    def __init__(self, config: MySigmoidWithBLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MySigmoidWithBLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        output_attentions = return_dict if return_dict is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Base model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
