import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Callable

import math
from transformers.cache_utils import Cache
from .configuration_llama import MySigmoidLlamaConfig
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
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
class MySigmoidLlamaAttention(LlamaAttention):
    config_class = MySigmoidLlamaConfig
    def __init__(self, config: MySigmoidLlamaConfig,layer_idx: int):
        super().__init__(config, layer_idx) 
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        #gqa
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        # seq_len = attn_weights.size(-1) 
        # bias = -torch.log(torch.tensor(seq_len, dtype=attn_weights.dtype, device=attn_weights.device))
        # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.sigmoid(attn_weights).to(query_states.dtype)
        # attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    #     attention_mask: Optional[torch.Tensor],
    #     past_key_value: Optional[Cache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Cache]]:
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, self.head_dim)

    #     query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #     value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    #     cos, sin = position_embeddings
    #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    #     if past_key_value is not None:
    #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    #     #Use sigmoid instead of softmax
    #     #manually compute attention output using sigmoid
    #     #shape: [batch, num_heads, q_len, k_len]
    #     if self.num_key_value_groups > 1:
    #         key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
    #         value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
    #     attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

    #     if attention_mask is not None:
    #         attn_scores = attn_scores + attention_mask
    #     attn_probs = torch.sigmoid(attn_scores)
    #     #replace sigmoid to softmax to checkout if the implementation wrong
    #     # attn_probs = F.softmax(attn_scores, dim=-1)
    #     attn_probs = F.dropout(attn_probs, p=self.attention_dropout, training=self.training)

    #     attn_output = torch.matmul(attn_probs, value_states)
    #     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #     attn_output = self.o_proj(attn_output)

    #     return attn_output, attn_probs, past_key_value  


class MySigmoidLlamaModel(LlamaModel):
    config_class = MySigmoidLlamaConfig

    def __init__(self, config: MySigmoidLlamaConfig):
        super().__init__(config)
        for idx, block in enumerate(self.layers):
            block.self_attn = MySigmoidLlamaAttention(config, layer_idx=idx)

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

class MySigmoidLlamaForCausalLM(LlamaForCausalLM):
    config_class = MySigmoidLlamaConfig

    def __init__(self, config: MySigmoidLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MySigmoidLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
