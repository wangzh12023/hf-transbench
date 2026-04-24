from typing import Type

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaModel


class CustomAttentionLlamaModel(LlamaModel):
    attention_cls: Type[LlamaAttention] = LlamaAttention

    def __init__(self, config):
        super().__init__(config)
        for idx, layer in enumerate(self.layers):
            layer.self_attn = self.attention_cls(config, layer_idx=idx)


class CustomAttentionLlamaForCausalLM(LlamaForCausalLM):
    model_cls: Type[LlamaModel] = LlamaModel

    def __init__(self, config):
        super().__init__(config)
        self.model = self.model_cls(config)
        self.post_init()