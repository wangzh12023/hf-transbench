# coding=utf-8
"""Custom LLaMA configuration classes used by AutoConfig registry."""

from transformers.models.llama.configuration_llama import LlamaConfig


class MyLlamaConfig(LlamaConfig):
    model_type = "my-llama"


class MySigmoidLlamaConfig(LlamaConfig):
    model_type = "my-llama-sigmoid"


class MyLinearLlamaConfig(LlamaConfig):
    model_type = "my-llama-linear"


class MySigmoidWithBLlamaConfig(LlamaConfig):
    model_type = "my-llama-sigmoid-with-b"


class HeadSoftmaxLlamaConfig(LlamaConfig):
    model_type = "my-llama-head-softmax"


class SelSoftmaxLlamaConfig(LlamaConfig):
    model_type = "my-llama-sel-softmax"


class HeadSoftmaxWithBLlamaConfig(LlamaConfig):
    model_type = "head-softmax-with-b"


class SoftmaxAndHeadSoftmaxLlamaConfig(LlamaConfig):
    model_type = "softmax-and-head-softmax"
