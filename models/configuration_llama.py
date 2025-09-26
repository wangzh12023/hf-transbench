# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from typing import Optional, Tuple, Union, List
import torch    


from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


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
    
class HeadSoftmaxWithBLlamaConfig(LlamaConfig):
    model_type = "head-softmax-with-b"
    
    
class SoftmaxAndHeadSoftmaxLlamaConfig(LlamaConfig):
    model_type = "softmax-and-head-softmax"