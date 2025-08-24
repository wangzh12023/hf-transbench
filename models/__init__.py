from .configuration_llama import MyLlamaConfig, MySigmoidLlamaConfig, MyLinearLlamaConfig, MySigmoidWithBLlamaConfig, HeadSoftmaxLlamaConfig
from .modeling_llama import MyLlamaModel, MyLlamaForCausalLM
from .sigmoid_llama import  MySigmoidLlamaModel, MySigmoidLlamaForCausalLM
from .Linear_llama import MyLinearLlamaModel, MyLinearLlamaForCausalLM
from .sigmoid_llama_with_b import MySigmoidWithBLlamaModel, MySigmoidWithBLlamaForCausalLM
from .head_softmax_llama import HeadSoftmaxLlamaModel, HeadSoftmaxLlamaForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification

AutoConfig.register("my-llama", MyLlamaConfig)
AutoModel.register(MyLlamaConfig, MyLlamaModel)
AutoModelForCausalLM.register(MyLlamaConfig, MyLlamaForCausalLM)

AutoConfig.register("my-llama-sigmoid", MySigmoidLlamaConfig)
AutoModel.register(MySigmoidLlamaConfig, MySigmoidLlamaModel)
AutoModelForCausalLM.register(MySigmoidLlamaConfig, MySigmoidLlamaForCausalLM)

AutoConfig.register("my-llama-linear", MyLinearLlamaConfig)
AutoModel.register(MyLinearLlamaConfig, MyLinearLlamaModel)
AutoModelForCausalLM.register(MyLinearLlamaConfig, MyLinearLlamaForCausalLM)

AutoConfig.register("my-llama-sigmoid-with-b", MySigmoidWithBLlamaConfig)
AutoModel.register(MySigmoidWithBLlamaConfig, MySigmoidWithBLlamaModel)
AutoModelForCausalLM.register(MySigmoidWithBLlamaConfig, MySigmoidWithBLlamaForCausalLM)

AutoConfig.register("my-llama-head-softmax", HeadSoftmaxLlamaConfig)
AutoModel.register(HeadSoftmaxLlamaConfig, HeadSoftmaxLlamaModel)
AutoModelForCausalLM.register(HeadSoftmaxLlamaConfig, HeadSoftmaxLlamaForCausalLM)
