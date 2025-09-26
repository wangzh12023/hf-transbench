from .configuration_llama import MyLlamaConfig, MySigmoidLlamaConfig, MyLinearLlamaConfig, MySigmoidWithBLlamaConfig, HeadSoftmaxLlamaConfig, HeadSoftmaxWithBLlamaConfig, SoftmaxAndHeadSoftmaxLlamaConfig
from .modeling_llama import MyLlamaModel, MyLlamaForCausalLM
from .sigmoid_llama import  MySigmoidLlamaModel, MySigmoidLlamaForCausalLM
from .Linear_llama import MyLinearLlamaModel, MyLinearLlamaForCausalLM
from .sigmoid_llama_with_b import MySigmoidWithBLlamaModel, MySigmoidWithBLlamaForCausalLM
from .head_softmax_llama import HeadSoftmaxLlamaModel, HeadSoftmaxLlamaForCausalLM
from .head_softmax_with_b import HeadSoftmaxLlamaModel as HeadSoftmaxWithBLlamaModel, HeadSoftmaxLlamaForCausalLM as HeadSoftmaxWithBLlamaForCausalLM
from .twosoftmax_llama import TwoSoftmaxLlamaModel, TwoSoftmaxLlamaForCausalLM
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

AutoConfig.register("head-softmax-with-b", HeadSoftmaxWithBLlamaConfig)
AutoModel.register(HeadSoftmaxWithBLlamaConfig, HeadSoftmaxWithBLlamaModel)
AutoModelForCausalLM.register(HeadSoftmaxWithBLlamaConfig, HeadSoftmaxWithBLlamaForCausalLM)

AutoConfig.register("softmax-and-head-softmax", SoftmaxAndHeadSoftmaxLlamaConfig)
AutoModel.register(SoftmaxAndHeadSoftmaxLlamaConfig, TwoSoftmaxLlamaModel)
AutoModelForCausalLM.register(SoftmaxAndHeadSoftmaxLlamaConfig, TwoSoftmaxLlamaForCausalLM)