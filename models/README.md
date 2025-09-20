
## 模型概览

| 模型 | 注意力机制 | 核心改变 | 文件 |
|------|------------|----------|------|
| MySigmoidLlama | Sigmoid注意力 | 用sigmoid替换softmax | `sigmoid_llama.py` |
| MySigmoidWithBLlama | 带偏置的Sigmoid | $sigmoid(x) = \frac{1}{1+e^{-(x+b)}}$ , where $b = -\log(n)$ | `sigmoid_llama_with_b.py` |
| HeadSoftmaxLlama | 多头维度Softmax | 在head维度应用softmax | `head_softmax_llama.py` |
| HeadSoftmaxLlamaWithB | 多头维度Softmax | 在head维度应用softmax,且加偏置b | `head_softmax_llama_with_b.py` |
| MyLlama | 标准LLaMA | 原始实现 | `modeling_llama.py` |
