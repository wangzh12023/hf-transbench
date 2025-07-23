
## 模型概览

| 模型 | 注意力机制 | 核心改变 | 文件 |
|------|------------|----------|------|
| MySigmoidLlama | Sigmoid注意力 | 用sigmoid替换softmax | `sigmoid_llama.py` |
| MySigmoidWithBLlama | 带偏置的Sigmoid | $sigmoid(x) = \frac{1}{1+e^{-(x+b)}}$ , where $b = -\log(n)$ | `sigmoid_llama_with_b.py` |
| MySoftmaxLlama | 多头维度Softmax | 在head维度应用softmax | `my_softmax_llama.py` |
| MyLinearLlama | 线性注意力 | 移除激活函数 | `Linear_llama.py` |
| MyLlama | 标准LLaMA | 原始实现 | `modeling_llama.py` |
