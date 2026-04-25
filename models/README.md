## Model Overview

| Model Type | Normalization / Variant | File |
|---|---|---|
| `my-llama` | Standard LLaMA attention | `modeling_llama.py` |
| `my-llama-sigmoid` | `sigmoid(score)` | `sigmoid_llama.py` |
| `my-llama-sigmoid-with-b` | `sigmoid(score - log(n))` | `sigmoid_llama_with_b.py` |
| `my-llama-linear` | raw score (no normalization) | `Linear_llama.py` |
| `my-llama-head-softmax` | softmax over GQA head-group dimension | `head_softmax_llama.py` |
| `my-llama-sel-softmax` | standard sequence softmax through custom attention path | `sel_softmax_llama.py` |
| `head-softmax-with-b` | sigmoid+b gate then head-group softmax | `head_softmax_with_b.py` |
| `softmax-and-head-softmax` | sequence softmax then head-group softmax | `twosoftmax_llama.py` |

## Architecture Notes

- Shared tensor/mask logic is in `attention_utils.py`.
- Shared model wrappers are in `custom_llama_base.py`.
- Custom manual-attention variants are intended for `eager` mode.
