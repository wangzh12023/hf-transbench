# HF Starter: Sigmoid/Head-Dim Attention Experiments

This repo is organized for experimenting with LLaMA-style CLM training where attention normalization is changed from sequence softmax to alternative forms (sigmoid, head-dim softmax, and hybrids), including GQA-aware variants.

## 1. Environment (cluster/Linux)

Your local Windows machine has no torch environment, so run training on cluster/Linux nodes.

```bash
conda create -n hf-attn python=3.10 -y
conda activate hf-attn
pip install -r requirements.txt
```

Optional quick GPU check:

```bash
bash test_gpu.sh
```

## 2. Attention Implementation Compatibility

Important:

- `my-llama` (baseline wrapper) can use `eager`, `sdpa`, or `flash_attention_2` (if CUDA + flash-attn are available).
- Custom manual attention variants are now forced to `eager` in [run_clm.py](run_clm.py). This avoids mask/API mismatch and causal bugs with flash/sdpa dispatch.

Custom eager-only model types:

- `my-llama-sigmoid`
- `my-llama-sigmoid-with-b`
- `my-llama-linear`
- `my-llama-head-softmax`
- `head-softmax-with-b`
- `softmax-and-head-softmax`

## 3. One-Command Training

All variant scripts call the shared launcher [run_experiment.sh](run_experiment.sh).

```bash
# Baseline softmax
bash run_softmax.sh

# Sigmoid attention
bash run_sigmoid.sh

# Sigmoid + bias b=-log(n)
bash run_sigmoid_with_b.sh

# Linear attention (no softmax/sigmoid normalization)
bash run_linear.sh

# Head-group softmax (GQA-aware)
bash run_head_softmax.sh

# Head-group softmax with sigmoid+b gate
bash run_head_softmax_with_b.sh

# Sequence softmax + head-group softmax
bash run_twosoftmax.sh
```

Generic form:

```bash
bash run_experiment.sh <config_json> <output_dir> [attn_implementation]
```

Example:

```bash
bash run_experiment.sh configs/my_llama_tiny_sigmoid.json results/sigmoid eager
```

## 4. Correctness Smoke Test

Run before long jobs:

```bash
python scripts/smoke_test_models.py
```

This checks all custom variants can instantiate and run one forward pass under `eager`.

## 5. Config Notes

- [configs/head_softmax.json](configs/head_softmax.json) now uses `num_key_value_heads=4` (with 8 attention heads), so head-group softmax is non-trivial under GQA.
- Keep `num_attention_heads % num_key_value_heads == 0` for GQA-based head grouping.

## 6. Suggested Experiment Order

1. Sanity baseline:

- Run `softmax` and `sigmoid` with identical hyperparameters.
- Confirm train loss and eval perplexity trend are stable.

2. Bias effects:

- Compare `sigmoid` vs `sigmoid_with_b`.
- Track average attention magnitude to see whether `b=-log(n)` stabilizes scaling.

3. Head-dim normalization under GQA:

- Compare `head_softmax`, `head_softmax_with_b`, `twosoftmax`.
- Sweep `num_key_value_heads` (e.g. 8, 4, 2) to control head-group size.

4. Efficiency and scaling:

- Measure tokens/s, GPU memory, convergence speed to target perplexity.

## 7. What To Log for a Paper

Recommended logging dimensions beyond loss/perplexity:

- Attention statistics per layer/head:
  - mean, std, max, entropy
  - sparsity proxy (fraction below threshold)
- Gradient norm and update norm:
  - attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Stability signals:
  - activation norm drift by layer
  - NaN/Inf checks
- Efficiency:
  - throughput (tokens/s)
  - peak memory

For visualization, start with:

- Heatmaps of attention matrices at selected layers/heads.
- Head-group distribution plots for GQA variants.
- Training-time trajectories of attention entropy.

## 8. Practical Paper Directions

Useful research axes:

- Normalization family: softmax vs sigmoid vs linear vs hybrid.
- GQA interaction: how grouping size changes behavior/performance.
- Length generalization: train short, eval longer context.
- Optimization robustness: sensitivity to LR/warmup and init scale.
- Interpretability: whether variants induce different head specialization patterns.

## 9. File Map

- Training entry: [run_clm.py](run_clm.py)
- Custom models: [models](models)
- Configs: [configs](configs)
- Variant runners: `run_*.sh`
- Shared runner: [run_experiment.sh](run_experiment.sh)
- Smoke test: [scripts/smoke_test_models.py](scripts/smoke_test_models.py)
