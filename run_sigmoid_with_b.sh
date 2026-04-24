#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
    configs/my_llama_tiny_sigmoid_with_b.json \
    results/sigmoid_with_b \
    eager
    
