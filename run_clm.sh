#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
    configs/my_llama_tiny.json \
    results/softmax \
    eager
    
