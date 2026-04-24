#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
    configs/head_softmax.json \
    results/head_softmax \
    eager

