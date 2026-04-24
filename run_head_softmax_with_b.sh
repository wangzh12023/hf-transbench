#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
    configs/head_softmax_with_b.json \
    results/head_softmax_with_b \
    eager

