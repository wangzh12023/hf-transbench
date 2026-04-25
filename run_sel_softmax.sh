#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
    configs/sel_softmax.json \
    results/sel_softmax \
    eager
