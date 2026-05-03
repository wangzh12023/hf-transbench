#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_head_softmax.json \
    results/pt_head_softmax \
    eager
