#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_head_softmax_with_b.json \
    results/pt_head_softmax_with_b \
    eager
