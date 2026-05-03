#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_sel_softmax.json \
    results/pt_sel_softmax \
    eager
