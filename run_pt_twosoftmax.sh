#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_twosoftmax.json \
    results/pt_twosoftmax \
    eager
