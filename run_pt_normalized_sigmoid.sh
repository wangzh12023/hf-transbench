#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_normalized_sigmoid.json \
    results/pt_normalized_sigmoid \
    eager
