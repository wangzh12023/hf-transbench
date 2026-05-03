#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_sigmoid.json \
    results/pt_sigmoid \
    eager
