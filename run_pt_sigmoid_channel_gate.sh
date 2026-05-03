#!/usr/bin/env bash
set -euo pipefail

bash run_pt_experiment.sh \
    configs/pt_96_sigmoid_channel_gate.json \
    results/pt_sigmoid_channel_gate \
    eager
