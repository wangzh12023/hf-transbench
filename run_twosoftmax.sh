#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
  configs/softmax_and_head_softmax.json \
  results/twosoftmax \
  eager
