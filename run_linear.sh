#!/usr/bin/env bash
set -euo pipefail

bash run_experiment.sh \
	configs/my_llama_tiny_linear.json \
	results/linear \
	eager
