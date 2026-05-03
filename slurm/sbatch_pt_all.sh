#!/usr/bin/env bash
#SBATCH --job-name=pt-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-10
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

VARIANTS=(
  softmax
  sel_softmax
  sigmoid
  sigmoid_with_b
  normalized_sigmoid
  linear
  channel_gate
  sigmoid_channel_gate
  head_softmax
  head_softmax_with_b
  twosoftmax
)

CONFIGS=(
  configs/pt_96.json
  configs/pt_96_sel_softmax.json
  configs/pt_96_sigmoid.json
  configs/pt_96_sigmoid_with_b.json
  configs/pt_96_normalized_sigmoid.json
  configs/pt_96_linear.json
  configs/pt_96_channel_gate.json
  configs/pt_96_sigmoid_channel_gate.json
  configs/pt_96_head_softmax.json
  configs/pt_96_head_softmax_with_b.json
  configs/pt_96_twosoftmax.json
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[${TASK_ID}]}"
CONFIG_NAME="${CONFIGS[${TASK_ID}]}"
OUTPUT_DIR="${OUTPUT_ROOT:-results}/pt_${VARIANT}"

bash run_pt_experiment.sh "${CONFIG_NAME}" "${OUTPUT_DIR}" eager
