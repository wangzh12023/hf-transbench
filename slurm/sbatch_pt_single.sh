#!/usr/bin/env bash
#SBATCH --job-name=pt-hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

CONFIG_NAME="${CONFIG_NAME:-configs/pt_96.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/pt_softmax}"

bash run_pt_experiment.sh "${CONFIG_NAME}" "${OUTPUT_DIR}" eager
