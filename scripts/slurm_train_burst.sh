#!/usr/bin/env bash
#SBATCH --job-name=bl1-burst-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/datasets/bl1/results/sharf_2022/burst_test_logs/%x_%j.out
#SBATCH --error=/data/datasets/bl1/results/sharf_2022/burst_test_logs/%x_%j.err
# ---------------------------------------------------------------------------
# Burst-enabled training test: 5K neurons, 5000ms sim, 100 epochs
#
# This script tests burst-enabled training with STP (short-term plasticity).
# Previous training runs produced 0.0 bursts/min because:
#   1. Training ran without STP (no synaptic facilitation/depression)
#   2. Sim duration was only 2s (too short for reliable burst detection)
#
# This script uses a 5s sim duration to allow burst detection and targets
# the 7month_2950 recording which has the highest burst target (8.0/min).
#
# The trainer.py STP integration must be complete before running this.
#
#   sbatch scripts/slurm_train_burst.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/home/mhough/dev/bl1"
VENV="${PROJECT_DIR}/.venv/bin"
DATA_DIR="/data/datasets/bl1/zenodo_sharf_2022"
OUT_DIR="/data/datasets/bl1/results/sharf_2022/burst_test"
LOG_DIR="/data/datasets/bl1/results/sharf_2022/burst_test_logs"

FILE="${DATA_DIR}/7month_2950.raw.h5"

if [[ ! -f "${FILE}" ]]; then
    echo "ERROR: Recording not found: ${FILE}"
    exit 1
fi

BASENAME=$(basename "$FILE" .raw.h5)

echo "============================================================"
echo "BL-1 Burst-Enabled Training Test"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  Recording:  ${BASENAME}"
echo "  Config:     5K neurons, 5000ms sim, 100 epochs"
echo "  Purpose:    Test STP-enabled training for burst matching"
echo "  Date:       $(date)"
echo "============================================================"

mkdir -p "${OUT_DIR}/${BASENAME}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

"${VENV}/python" scripts/train_culture.py \
    --n-neurons 5000 \
    --n-epochs 100 \
    --sim-duration-ms 5000 \
    --lr 5e-5 \
    --surrogate-beta 5.0 \
    --auto-noise \
    --from-recording "${FILE}" \
    --output-dir "${OUT_DIR}/${BASENAME}"

echo ""
echo "Job complete: ${BASENAME}"
echo "============================================================"
