#!/usr/bin/env bash
#SBATCH --job-name=bl1-hifi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/datasets/bl1/results/sharf_2022/hifi_logs/%x_%A_%a.out
#SBATCH --error=/data/datasets/bl1/results/sharf_2022/hifi_logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# High-fidelity training: 5K neurons, 2000ms sim, 200 epochs
#
#   sbatch --array=0-9 scripts/slurm_train_hifi.sh    # 10 baseline recordings
#   sbatch scripts/slurm_train_hifi.sh 0               # single recording
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/home/mhough/dev/bl1"
VENV="${PROJECT_DIR}/.venv/bin"
DATA_DIR="/data/datasets/bl1/zenodo_sharf_2022"
OUT_DIR="/data/datasets/bl1/results/sharf_2022/hifi"

# Use baseline recordings (best-characterized, ~0.3-0.86 Hz)
mapfile -t FILES < <(ls -1 "${DATA_DIR}"/7month_*.raw.h5 2>/dev/null | sort)

IDX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
if [[ $IDX -ge ${#FILES[@]} ]]; then
    echo "Index $IDX out of range (${#FILES[@]} files)"
    exit 1
fi

FILE="${FILES[$IDX]}"
BASENAME=$(basename "$FILE" .raw.h5)

echo "============================================================"
echo "BL-1 High-Fidelity Training"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  Recording:  ${BASENAME}"
echo "  Config:     5K neurons, 2000ms sim, 200 epochs"
echo "  Date:       $(date)"
echo "============================================================"

mkdir -p "${OUT_DIR}/${BASENAME}"

cd "${PROJECT_DIR}"

"${VENV}/python" scripts/train_culture.py \
    --n-neurons 5000 \
    --n-epochs 200 \
    --sim-duration-ms 2000 \
    --lr 5e-5 \
    --surrogate-beta 5.0 \
    --auto-noise \
    --from-recording "${FILE}" \
    --output-dir "${OUT_DIR}/${BASENAME}"

echo ""
echo "Job complete: ${BASENAME}"
echo "============================================================"
