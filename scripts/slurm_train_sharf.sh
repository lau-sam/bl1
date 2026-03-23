#!/usr/bin/env bash
#SBATCH --job-name=bl1-sharf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/datasets/bl1/results/sharf_2022/slurm_logs/%x_%A_%a.out
#SBATCH --error=/data/datasets/bl1/results/sharf_2022/slurm_logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Submit as array job:
#   sbatch --array=0-32 scripts/slurm_train_sharf.sh
#
# Or submit one recording:
#   sbatch scripts/slurm_train_sharf.sh 0
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/home/mhough/dev/bl1"
VENV="${PROJECT_DIR}/.venv/bin"
DATA_DIR="/data/datasets/bl1/zenodo_sharf_2022"

# Get the list of files
mapfile -t FILES < <(ls -1 "${DATA_DIR}"/*.raw.h5 2>/dev/null | sort)

# Determine which file to process
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX="${SLURM_ARRAY_TASK_ID}"
elif [[ $# -ge 1 ]]; then
    IDX="$1"
else
    echo "Usage: sbatch --array=0-$((${#FILES[@]}-1)) scripts/slurm_train_sharf.sh"
    echo "   or: sbatch scripts/slurm_train_sharf.sh <index>"
    exit 1
fi

if [[ $IDX -ge ${#FILES[@]} ]]; then
    echo "Index $IDX out of range (${#FILES[@]} files)"
    exit 1
fi

FILE="${FILES[$IDX]}"
BASENAME=$(basename "$FILE" .raw.h5)

echo "============================================================"
echo "BL-1 Slurm Training Job"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  Array ID:   ${SLURM_ARRAY_TASK_ID:-$IDX}"
echo "  File:       ${FILE}"
echo "  Recording:  ${BASENAME}"
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

"${VENV}/python" scripts/train_culture.py \
    --n-neurons 5000 \
    --n-epochs 100 \
    --sim-duration-ms 500 \
    --lr 1e-4 \
    --surrogate-beta 5.0 \
    --auto-noise \
    --from-recording "${FILE}" \
    --output-dir "/data/datasets/bl1/results/sharf_2022/slurm/${BASENAME}"

echo ""
echo "Job complete: ${BASENAME}"
echo "============================================================"
