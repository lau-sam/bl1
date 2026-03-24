#!/bin/bash
#SBATCH --job-name=bl1-viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.log
#
# Generate BL-1 spike and burst visualizations via SpikeInterface.
#
# Submit:
#   cd ~/dev/bl1 && sbatch scripts/slurm_visualize.sh

set -euo pipefail
cd ~/dev/bl1

echo "============================================"
echo " BL-1 Spike Visualization (SpikeInterface)"
echo " Job ID: $SLURM_JOB_ID"
echo "============================================"

# Install spikeinterface extra if needed
source ~/.local/bin/env 2>/dev/null || true
uv pip show spikeinterface >/dev/null 2>&1 || uv pip install "spikeinterface[full]>=0.101"

# Run visualization script
uv run python scripts/visualize_spikes.py \
    --output-dir figures/ \
    --n-neurons 5000 \
    --duration-s 60 \
    --seed 42

echo ""
echo "Figures saved to ~/dev/bl1/figures/"
ls -la figures/*.png 2>/dev/null
