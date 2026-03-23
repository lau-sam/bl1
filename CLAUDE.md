# BL-1 Development Guide

## Project Overview

BL-1 is a JAX-based in-silico cortical culture simulator. It models dissociated cortical neurons on multi-electrode arrays (MEAs) with biologically detailed spiking neurons, conductance-based synapses, four timescales of plasticity, and closed-loop game experiments. The entire simulation loop is JIT-compiled via `jax.lax.scan` and differentiable through surrogate gradients.

## Environment

- **Platform**: DGX Spark (aarch64), NVIDIA GB10 GPU, CUDA 13, JAX 0.9.2
- **Python**: 3.12, venv at `.venv/`
- **NAS**: TrueNAS at `/data` (NFS mount, 41 TB, shared with fedora-legion)
- **Install**: `pip install -e ".[dev]"` then `pip install trackio pynwb dandi optax`

## Commands

```bash
# Tests (must pass before any commit)
make test                              # 536 tests, ~4 min
.venv/bin/pytest tests/test_validation.py -v  # validation framework

# Full validation suite
bash scripts/run_validation.sh --quick  # tests + benchmarks + bio-validation

# Training
python scripts/train_culture.py --from-recording FILE.nwb --n-neurons 5000 --n-epochs 100
python scripts/train_all_sharf.py      # batch training, all 33 recordings

# Dataset analysis
python scripts/analyze_all_datasets.py  # stats for all downloaded data
python scripts/validate_real_data.py    # real vs simulated comparison
```

## Architecture

```
src/bl1/
  core/          # Izhikevich/AdEx neurons, AMPA/NMDA/GABA synapses, integrator
  plasticity/    # STP, STDP, homeostatic, structural
  network/       # Topology, connectivity, Culture factory
  mea/           # Virtual MEA (64-ch, HD-MEA 26K electrodes)
  training/      # Differentiable training loop (trainer.py, loss.py)
  validation/    # Dataset catalog, loaders (NWB/HDF5), comparison framework
  analysis/      # Bursts, criticality, connectivity, information theory
  visualization/ # Raster plots, rates, MEA heatmaps
  games/         # Pong, Doom closed-loop environments
```

## Data on NAS (`/data`)

All large files live on the NAS, visible from both DGX Spark and fedora-legion:

```
/data/datasets/bl1/
  dandi_001611_rat_cortical/   # 2,700 NWB files, 20 GB — rat cortical HD-MEA
  zenodo_sharf_2022/           # 33 HDF5 files, 67 GB — human brain organoid
  osf_dishbrain/               # DishBrain spike data
  results/
    sharf_2022/                # Training results (organized by condition)
      baseline/                #   10 recordings
      development/             #   4 recordings
      drug_dose_response/      #   19 recordings
      trackio/                 #   Experiment tracking logs
      summary_*.csv            #   Spreadsheet-ready results
    dataset_analysis/          # Cross-dataset statistics (JSON)
```

## Validated Parameters (Wagenaar 2006)

These are the calibrated simulation parameters that pass 6/6 bio-validation metrics. Do not change without re-running validation:

- `n_neurons=5000`, `p_max=0.21`, `g_exc=0.12`, `g_inh=0.36`
- AMPA/NMDA split: `nmda_ratio=0.37`
- STP: `U_exc=0.30`, `tau_rec=800ms`
- Burst detection: `threshold_std=1.5`
- Duration: 60s for robust IBI statistics

Config file: `configs/wagenaar_calibrated.yaml`

## Training Pipeline

### How it works

1. Load real recording (NWB or Maxwell HDF5) via `bl1.validation.loaders`
2. Extract targets: firing rate and burst rate from the activity window
3. Build network with `build_connectivity` (sparse BCOO), convert to dense
4. Scale weights down by `init_weight_scale=0.1` (training runs without STP)
5. Forward pass: `simulate(..., surrogate=True)` through `jax.lax.scan`
6. Loss: log-scale FR + differentiable burst proxy + synchrony + weight reg
7. Backward pass: `jax.grad` with SuperSpike surrogate (beta=5.0)
8. Update via Adam + gradient clipping + NaN protection + weight clamping
9. Log to trackio every epoch

### Known Issue: FR Floor

Training converges to ~0.178 Hz regardless of target. Root cause: with `init_weight_scale=0.1` and `I_noise_amplitude=2.0`, the network's minimum sustainable firing rate is ~0.18 Hz. The optimizer shrinks weights (W_exc drops from 0.05 to 0.012) but can't push below this floor.

**Approaches to fix (in priority order):**

1. **Adaptive noise**: Set `I_noise_amplitude = max(target_fr * 2.5, 0.5)`. Low targets need less noise.
2. **Trainable noise**: Add `I_noise_amplitude` and `bg_mean` as learnable parameters with separate LR.
3. **Curriculum**: Start from validated 1.6 Hz params, gradually shift targets toward recording stats.
4. **Two-phase training**: Phase 1 at 500ms for FR convergence, Phase 2 at 5000ms for burst matching.

### Key files

| File | Purpose |
|------|---------|
| `src/bl1/training/trainer.py` | Core training loop, `train_weights()` |
| `src/bl1/training/loss.py` | Loss function components |
| `scripts/train_culture.py` | CLI entry point, `--from-recording` |
| `scripts/train_all_sharf.py` | Batch training with trackio |
| `configs/wagenaar_calibrated.yaml` | Validated simulation parameters |

## Autoresearch Experiment Loop

For autonomous agents iterating on the training pipeline:

```
1. Pick a recording from /data/datasets/bl1/zenodo_sharf_2022/
2. Extract targets: python scripts/analyze_all_datasets.py
3. Hypothesize a parameter change (noise, weight scale, LR, sim duration)
4. Run: python scripts/train_culture.py \
     --from-recording /data/.../recording.h5 \
     --n-neurons 5000 --n-epochs 50 --sim-duration-ms 500
5. Check: does final_fr match target_fr within 20%?
6. If gap > 20%: adjust hypothesis, go to 3
7. If gap < 20%: extend to 100 epochs, try longer sim-duration-ms
8. Run validation: bash scripts/run_validation.sh --quick
9. If passing: commit and push
```

**Guardrails:**
- `make test` must pass (536 tests) before any commit
- Bio-validation must remain 6/6 on Wagenaar metrics
- Never modify `configs/wagenaar_calibrated.yaml` without re-running full validation
- Results go to `/data/datasets/bl1/results/` (NAS), not local `results/`
- Use trackio for all training runs: `trackio.init(project="bl1-...", dir="/data/.../trackio")`

## GPU Performance

| Neurons | Realtime Factor | Notes |
|---------|----------------|-------|
| 1,000 | 15.6x | |
| 5,000 | 8.6x | Validated config |
| 10,000 | 6.2x | |
| 20,000 | 2.3x | |
| 50,000 | 0.24x | Below realtime |
| 100,000 | 0.06x | Needs multi-GPU |

BCOO/cuSPARSE is the fastest sparse matmul at all scales tested. Event-driven CSC kernels (in `pallas_ops.py`) are correct but slower due to JAX dispatch overhead. Path to >100K is multi-GPU sharding, not custom kernels.

## NSG (Supercomputer) Submission

For large-scale jobs on SDSC Expanse GPUs:

```bash
export NSG_USERNAME=... NSG_PASSWORD=... NSG_APPKEY=...
python scripts/nsg_submit.py --list-tools      # available tools
python scripts/nsg_submit.py --submit           # submit job
python scripts/nsg_submit.py --status JOB_ID    # check status
python scripts/nsg_submit.py --download JOB_ID  # get results
```

Tool: `GPU_PY_EXPANSE` (Python on Expanse GPUs, V100s)
