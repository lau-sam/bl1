#!/usr/bin/env python3
"""Train one BL-1 weight set against pooled statistics from multiple recordings.

Tests whether BL-1 captures a condition's dynamics rather than one recording's noise.

Usage:
    python scripts/train_pooled.py --condition baseline
    python scripts/train_pooled.py --condition development
    python scripts/train_pooled.py --condition drug_control
    python scripts/train_pooled.py --condition all
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SHARF_DIR = Path("/data/datasets/bl1/zenodo_sharf_2022")
OUTPUT_BASE = Path("/data/datasets/bl1/results/sharf_2022/pooled")

CONDITION_PATTERNS = {
    "baseline": "7month_*.raw.h5",
    "development": "Development_*.raw.h5",
    "drug_control": "Drug_*_control*.raw.h5",
    "drug_3uM": "Drug_*_3uM*.raw.h5",
    "drug_10uM": "Drug_*_10uM*.raw.h5",
    "drug_30uM": "Drug_*_30uM*.raw.h5",
    "drug_50uM": "Drug_*_50uM*.raw.h5",
    "all": "*.raw.h5",
}


def extract_targets(filepath):
    """Load recording and extract targets from activity window."""
    from bl1.validation.loaders import load_maxwell_h5, compute_recording_statistics

    data = load_maxwell_h5(str(filepath))
    if data["n_units"] < 5:
        return None
    active = [st for st in data["spike_times"] if len(st) > 0]
    if not active:
        return None
    t_min = float(min(st.min() for st in active))
    t_max = float(max(st.max() for st in active))
    dur = t_max - t_min
    if dur < 5:
        return None
    use_dur = min(dur, 120.0)
    trimmed = {
        "spike_times": [st[(st >= t_min) & (st <= t_min + use_dur)] - t_min
                        for st in data["spike_times"]],
        "duration_s": use_dur,
        "n_units": data["n_units"],
    }
    stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
    return {
        "filename": filepath.name,
        "target_fr": stats["mean_firing_rate_hz"],
        "target_burst": stats["burst_rate_per_min"],
        "n_units": data["n_units"],
        "duration_s": use_dur,
    }


def main():
    parser = argparse.ArgumentParser(description="Pooled training across recordings")
    parser.add_argument("--condition", required=True, choices=list(CONDITION_PATTERNS.keys()))
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=100)
    args = parser.parse_args()

    pattern = CONDITION_PATTERNS[args.condition]
    files = sorted(SHARF_DIR.glob(pattern))
    if not files:
        print(f"No files matching {pattern} in {SHARF_DIR}")
        sys.exit(1)

    print(f"Condition: {args.condition} ({len(files)} files)")

    # Extract per-recording targets
    all_targets = []
    for f in files:
        t = extract_targets(f)
        if t:
            all_targets.append(t)
            print(f"  {f.name}: FR={t['target_fr']:.3f} Hz, burst={t['target_burst']:.1f}/min")

    if not all_targets:
        print("No valid recordings")
        sys.exit(1)

    # Pool: use median (robust to outliers)
    pooled_fr = float(np.median([t["target_fr"] for t in all_targets]))
    pooled_burst = float(np.median([t["target_burst"] for t in all_targets]))
    print(f"\nPooled targets (median of {len(all_targets)} recordings):")
    print(f"  FR: {pooled_fr:.3f} Hz")
    print(f"  Burst rate: {pooled_burst:.1f}/min")

    # Train
    from bl1.training.trainer import TrainingConfig, train_weights
    import trackio

    output_dir = OUTPUT_BASE / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = None
    try:
        tracker = trackio.init(
            project="bl1-sharf-pooled",
            name=f"pooled/{args.condition}",
            dir=str(OUTPUT_BASE / "trackio"),
            config={"condition": args.condition, "n_recordings": len(all_targets),
                    "pooled_fr": pooled_fr, "pooled_burst": pooled_burst},
        )
    except Exception as e:
        print(f"Trackio init failed: {e}")

    config = TrainingConfig(
        n_neurons=args.n_neurons,
        n_epochs=args.n_epochs,
        sim_duration_ms=500.0,
        learning_rate=1e-4,
        target_firing_rate_hz=pooled_fr,
        target_burst_rate_per_min=pooled_burst,
        surrogate_beta=5.0,
        auto_noise=True,
    )

    result = train_weights(config, tracker=tracker)

    # Save
    np.savez_compressed(output_dir / "trained_weights.npz",
                        W_exc=np.array(result.W_exc), W_inh=np.array(result.W_inh))
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(result.loss_history, f, indent=2, default=str)

    # Validation: how well do pooled weights match each individual recording?
    final = result.loss_history[-1]
    validation = []
    for t in all_targets:
        gap = abs(final["mean_fr_hz"] - t["target_fr"]) / max(t["target_fr"], 0.01)
        validation.append({
            "filename": t["filename"],
            "target_fr": t["target_fr"],
            "sim_fr": final["mean_fr_hz"],
            "gap_pct": gap * 100,
        })

    with open(output_dir / "pooled_targets.json", "w") as f:
        json.dump({"pooled_fr": pooled_fr, "pooled_burst": pooled_burst,
                    "per_recording": all_targets}, f, indent=2, default=str)
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    if tracker:
        try:
            trackio.finish()
        except Exception:
            pass

    print(f"\nResults saved to {output_dir}")
    print(f"Pooled FR target: {pooled_fr:.3f} Hz → Achieved: {final['mean_fr_hz']:.3f} Hz")
    gaps = [v["gap_pct"] for v in validation]
    print(f"Per-recording gap: mean={np.mean(gaps):.1f}%, max={np.max(gaps):.1f}%")


if __name__ == "__main__":
    main()
