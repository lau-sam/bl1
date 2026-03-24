#!/usr/bin/env python3
"""Automated parameter sweep to close the FR convergence gap.

This script is designed to be run by an autoresearch agent. It:
1. Sweeps init_weight_scale and noise multiplier combinations
2. Trains for 30 epochs each (fast iteration)
3. Measures sim_fr / target_fr ratio
4. Saves results to /data with the best config identified
5. Exits with code 0 if any config achieves >70% ratio, 1 otherwise

Usage:
    python scripts/autoresearch_fr_sweep.py
    python scripts/autoresearch_fr_sweep.py --target-fr 0.3
"""

import argparse
import json
import itertools
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("/data/datasets/bl1/results/autoresearch/fr_sweep")


def run_one(target_fr, init_weight_scale, noise_mult, n_neurons=1000, n_epochs=30):
    """Train with given params and return sim_fr / target_fr ratio."""
    from bl1.training.trainer import TrainingConfig, train_weights

    noise_amp = target_fr * noise_mult + 0.3
    noise_amp = max(min(noise_amp, 15.0), 0.3)

    config = TrainingConfig(
        n_neurons=n_neurons,
        n_epochs=n_epochs,
        sim_duration_ms=500.0,
        learning_rate=1e-4,
        target_firing_rate_hz=target_fr,
        target_burst_rate_per_min=0.0,
        surrogate_beta=5.0,
        init_weight_scale=init_weight_scale,
        I_noise_amplitude=noise_amp,
        auto_noise=False,  # we control noise directly
        w_burst_rate=0.0,  # focus on FR only
    )

    result = train_weights(config)
    final = result.loss_history[-1]
    sim_fr = final["mean_fr_hz"]
    ratio = sim_fr / max(target_fr, 0.001)
    return {
        "target_fr": target_fr,
        "sim_fr": sim_fr,
        "ratio": ratio,
        "init_weight_scale": init_weight_scale,
        "noise_mult": noise_mult,
        "noise_amp": noise_amp,
        "final_loss": final["total"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fr", type=float, default=0.3,
                        help="Target firing rate to sweep around")
    parser.add_argument("--n-neurons", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameter grid
    weight_scales = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    noise_mults = [1.0, 1.5, 2.0, 3.0, 5.0]

    targets = [args.target_fr]
    # Also test a few other targets to check generalization
    if args.target_fr < 1.0:
        targets.extend([0.1, 0.5, 0.86])
    else:
        targets.extend([1.0, 2.0, 5.0])
    targets = sorted(set(targets))

    print("=" * 70)
    print("  BL-1 Autoresearch: FR Convergence Sweep")
    print("=" * 70)
    print(f"  Targets:       {targets}")
    print(f"  Weight scales: {weight_scales}")
    print(f"  Noise mults:   {noise_mults}")
    print(f"  Total configs: {len(weight_scales) * len(noise_mults) * len(targets)}")
    print("=" * 70)

    all_results = []
    t0 = time.time()

    for target_fr in targets:
        print(f"\n--- Target FR = {target_fr:.2f} Hz ---")
        for ws, nm in itertools.product(weight_scales, noise_mults):
            r = run_one(target_fr, ws, nm, args.n_neurons, args.n_epochs)
            all_results.append(r)
            marker = "***" if 0.7 <= r["ratio"] <= 1.5 else "   "
            print(f"  {marker} ws={ws:.2f} nm={nm:.1f} → FR={r['sim_fr']:.3f} "
                  f"ratio={r['ratio']:.2f} {marker}")

    # Find best config per target
    print("\n" + "=" * 70)
    print("  Best Configs (closest to ratio=1.0)")
    print("=" * 70)

    best_overall = None
    for target_fr in targets:
        target_results = [r for r in all_results if r["target_fr"] == target_fr]
        # Best = closest ratio to 1.0
        best = min(target_results, key=lambda r: abs(r["ratio"] - 1.0))
        print(f"  FR={target_fr:.2f}: ws={best['init_weight_scale']:.2f} "
              f"nm={best['noise_mult']:.1f} → sim={best['sim_fr']:.3f} "
              f"ratio={best['ratio']:.2f}")
        if best_overall is None or abs(best["ratio"] - 1.0) < abs(best_overall["ratio"] - 1.0):
            best_overall = best

    # Success criteria: any config achieves 70-150% of target
    good_configs = [r for r in all_results if 0.7 <= r["ratio"] <= 1.5]
    success = len(good_configs) > 0

    print(f"\n  Configs within 70-150% of target: {len(good_configs)}/{len(all_results)}")
    if best_overall:
        print(f"  Best overall: ws={best_overall['init_weight_scale']}, "
              f"nm={best_overall['noise_mult']}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  SUCCESS: {'YES' if success else 'NO'}")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"sweep_{timestamp}.json", "w") as f:
        json.dump({
            "results": all_results,
            "best_overall": best_overall,
            "success": success,
            "n_good_configs": len(good_configs),
        }, f, indent=2, default=str)

    print(f"  Saved: {OUTPUT_DIR / f'sweep_{timestamp}.json'}")

    # Exit code signals success to the autoresearch agent
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
