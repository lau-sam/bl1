#!/usr/bin/env python3
"""Focused FR convergence training with adaptive LR.

Instead of fixed LR, this uses a simple feedback loop:
  - If FR > target: decrease weights (multiply by 0.99)
  - If FR < target: increase weights (multiply by 1.01)
  - Plus gradient-based refinement at small LR

This is essentially coordinate descent on the weight scale,
combined with gradient-based fine-tuning on the shape.
Much faster convergence than pure gradient descent for FR matching.

Usage:
    python scripts/train_converge.py --target-fr 0.5
    python scripts/train_converge.py --from-recording FILE.h5
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fr", type=float, default=None)
    parser.add_argument("--target-burst", type=float, default=0.0)
    parser.add_argument("--from-recording", type=str, default=None)
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--sim-ms", type=float, default=2000.0)
    parser.add_argument("--output-dir", type=str, default="/data/datasets/bl1/results/converge")
    args = parser.parse_args()

    # Extract targets from recording if provided
    target_fr = args.target_fr
    if args.from_recording:
        from bl1.validation.loaders import load_maxwell_h5, load_nwb_spike_trains, compute_recording_statistics
        ext = args.from_recording.rsplit(".", 1)[-1].lower()
        if ext == "nwb":
            rec = load_nwb_spike_trains(args.from_recording)
        else:
            rec = load_maxwell_h5(args.from_recording)
        if rec["duration_s"] > 86400:
            sr = rec.get("sampling_rate", 20000.0)
            rec["spike_times"] = [st / sr for st in rec["spike_times"]]
            rec["duration_s"] /= sr
        active = [st for st in rec["spike_times"] if len(st) > 0]
        if active:
            t_min = float(min(st.min() for st in active))
            use_dur = min(float(max(st.max() for st in active)) - t_min, 120.0)
            trimmed = {
                "spike_times": [st[(st >= t_min) & (st <= t_min + use_dur)] - t_min
                                for st in rec["spike_times"]],
                "duration_s": use_dur, "n_units": rec["n_units"],
            }
            stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
            target_fr = stats["mean_firing_rate_hz"]
            print(f"From recording: FR={target_fr:.3f} Hz")

    if target_fr is None:
        target_fr = 0.5

    from bl1.core.izhikevich import create_population, izhikevich_step
    from bl1.core.synapses import (SynapseState, create_synapse_state,
        ampa_step, gaba_a_step, nmda_step, compute_synaptic_current)
    from bl1.network.topology import place_neurons, build_connectivity
    from bl1.plasticity.stp import STPParams, init_stp_state, stp_step

    N = args.n_neurons
    DT = 0.5
    DUR = args.sim_ms
    NMDA_R = 0.37
    G_EXC, G_INH = 0.12, 0.36
    U_EXC, TAU_REC = 0.30, 800.0

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    positions = place_neurons(k1, N, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, N)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=G_EXC, g_inh=G_INH)
    syn = create_synapse_state(N)
    W_ampa = W_exc * (1.0 - NMDA_R)
    W_nmda = W_exc * NMDA_R

    U = jnp.where(is_exc, U_EXC, 0.04)
    tau_rec = jnp.where(is_exc, TAU_REC, 100.0)
    tau_fac = jnp.where(is_exc, 0.001, 1000.0)
    stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
    stp_state = init_stp_state(N, stp_params)

    n_steps = int(DUR / DT)

    def simulate_once(W_ampa, W_nmda, W_inh, noise_key):
        I_noise = jax.random.normal(noise_key, (n_steps, N))
        def step_fn(carry, I_t):
            ns, ss, st = carry
            ns = izhikevich_step(ns, params, compute_synaptic_current(ss, ns.v) + I_t, DT)
            st, scale = stp_step(st, stp_params, ns.spikes, DT)
            nr, nd, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
            ss = SynapseState(
                ampa_step(ss.g_ampa, scale, W_ampa, DT),
                gaba_a_step(ss.g_gaba_a, scale, W_inh, DT),
                nr, nd, ss.g_gaba_b_rise, ss.g_gaba_b_decay)
            return (ns, ss, st), ns.spikes
        (_, _, _), spikes = jax.lax.scan(step_fn, (state, syn, stp_state), I_noise)
        return spikes

    # Phase 1: Binary search on noise amplitude to find the right operating point
    print(f"Target FR: {target_fr:.3f} Hz")
    print(f"Phase 1: Finding noise amplitude via binary search...")

    noise_lo, noise_hi = 0.1, 20.0
    best_noise = 1.0
    best_ratio = 0.0

    sim_jit = jax.jit(simulate_once)
    # Warmup JIT
    _ = sim_jit(W_ampa, W_nmda, W_inh, jax.random.PRNGKey(0))

    for step in range(20):
        noise_mid = (noise_lo + noise_hi) / 2.0
        I_scale = noise_mid
        spikes = sim_jit(W_ampa * 1.0, W_nmda * 1.0, W_inh * 1.0,
                         jax.random.PRNGKey(step))
        # Scale noise by rerunning with scaled input
        I_noise_scaled = noise_mid * jax.random.normal(jax.random.PRNGKey(step), (n_steps, N))
        def step_fn_scaled(carry, I_t):
            ns, ss, st = carry
            ns = izhikevich_step(ns, params, compute_synaptic_current(ss, ns.v) + I_t, DT)
            st, scale = stp_step(st, stp_params, ns.spikes, DT)
            nr, nd, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
            ss = SynapseState(
                ampa_step(ss.g_ampa, scale, W_ampa, DT),
                gaba_a_step(ss.g_gaba_a, scale, W_inh, DT),
                nr, nd, ss.g_gaba_b_rise, ss.g_gaba_b_decay)
            return (ns, ss, st), ns.spikes
        (_, _, _), spikes = jax.lax.scan(step_fn_scaled, (state, syn, stp_state), I_noise_scaled)
        spikes = spikes.block_until_ready()

        raster = np.asarray(spikes)
        fr = float(raster.sum()) / (N * DUR / 1000.0)
        ratio = fr / max(target_fr, 0.001)

        if abs(ratio - 1.0) < abs(best_ratio - 1.0):
            best_noise = noise_mid
            best_ratio = ratio

        if fr > target_fr:
            noise_hi = noise_mid
        else:
            noise_lo = noise_mid

        print(f"  step {step:2d}: noise={noise_mid:.3f} → FR={fr:.3f} Hz (ratio={ratio:.2f})")

        if 0.9 <= ratio <= 1.1:
            print(f"  Converged at noise={noise_mid:.3f}")
            break

    # Phase 2: Fine-tune with weight scaling
    print(f"\nPhase 2: Fine-tuning weight scale...")
    scale_lo, scale_hi = 0.5, 2.0
    best_scale = 1.0
    best_fr = fr

    for step in range(15):
        scale_mid = (scale_lo + scale_hi) / 2.0
        I_noise_final = best_noise * jax.random.normal(jax.random.PRNGKey(100 + step), (n_steps, N))
        W_a_s = W_ampa * scale_mid
        W_n_s = W_nmda * scale_mid
        W_i_s = W_inh * scale_mid

        def step_fn_s(carry, I_t):
            ns, ss, st = carry
            ns = izhikevich_step(ns, params, compute_synaptic_current(ss, ns.v) + I_t, DT)
            st, scale = stp_step(st, stp_params, ns.spikes, DT)
            nr, nd, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_n_s, DT)
            ss = SynapseState(
                ampa_step(ss.g_ampa, scale, W_a_s, DT),
                gaba_a_step(ss.g_gaba_a, scale, W_i_s, DT),
                nr, nd, ss.g_gaba_b_rise, ss.g_gaba_b_decay)
            return (ns, ss, st), ns.spikes
        (_, _, _), spikes = jax.lax.scan(step_fn_s, (state, syn, stp_state), I_noise_final)
        spikes = spikes.block_until_ready()

        raster = np.asarray(spikes)
        fr = float(raster.sum()) / (N * DUR / 1000.0)
        ratio = fr / max(target_fr, 0.001)

        if abs(fr - target_fr) < abs(best_fr - target_fr):
            best_scale = scale_mid
            best_fr = fr

        if fr > target_fr:
            scale_hi = scale_mid
        else:
            scale_lo = scale_mid

        print(f"  step {step:2d}: scale={scale_mid:.4f} → FR={fr:.3f} Hz (ratio={ratio:.2f})")

        if 0.95 <= ratio <= 1.05:
            print(f"  Converged at scale={scale_mid:.4f}")
            break

    final_ratio = best_fr / max(target_fr, 0.001)
    print(f"\n{'='*60}")
    print(f"Result: target={target_fr:.3f} Hz, achieved={best_fr:.3f} Hz, ratio={final_ratio:.0%}")
    print(f"Config: noise={best_noise:.3f}, weight_scale={best_scale:.4f}")
    print(f"{'='*60}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "converge_result.json", "w") as f:
        json.dump({
            "target_fr": target_fr,
            "achieved_fr": best_fr,
            "ratio": final_ratio,
            "noise_amp": best_noise,
            "weight_scale": best_scale,
            "n_neurons": N,
            "sim_ms": DUR,
        }, f, indent=2)

    return 0 if 0.8 <= final_ratio <= 1.2 else 1

if __name__ == "__main__":
    exit(main())
