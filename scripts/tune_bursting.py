#!/usr/bin/env python3
"""Round 3: Add NMDA to sustain bursts past 100ms."""

import itertools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    nmda_step,
    compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step
from bl1.analysis.bursts import detect_bursts, burst_statistics

N = 5000
DUR_MS = 30_000.0
DT = 0.5
SEED = 42

# Round 4: focused sweep — moderate NMDA (0.35-0.65) for duration + frequency
param_grid = {
    "U_exc":       [0.25, 0.30, 0.35],
    "tau_rec":     [500.0, 800.0],
    "g_exc":       [0.12, 0.15, 0.18],
    "g_inh_ratio": [3.0, 3.5],
    "nmda_ratio":  [0.35, 0.45, 0.55, 0.65],
    "bg_mean":     [1.0],
    "bg_std":      [3.0],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*[param_grid[k] for k in keys]))
print(f"Testing {len(combos)} parameter combinations\n")

header = (
    f"{'U':>5s} {'trec':>5s} {'gE':>5s} {'gI':>5s} {'nmda':>5s} "
    f"{'#bst':>5s} {'dur':>6s} {'rate':>6s} "
    f"{'ibi':>7s} {'cv':>5s} {'rec':>5s} {'FR':>5s} {'t':>4s} {'score':>6s}"
)
print(header)
print("-" * len(header))

results = []

for combo in combos:
    p = dict(zip(keys, combo))
    g_inh = p["g_exc"] * p["g_inh_ratio"]
    nmda_r = p["nmda_ratio"]

    key = jax.random.PRNGKey(SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    positions = place_neurons(k1, N, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, N)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=p["g_exc"], g_inh=g_inh,
    )
    syn = create_synapse_state(N)

    # Split exc weights: AMPA gets (1-nmda_ratio), NMDA gets nmda_ratio
    W_ampa = W_exc * (1.0 - nmda_r)
    W_nmda = W_exc * nmda_r

    # Custom STP params
    U = jnp.where(is_exc, p["U_exc"], 0.04)
    tau_rec = jnp.where(is_exc, p["tau_rec"], 100.0)
    tau_fac = jnp.where(is_exc, 0.001, 1000.0)
    stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
    stp_state = init_stp_state(N, stp_params)

    n_steps = int(DUR_MS / DT)
    I_noise = p["bg_mean"] + p["bg_std"] * jax.random.normal(k4, (n_steps, N))

    def step_fn(carry, I_t):
        ns, ss, stp_st = carry
        I_syn = compute_synaptic_current(ss, ns.v)
        I_total = I_syn + I_t
        ns = izhikevich_step(ns, params, I_total, DT)
        stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, DT)
        new_ampa = ampa_step(ss.g_ampa, scale, W_ampa, DT)
        new_gaba = gaba_a_step(ss.g_gaba_a, scale, W_inh, DT)
        new_nmda_rise, new_nmda_decay, _ = nmda_step(
            ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT
        )
        ss = SynapseState(
            g_ampa=new_ampa, g_gaba_a=new_gaba,
            g_nmda_rise=new_nmda_rise, g_nmda_decay=new_nmda_decay,
            g_gaba_b_rise=ss.g_gaba_b_rise, g_gaba_b_decay=ss.g_gaba_b_decay,
        )
        return (ns, ss, stp_st), ns.spikes

    t0 = time.perf_counter()
    (_, _, _), spikes = jax.lax.scan(step_fn, (state, syn, stp_state), I_noise)
    spikes.block_until_ready()
    wall = time.perf_counter() - t0

    raster = np.asarray(spikes)
    total_time_s = n_steps * DT / 1000.0
    fr_hz = float(raster.sum()) / (N * total_time_s)

    bursts = detect_bursts(raster, dt_ms=DT, threshold_std=2.0, min_duration_ms=50.0)
    bs = burst_statistics(bursts)
    n_bursts = len(bursts)
    burst_rate = n_bursts / (total_time_s / 60.0)

    dur_s = f"{bs['duration_mean']:.0f}" if not math.isnan(bs['duration_mean']) else "N/A"
    ibi_s = f"{bs['ibi_mean']:.0f}" if not math.isnan(bs['ibi_mean']) else "N/A"
    cv_s = f"{bs['ibi_cv']:.2f}" if not math.isnan(bs['ibi_cv']) else "N/A"
    rec_s = f"{bs['recruitment_mean']:.2f}" if not math.isnan(bs['recruitment_mean']) else "N/A"

    # Score against Wagenaar ranges
    in_range = 0
    total_check = 0
    checks = {
        "fr": (fr_hz, 0.1, 5.0),
        "rate": (burst_rate, 0.2, 20.0),
        "dur": (bs["duration_mean"], 100.0, 2000.0),
        "ibi": (bs["ibi_mean"], 3000.0, 300000.0),
        "cv": (bs["ibi_cv"], 0.3, 2.0),
        "rec": (bs["recruitment_mean"], 0.1, 0.95),
    }
    detail = []
    for name, (val, lo, hi) in checks.items():
        if not math.isnan(val):
            total_check += 1
            ok = lo <= val <= hi
            if ok:
                in_range += 1
            detail.append(f"{name}:{'OK' if ok else 'X'}")

    print(
        f"{p['U_exc']:5.2f} {p['tau_rec']:5.0f} {p['g_exc']:5.2f} {g_inh:5.2f} {nmda_r:5.2f} "
        f"{n_bursts:5d} {dur_s:>6s} {burst_rate:6.1f} "
        f"{ibi_s:>7s} {cv_s:>5s} {rec_s:>5s} {fr_hz:5.1f} {wall:4.1f} "
        f"[{in_range}/{total_check}] {' '.join(detail)}"
    )

    results.append({**p, "g_inh": g_inh, "n_bursts": n_bursts,
                     "dur": bs["duration_mean"], "burst_rate": burst_rate,
                     "ibi": bs["ibi_mean"], "ibi_cv": bs["ibi_cv"],
                     "recruit": bs["recruitment_mean"], "fr_hz": fr_hz,
                     "score": in_range, "total": total_check})

# Print best results
print("\n=== TOP 10 BY SCORE ===")
results.sort(key=lambda r: (-r["score"], -r["n_bursts"]))
for r in results[:10]:
    dur_s = f"{r['dur']:.0f}" if not math.isnan(r['dur']) else "N/A"
    print(f"  U={r['U_exc']:.2f} trec={r['tau_rec']:.0f} gE={r['g_exc']:.2f} "
          f"gI={r['g_inh']:.2f} nmda={r['nmda_ratio']:.1f} → {r['score']}/{r['total']} "
          f"{r['n_bursts']}bst dur={dur_s}ms rate={r['burst_rate']:.1f}/min fr={r['fr_hz']:.1f}Hz")
