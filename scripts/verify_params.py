#!/usr/bin/env python3
"""Final verification: 60s, thresh=1.5, NMDA 0.35/0.37/0.40."""

import math, time as _time
import jax, jax.numpy as jnp, numpy as np
from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (SynapseState, create_synapse_state,
    ampa_step, gaba_a_step, nmda_step, compute_synaptic_current)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.validation.datasets import compare_statistics

N, DUR_MS, DT = 5000, 60_000.0, 0.5
P_MAX, U_EXC, TAU_REC = 0.21, 0.30, 800.0
G_EXC, G_INH = 0.12, 0.36
BG_MEAN, BG_STD, THRESH = 1.0, 3.0, 1.5
SEEDS = [42, 123, 7, 2024, 9999, 314, 271, 55]

for NMDA_R in [0.35, 0.37, 0.40]:
    print(f"\n{'='*78}")
    print(f"NMDA={NMDA_R} | 60s | thresh={THRESH}")
    print(f"{'='*78}")
    scores = []
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        positions = place_neurons(k1, N, (3000.0, 3000.0))
        params, state, is_exc = create_population(k2, N)
        W_exc, W_inh, _ = build_connectivity(k3, positions, is_exc,
            lambda_um=200.0, p_max=P_MAX, g_exc=G_EXC, g_inh=G_INH)
        syn = create_synapse_state(N)
        W_ampa, W_nmda = W_exc * (1.0 - NMDA_R), W_exc * NMDA_R
        U = jnp.where(is_exc, U_EXC, 0.04)
        tr = jnp.where(is_exc, float(TAU_REC), 100.0)
        tf = jnp.where(is_exc, 0.001, 1000.0)
        stp_params = STPParams(U=U, tau_rec=tr, tau_fac=tf)
        stp_state = init_stp_state(N, stp_params)
        n_steps = int(DUR_MS / DT)
        I_noise = BG_MEAN + BG_STD * jax.random.normal(k4, (n_steps, N))

        def step_fn(carry, I_t):
            ns, ss, st = carry
            ns = izhikevich_step(ns, params, compute_synaptic_current(ss, ns.v) + I_t, DT)
            st, scale = stp_step(st, stp_params, ns.spikes, DT)
            nr, nd, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
            ss = SynapseState(ampa_step(ss.g_ampa, scale, W_ampa, DT),
                              gaba_a_step(ss.g_gaba_a, scale, W_inh, DT),
                              nr, nd, ss.g_gaba_b_rise, ss.g_gaba_b_decay)
            return (ns, ss, st), ns.spikes

        t0 = _time.perf_counter()
        (_, _, _), spikes = jax.lax.scan(step_fn, (state, syn, stp_state), I_noise)
        spikes.block_until_ready()
        wall = _time.perf_counter() - t0

        raster = np.asarray(spikes)
        total_s = n_steps * DT / 1000.0
        fr = float(raster.sum()) / (N * total_s)
        bursts = detect_bursts(raster, dt_ms=DT, threshold_std=THRESH, min_duration_ms=50.0)
        bs = burst_statistics(bursts)
        rate = len(bursts) / (total_s / 60.0)
        sim = {"mean_firing_rate_hz": fr, "burst_rate_per_min": rate,
               "burst_duration_mean_ms": bs["duration_mean"],
               "ibi_mean_ms": bs["ibi_mean"], "ibi_cv": bs["ibi_cv"],
               "recruitment_mean": bs["recruitment_mean"]}
        res = compare_statistics(sim, "wagenaar_2006")
        ok = sum(1 for r in res.values() if r["in_range"] is True)
        tot = sum(1 for r in res.values() if r["in_range"] is not None)
        fails = [k.replace("_mean_ms","").replace("_per_min","").replace("_mean","").replace("_hz","")
                 for k, r in res.items() if r["in_range"] is False]
        d = f"{bs['duration_mean']:.0f}" if not math.isnan(bs['duration_mean']) else "N/A"
        scores.append(ok)
        print(f"  seed={seed:5d} #bst={len(bursts):2d} dur={d:>4s}ms rate={rate:5.1f}/m "
              f"ibi={bs['ibi_mean']:8.0f}ms cv={bs['ibi_cv']:5.2f} rec={bs['recruitment_mean']:.2f} "
              f"FR={fr:.1f}Hz {wall:.1f}s [{ok}/{tot}] {' '.join(fails)}")

    n6 = sum(1 for s in scores if s >= 6)
    n5 = sum(1 for s in scores if s >= 5)
    print(f"  >> 6/6: {n6}/{len(scores)}  5+/6: {n5}/{len(scores)}  mean: {sum(scores)/len(scores):.1f}")
