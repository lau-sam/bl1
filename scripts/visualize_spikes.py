"""Generate spiking and burst visualizations from BL-1 using SpikeInterface.

Runs a 60-second Wagenaar-calibrated cortical culture simulation, converts
the output to SpikeInterface format, and produces standard neuroscience
visualizations:

  1. Spike raster plot (full + zoomed burst window)
  2. Population firing rate with burst overlay
  3. ISI distribution (log scale)
  4. Unit firing rate distribution
  5. Auto-/cross-correlograms
  6. Amplitude distribution per unit

Usage:
    uv run python scripts/visualize_spikes.py [--output-dir figures/]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.core as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

from bl1.analysis.bursts import burst_statistics, detect_bursts
from bl1.core.izhikevich import NeuronState, create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
)
from bl1.network.topology import build_connectivity, place_neurons


def run_simulation(
    n_neurons: int = 5000,
    duration_ms: float = 60_000,
    dt: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run Wagenaar-calibrated simulation, return raster and metadata."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Wagenaar-calibrated parameters
    g_exc, g_inh = 0.12, 0.36
    p_max, lambda_um = 0.21, 200.0
    bg_mean, bg_std = 3.5, 2.5

    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _delays = build_connectivity(
        k3, positions, is_exc,
        lambda_um=lambda_um, p_max=p_max, g_exc=g_exc, g_inh=g_inh,
    )
    syn_state = create_synapse_state(n_neurons)
    n_steps = int(duration_ms / dt)

    I_noise = bg_mean + bg_std * jax.random.normal(k4, (n_steps, n_neurons))

    def step_fn(carry, I_t):
        ns, ss = carry
        I_syn = compute_synaptic_current(ss, ns.v)
        ns = izhikevich_step(ns, params, I_syn + I_t, dt)
        spikes_f = ns.spikes.astype(jnp.float32)
        ss = SynapseState(
            g_ampa=ampa_step(ss.g_ampa, spikes_f, W_exc, dt),
            g_gaba_a=gaba_a_step(ss.g_gaba_a, spikes_f, W_inh, dt),
            g_nmda_rise=ss.g_nmda_rise, g_nmda_decay=ss.g_nmda_decay,
            g_gaba_b_rise=ss.g_gaba_b_rise, g_gaba_b_decay=ss.g_gaba_b_decay,
        )
        return (ns, ss), ns.spikes

    print(f"Simulating {n_neurons} neurons for {duration_ms/1000:.0f}s (dt={dt}ms)...")
    t0 = time.perf_counter()
    _, spike_history = jax.lax.scan(step_fn, (state, syn_state), I_noise)
    spike_history.block_until_ready()
    wall = time.perf_counter() - t0
    print(f"  Done in {wall:.1f}s ({duration_ms/1000/wall:.1f}x realtime)")

    raster = np.asarray(spike_history)  # (T, N) bool
    is_exc_np = np.asarray(is_exc)
    positions_np = np.asarray(positions)

    return raster, is_exc_np, positions_np, {"dt": dt, "duration_ms": duration_ms}


def raster_to_sorting(
    raster: np.ndarray, dt_ms: float, is_exc: np.ndarray,
) -> si.NumpySorting:
    """Convert BL-1 (T, N) boolean raster to SpikeInterface NumpySorting."""
    sampling_frequency = 1000.0 / dt_ms  # Hz

    # Extract spike times per unit
    spike_times, unit_ids = np.nonzero(raster)

    # Sort by time
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    unit_ids = unit_ids[order]

    sorting = si.NumpySorting.from_times_labels(
        times_list=spike_times,
        labels_list=unit_ids,
        sampling_frequency=sampling_frequency,
    )
    return sorting


def plot_spikeinterface_raster(
    sorting: si.BaseSorting,
    is_exc: np.ndarray,
    bursts: list,
    dt_ms: float,
    output_dir: Path,
    time_range: tuple[float, float] | None = None,
):
    """Full raster plot with E/I coloring and burst markers."""
    n_units = sorting.get_num_units()
    fs = sorting.get_sampling_frequency()

    # Subsample units for readability
    max_units = 500
    if n_units > max_units:
        unit_ids = np.sort(
            np.random.default_rng(0).choice(n_units, max_units, replace=False)
        )
    else:
        unit_ids = np.arange(n_units)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={"hspace": 0.05})

    # -- Raster --
    ax = axes[0]
    for rank, uid in enumerate(unit_ids):
        st = sorting.get_unit_spike_train(uid)
        times_s = st / fs
        if time_range:
            mask = (times_s >= time_range[0]) & (times_s <= time_range[1])
            times_s = times_s[mask]
        color = "#2196F3" if is_exc[uid] else "#F44336"
        ax.scatter(times_s, np.full_like(times_s, rank), s=0.3, c=color,
                   alpha=0.5, linewidths=0, rasterized=True)

    # Burst shading
    for start_ms, end_ms, _, _ in bursts:
        start_s, end_s = start_ms / 1000, end_ms / 1000
        if time_range and (end_s < time_range[0] or start_s > time_range[1]):
            continue
        ax.axvspan(start_s, end_s, alpha=0.08, color="#FF9800", zorder=0)

    ax.set_ylabel(f"Neuron (of {max_units} shown)")
    ax.set_title("BL-1 Cortical Culture — Spike Raster (SpikeInterface)")
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
                   markersize=6, label="Excitatory"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
                   markersize=6, label="Inhibitory"),
            plt.Rectangle((0, 0), 1, 1, fc="#FF9800", alpha=0.3, label="Burst"),
        ],
        loc="upper right", fontsize=8, framealpha=0.8,
    )

    # -- Population rate --
    ax2 = axes[1]
    all_spikes = np.concatenate(
        [sorting.get_unit_spike_train(uid) for uid in range(n_units)]
    )
    all_times_s = all_spikes / fs
    if time_range:
        all_times_s = all_times_s[
            (all_times_s >= time_range[0]) & (all_times_s <= time_range[1])
        ]
    bin_width = 0.01  # 10ms bins
    if time_range:
        bins = np.arange(time_range[0], time_range[1], bin_width)
    else:
        bins = np.arange(0, sorting.get_total_duration() / fs, bin_width)
    counts, edges = np.histogram(all_times_s, bins=bins)
    rate_hz = counts / (bin_width * n_units)
    ax2.fill_between(edges[:-1], rate_hz, alpha=0.6, color="#2196F3")
    ax2.set_ylabel("Rate (Hz)")
    ax2.set_xlabel("Time (s)")

    suffix = ""
    if time_range:
        suffix = f"_{time_range[0]:.0f}-{time_range[1]:.0f}s"
    fig.savefig(output_dir / f"raster{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved raster{suffix}.png")


def plot_isi_distribution(sorting: si.BaseSorting, output_dir: Path):
    """Log-scale ISI histogram across all units."""
    fs = sorting.get_sampling_frequency()
    all_isis = []
    for uid in sorting.get_unit_ids()[:1000]:  # cap for speed
        st = sorting.get_unit_spike_train(uid)
        if len(st) > 1:
            isis = np.diff(st) / fs * 1000  # ms
            all_isis.append(isis)

    if not all_isis:
        print("  No ISIs to plot (insufficient spikes)")
        return

    all_isis = np.concatenate(all_isis)
    all_isis = all_isis[all_isis > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(np.log10(0.5), np.log10(all_isis.max()), 100)
    ax.hist(all_isis, bins=bins, color="#2196F3", alpha=0.7, edgecolor="white",
            linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(np.median(all_isis), color="#F44336", ls="--", lw=1.5,
               label=f"Median: {np.median(all_isis):.1f} ms")
    ax.set_xlabel("Inter-Spike Interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title("ISI Distribution (log-log)")
    ax.legend()
    fig.savefig(output_dir / "isi_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved isi_distribution.png")


def plot_firing_rates(sorting: si.BaseSorting, is_exc: np.ndarray, output_dir: Path):
    """Per-unit firing rate histogram with E/I split."""
    fs = sorting.get_sampling_frequency()
    duration_s = sorting.get_total_duration() / fs
    rates = []
    for uid in sorting.get_unit_ids():
        n_spikes = len(sorting.get_unit_spike_train(uid))
        rates.append(n_spikes / duration_s)
    rates = np.array(rates)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, min(rates.max(), 20), 50)
    ax.hist(rates[is_exc[:len(rates)].astype(bool)], bins=bins, alpha=0.6,
            color="#2196F3", label=f"Excitatory (mean={rates[is_exc[:len(rates)].astype(bool)].mean():.2f} Hz)")
    ax.hist(rates[~is_exc[:len(rates)].astype(bool)], bins=bins, alpha=0.6,
            color="#F44336", label=f"Inhibitory (mean={rates[~is_exc[:len(rates)].astype(bool)].mean():.2f} Hz)")
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Firing Rate Distribution by Cell Type")
    ax.legend()
    fig.savefig(output_dir / "firing_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved firing_rates.png")


def plot_burst_analysis(bursts: list, duration_ms: float, output_dir: Path):
    """Burst statistics summary panel."""
    if len(bursts) < 2:
        print("  Insufficient bursts for analysis panel")
        return

    stats = burst_statistics(bursts)
    durations = [e - s for s, e, _, _ in bursts]
    ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    recruitments = [r for _, _, _, r in bursts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Burst duration
    axes[0].hist(durations, bins=20, color="#FF9800", alpha=0.7, edgecolor="white")
    axes[0].axvline(np.mean(durations), color="k", ls="--",
                    label=f"Mean: {np.mean(durations):.0f} ms")
    axes[0].set_xlabel("Burst Duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Burst Duration")
    axes[0].legend(fontsize=8)

    # IBI
    axes[1].hist(ibis, bins=20, color="#4CAF50", alpha=0.7, edgecolor="white")
    axes[1].axvline(np.mean(ibis), color="k", ls="--",
                    label=f"Mean: {np.mean(ibis)/1000:.1f} s (CV={stats.get('ibi_cv', 0):.2f})")
    axes[1].set_xlabel("Inter-Burst Interval (ms)")
    axes[1].set_title("Inter-Burst Intervals")
    axes[1].legend(fontsize=8)

    # Recruitment
    axes[2].hist(recruitments, bins=20, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[2].axvline(np.mean(recruitments), color="k", ls="--",
                    label=f"Mean: {np.mean(recruitments)*100:.0f}%")
    axes[2].set_xlabel("Fraction Recruited")
    axes[2].set_title("Burst Recruitment")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Burst Analysis — {len(bursts)} bursts in {duration_ms/1000:.0f}s "
        f"({len(bursts)/(duration_ms/60000):.1f}/min)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "burst_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved burst_analysis.png")


def plot_correlograms(sorting: si.BaseSorting, is_exc: np.ndarray, output_dir: Path):
    """Auto- and cross-correlograms for a few example units."""
    n_units = sorting.get_num_units()
    if n_units < 4:
        return

    # Pick 2 excitatory, 2 inhibitory with highest spike counts
    counts = np.array([len(sorting.get_unit_spike_train(u)) for u in range(n_units)])
    exc_ids = np.where(is_exc[:n_units])[0]
    inh_ids = np.where(~is_exc[:n_units].astype(bool))[0]

    top_exc = exc_ids[np.argsort(counts[exc_ids])[-2:]] if len(exc_ids) >= 2 else exc_ids[:2]
    top_inh = inh_ids[np.argsort(counts[inh_ids])[-2:]] if len(inh_ids) >= 2 else inh_ids[:2]
    example_units = np.concatenate([top_exc, top_inh])

    fs = sorting.get_sampling_frequency()
    window_ms = 50
    bin_ms = 1
    window_samples = int(window_ms / 1000 * fs)
    bin_samples = max(1, int(bin_ms / 1000 * fs))

    fig, axes = plt.subplots(len(example_units), 1, figsize=(10, 2.5 * len(example_units)))
    if len(example_units) == 1:
        axes = [axes]

    for ax, uid in zip(axes, example_units):
        st = sorting.get_unit_spike_train(uid)
        if len(st) < 10:
            ax.text(0.5, 0.5, "Too few spikes", transform=ax.transAxes, ha="center")
            continue

        # Compute auto-correlogram
        diffs = []
        for i in range(len(st)):
            nearby = st[(st > st[i] - window_samples) & (st < st[i] + window_samples)]
            d = nearby - st[i]
            d = d[d != 0]
            diffs.extend(d)

            if len(diffs) > 100_000:  # cap for speed
                break

        diffs_ms = np.array(diffs) / fs * 1000
        bins = np.arange(-window_ms, window_ms + bin_ms, bin_ms)
        color = "#2196F3" if is_exc[uid] else "#F44336"
        label = f"Unit {uid} ({'E' if is_exc[uid] else 'I'}, {len(st)} spikes)"
        ax.hist(diffs_ms, bins=bins, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.3)
        ax.set_ylabel("Count")
        ax.set_title(label, fontsize=10)

    axes[-1].set_xlabel("Lag (ms)")
    fig.suptitle("Auto-Correlograms", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "correlograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved correlograms.png")


def main():
    parser = argparse.ArgumentParser(description="BL-1 spike visualizations via SpikeInterface")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--duration-s", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation
    raster, is_exc, positions, meta = run_simulation(
        n_neurons=args.n_neurons,
        duration_ms=args.duration_s * 1000,
        seed=args.seed,
    )
    dt_ms = meta["dt"]

    # Detect bursts
    bursts = detect_bursts(raster, dt_ms=dt_ms, threshold_std=1.5, min_duration_ms=50)
    stats = burst_statistics(bursts)
    print(f"\nBurst detection: {len(bursts)} bursts ({len(bursts)/(args.duration_s/60):.1f}/min)")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Convert to SpikeInterface
    print("\nConverting to SpikeInterface format...")
    sorting = raster_to_sorting(raster, dt_ms, is_exc)
    print(f"  {sorting.get_num_units()} units, {sorting.get_total_duration()/sorting.get_sampling_frequency():.1f}s")

    # Generate visualizations
    print("\nGenerating figures...")

    # 1. Full raster
    plot_spikeinterface_raster(sorting, is_exc, bursts, dt_ms, output_dir)

    # 2. Zoomed burst window (find first burst and show ±2s around it)
    if bursts:
        mid = (bursts[0][0] + bursts[0][1]) / 2 / 1000  # seconds
        window = (max(0, mid - 2), mid + 2)
        plot_spikeinterface_raster(sorting, is_exc, bursts, dt_ms, output_dir,
                                   time_range=window)

    # 3. ISI distribution
    plot_isi_distribution(sorting, output_dir)

    # 4. Firing rate distribution
    plot_firing_rates(sorting, is_exc, output_dir)

    # 5. Burst analysis panel
    plot_burst_analysis(bursts, args.duration_s * 1000, output_dir)

    # 6. Correlograms
    plot_correlograms(sorting, is_exc, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
