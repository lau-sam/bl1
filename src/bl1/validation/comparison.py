"""Compute and compare culture-level statistics between simulation and data.

This module provides two main entry points:

- :func:`compute_culture_statistics` takes a spike raster (the standard
  BL-1 simulation output) and computes a comprehensive set of summary
  statistics that match the metrics reported in published cortical
  culture datasets.

- :func:`generate_comparison_report` produces a human-readable text
  report comparing a simulation's statistics against a named reference
  dataset, with pass/fail indicators for each metric.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from bl1.validation.datasets import DATASETS, compare_statistics

# ============================================================================
# Comprehensive culture statistics
# ============================================================================


def compute_culture_statistics(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    burst_threshold_std: float = 2.0,
    burst_min_duration_ms: float = 50.0,
    avalanche_bin_ms: float = 4.0,
) -> dict[str, float]:
    """Compute a comprehensive set of culture statistics from a spike raster.

    This function wraps the individual analysis routines in
    :mod:`bl1.analysis.bursts` and :mod:`bl1.analysis.criticality` to
    produce a single dict of metrics that can be directly compared
    against published dataset ranges via
    :func:`bl1.validation.datasets.compare_statistics`.

    Args:
        spike_raster: ``(T, N)`` boolean or 0/1 spike raster where
            ``T`` is the number of timesteps and ``N`` is the number of
            neurons/electrodes.
        dt_ms: Simulation timestep in ms (default 0.5).
        burst_threshold_std: Number of standard deviations for burst
            onset detection (default 2.0).
        burst_min_duration_ms: Minimum burst duration in ms
            (default 50.0).
        avalanche_bin_ms: Bin width in ms for avalanche detection
            (default 4.0).

    Returns:
        Dict with the following keys (all values are ``float``):

        - ``mean_firing_rate_hz`` -- Mean firing rate across all
          neurons, in Hz.
        - ``burst_rate_per_min`` -- Number of detected bursts per
          minute of simulated time.
        - ``ibi_mean_ms`` -- Mean inter-burst interval in ms.
        - ``ibi_cv`` -- Coefficient of variation of inter-burst
          intervals.
        - ``burst_duration_mean_ms`` -- Mean burst duration in ms.
        - ``recruitment_mean`` -- Mean fraction of neurons recruited
          per burst.
        - ``branching_ratio`` -- Branching ratio sigma (1.0 =
          critical).
        - ``avalanche_size_exponent`` -- Estimated power-law exponent
          for the avalanche size distribution (maximum-likelihood,
          Clauset et al. 2009).  Returned as ``-alpha`` (negative for
          power-law decays; reference ``-1.5``).
        - ``avalanche_duration_exponent`` -- Estimated power-law
          exponent for the avalanche duration distribution (reference
          ``-2.0``).
        - ``population_rate_cv`` -- Coefficient of variation of the
          population spike count time series.
        - ``fraction_active`` -- Fraction of neurons that fire at
          least once during the recording.
    """
    from bl1.analysis.bursts import burst_statistics, detect_bursts
    from bl1.analysis.criticality import (
        avalanche_size_distribution,
    )
    from bl1.analysis.criticality import (
        branching_ratio as compute_branching_ratio,
    )

    raster = np.asarray(spike_raster, dtype=np.float32)
    T, N = raster.shape
    total_time_s = T * dt_ms / 1000.0

    stats: dict[str, float] = {}

    # --- Firing rate --------------------------------------------------------
    total_spikes = float(raster.sum())
    if N > 0 and total_time_s > 0:
        stats["mean_firing_rate_hz"] = total_spikes / (N * total_time_s)
    else:
        stats["mean_firing_rate_hz"] = 0.0

    # --- Burst detection and statistics ------------------------------------
    bursts = detect_bursts(
        raster,
        dt_ms=dt_ms,
        threshold_std=burst_threshold_std,
        min_duration_ms=burst_min_duration_ms,
    )
    bstats = burst_statistics(bursts)

    n_bursts = len(bursts)
    total_time_min = total_time_s / 60.0
    stats["burst_rate_per_min"] = n_bursts / total_time_min if total_time_min > 0 else 0.0
    stats["ibi_mean_ms"] = bstats["ibi_mean"]
    stats["ibi_cv"] = bstats["ibi_cv"]
    stats["burst_duration_mean_ms"] = bstats["duration_mean"]
    stats["recruitment_mean"] = bstats["recruitment_mean"]

    # --- Criticality metrics -----------------------------------------------
    sigma = compute_branching_ratio(raster, dt_ms=dt_ms, bin_ms=avalanche_bin_ms)
    stats["branching_ratio"] = sigma

    sizes, durations = avalanche_size_distribution(raster, dt_ms=dt_ms, bin_ms=avalanche_bin_ms)

    stats["avalanche_size_exponent"] = _estimate_power_law_exponent(sizes)
    stats["avalanche_duration_exponent"] = _estimate_power_law_exponent(
        durations.astype(np.float64)
    )

    # --- Population rate statistics ----------------------------------------
    pop_count = raster.sum(axis=1)  # (T,)
    pop_mean = float(np.mean(pop_count))
    pop_std = float(np.std(pop_count))
    stats["population_rate_cv"] = pop_std / pop_mean if pop_mean > 0 else 0.0

    # --- Fraction of active neurons ----------------------------------------
    neuron_spike_counts = raster.sum(axis=0)  # (N,)
    stats["fraction_active"] = float(np.mean(neuron_spike_counts > 0))

    # --- Functional connectivity & information metrics ---------------------
    # These are O(N^2) or worse, so only compute on a small subset of
    # neurons to keep runtime manageable for large networks.
    from bl1.analysis.connectivity import cross_correlation_matrix, transfer_entropy
    from bl1.analysis.information import active_information_storage
    from bl1.analysis.information import integration as compute_integration

    _FC_SUBSET = 50
    subset_raster = raster[:, : min(N, _FC_SUBSET)]

    try:
        cc_mat = cross_correlation_matrix(subset_raster, dt_ms=dt_ms)
        # Mean off-diagonal cross-correlation
        n_sub = cc_mat.shape[0]
        if n_sub > 1:
            mask = ~np.eye(n_sub, dtype=bool)
            stats["mean_cross_correlation"] = float(np.mean(cc_mat[mask]))
        else:
            stats["mean_cross_correlation"] = 0.0
    except Exception:
        stats["mean_cross_correlation"] = float("nan")

    try:
        te_mat = transfer_entropy(
            subset_raster,
            dt_ms=dt_ms,
            history_bins=3,
            subset=min(N, _FC_SUBSET),
        )
        n_sub = te_mat.shape[0]
        if n_sub > 1:
            mask = ~np.eye(n_sub, dtype=bool)
            stats["transfer_entropy_mean"] = float(np.mean(te_mat[mask]))
        else:
            stats["transfer_entropy_mean"] = 0.0
    except Exception:
        stats["transfer_entropy_mean"] = float("nan")

    try:
        ais = active_information_storage(
            subset_raster,
            dt_ms=dt_ms,
            history_length=3,
        )
        stats["active_information_storage_mean"] = float(np.mean(ais))
    except Exception:
        stats["active_information_storage_mean"] = float("nan")

    try:
        stats["integration"] = compute_integration(
            subset_raster,
            dt_ms=dt_ms,
            n_samples=50,
        )
    except Exception:
        stats["integration"] = float("nan")

    return stats


# ============================================================================
# Power-law exponent estimation (simple log-log regression)
# ============================================================================


def _mle_alpha(tail: NDArray, xmin: float) -> float:
    """Discrete MLE of the power-law exponent for ``x >= xmin``.

    Uses the discrete maximum-likelihood approximation of Clauset,
    Shalizi & Newman (2009), eq. 3.7:

        alpha = 1 + n * [ sum_i ln(x_i / (xmin - 1/2)) ]^{-1}

    Returns the (positive) pdf exponent ``alpha``, or ``nan`` if the
    log-sum is non-positive (degenerate).
    """
    n = len(tail)
    log_sum = float(np.sum(np.log(tail / (xmin - 0.5))))
    if log_sum <= 0.0:
        return float("nan")
    return 1.0 + n / log_sum


def _estimate_power_law_exponent(values: NDArray) -> float:
    """Estimate the power-law exponent by maximum likelihood.

    Fits a power law ``p(x) ~ x^{-alpha}`` to the tail of *values* using
    the discrete maximum-likelihood estimator of Clauset, Shalizi &
    Newman (2009), with the lower cutoff ``xmin`` selected by minimising
    the Kolmogorov-Smirnov distance between the empirical and fitted
    distributions.  This replaces the earlier log-log CCDF regression,
    which is known to be biased (Clauset et al. 2009, §3).

    The returned value is ``-alpha``: negative for a power-law decay and
    directly comparable to the reference avalanche exponents (size
    ``-1.5``, duration ``-2.0``; Beggs & Plenz 2003).

    Args:
        values: 1-D array of positive values (e.g. avalanche sizes).

    Returns:
        ``-alpha`` (negative for power-law decay), or ``nan`` if there
        is insufficient data.
    """
    if len(values) < 5:
        return float("nan")

    pos = np.asarray(values, dtype=np.float64)
    pos = pos[pos > 0]
    if len(pos) < 5:
        return float("nan")

    sorted_pos = np.sort(pos)
    unique_vals = np.unique(sorted_pos)
    if len(unique_vals) < 3:
        return float("nan")

    # Select xmin by KS minimisation over candidate cutoffs, keeping a
    # tail of at least 5 points (Clauset et al. 2009, §3.3).
    best_alpha = float("nan")
    best_ks = float("inf")
    for xmin in unique_vals[:-2]:
        tail = sorted_pos[sorted_pos >= xmin]
        if len(tail) < 5:
            continue
        alpha = _mle_alpha(tail, xmin)
        if not math.isfinite(alpha):
            continue
        # KS distance vs the fitted (continuous) power-law CDF on the tail.
        cdf_emp = np.arange(1, len(tail) + 1) / len(tail)
        cdf_fit = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        ks = float(np.max(np.abs(cdf_emp - cdf_fit)))
        if ks < best_ks:
            best_ks = ks
            best_alpha = alpha

    # Fallback: fit the whole positive support if no tail cutoff qualified.
    if not math.isfinite(best_alpha):
        best_alpha = _mle_alpha(sorted_pos, float(unique_vals[0]))
        if not math.isfinite(best_alpha):
            return float("nan")

    return -best_alpha


# ============================================================================
# Comparison report
# ============================================================================


def generate_comparison_report(
    sim_stats: dict[str, float],
    dataset_name: str = "wagenaar_2006",
) -> str:
    """Generate a text report comparing simulation to published data.

    Args:
        sim_stats: Dict of simulation statistics, as returned by
            :func:`compute_culture_statistics`.
        dataset_name: Key into :data:`~bl1.validation.datasets.DATASETS`
            for the reference dataset.

    Returns:
        Formatted multi-line string with pass/fail for each metric.
    """
    ds = DATASETS[dataset_name]
    comparison = compare_statistics(sim_stats, dataset_name)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"BL-1 Validation Report vs. {ds.name}")
    lines.append(f"Paper: {ds.paper}")
    lines.append(f"Species: {ds.species}  |  Culture: {ds.culture_type}  |  DIV: {ds.div_range}")
    lines.append("=" * 72)
    lines.append("")

    n_pass = 0
    n_fail = 0
    n_no_ref = 0

    for metric_key, result in sorted(comparison.items()):
        sim_val = result["sim_value"]
        ref_range = result["ref_range"]
        in_range = result["in_range"]

        if in_range is True:
            status = "PASS"
            n_pass += 1
        elif in_range is False:
            status = "FAIL"
            n_fail += 1
        else:
            status = "N/A "
            n_no_ref += 1

        if ref_range is not None:
            ref_str = f"[{ref_range[0]:.3g}, {ref_range[1]:.3g}]"
        else:
            ref_str = "no reference"

        # Format sim value, handling NaN
        if math.isnan(sim_val):
            val_str = "NaN"
        else:
            val_str = f"{sim_val:.4g}"

        lines.append(f"  [{status}]  {metric_key:<30s}  sim={val_str:<12s}  ref={ref_str}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"Summary: {n_pass} passed, {n_fail} failed, {n_no_ref} no reference data")
    lines.append("-" * 72)

    return "\n".join(lines)
