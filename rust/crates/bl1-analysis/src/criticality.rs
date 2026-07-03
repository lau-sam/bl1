//! Criticality metrics: branching ratio and neuronal avalanches
//! (Beggs & Plenz 2003), with a maximum-likelihood power-law exponent
//! estimator (Clauset, Shalizi & Newman 2009).
//!
//! A branching ratio `sigma ≈ 1` indicates operation near the critical point;
//! avalanche size/duration distributions should follow power laws with
//! exponents near `-1.5` and `-2.0`.

use bl1_core::Raster;

/// Bin the raster into total spike counts per `bin_ms` window.
fn bin_spikes(raster: &Raster, dt_ms: f32, bin_ms: f32) -> Vec<f64> {
    if raster.n_steps == 0 || raster.n_neurons == 0 {
        return Vec::new();
    }
    let steps_per_bin = ((bin_ms / dt_ms).round() as usize).max(1);
    let n_bins = raster.n_steps / steps_per_bin;
    let mut binned = vec![0.0f64; n_bins];
    for (b, slot) in binned.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for row in b * steps_per_bin..(b + 1) * steps_per_bin {
            acc += raster.row(row).iter().map(|&x| x as f64).sum::<f64>();
        }
        *slot = acc;
    }
    binned
}

/// Branching ratio `sigma = <n[t+1] / n[t]>` over bins with `n[t] > 0`.
/// Returns `NaN` if there are no active ancestor bins.
pub fn branching_ratio(raster: &Raster, dt_ms: f32, bin_ms: f32) -> f64 {
    let binned = bin_spikes(raster, dt_ms, bin_ms);
    if binned.len() < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for w in binned.windows(2) {
        if w[0] > 0.0 {
            sum += w[1] / w[0];
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Avalanche size (total spikes) and duration (active bins) distributions.
/// An avalanche is a maximal run of bins with at least one spike.
pub fn avalanche_distributions(raster: &Raster, dt_ms: f32, bin_ms: f32) -> (Vec<f64>, Vec<u64>) {
    let binned = bin_spikes(raster, dt_ms, bin_ms);
    let mut sizes = Vec::new();
    let mut durations = Vec::new();
    let mut in_av = false;
    let mut size = 0.0f64;
    let mut dur = 0u64;
    for &b in &binned {
        if b > 0.0 {
            if !in_av {
                in_av = true;
                size = 0.0;
                dur = 0;
            }
            size += b;
            dur += 1;
        } else if in_av {
            sizes.push(size);
            durations.push(dur);
            in_av = false;
        }
    }
    if in_av {
        sizes.push(size);
        durations.push(dur);
    }
    (sizes, durations)
}

/// Discrete MLE of the power-law exponent for `x >= xmin`
/// (Clauset et al. 2009, eq. 3.7). Returns `alpha`, or `NaN` if degenerate.
fn mle_alpha(tail: &[f64], xmin: f64) -> f64 {
    let n = tail.len() as f64;
    let log_sum: f64 = tail.iter().map(|&x| (x / (xmin - 0.5)).ln()).sum();
    if log_sum <= 0.0 {
        return f64::NAN;
    }
    1.0 + n / log_sum
}

/// Estimate the power-law exponent of *values* by maximum likelihood with
/// KS-based `xmin` selection, returning `-alpha` (negative for a power-law
/// decay; directly comparable to the reference avalanche exponents `-1.5` /
/// `-2.0`). Returns `NaN` if there is insufficient data.
///
/// This mirrors the corrected Python estimator and replaces the biased log-log
/// CCDF regression.
pub fn estimate_power_law_exponent(values: &[f64]) -> f64 {
    if values.len() < 5 {
        return f64::NAN;
    }
    let mut pos: Vec<f64> = values.iter().copied().filter(|&x| x > 0.0).collect();
    if pos.len() < 5 {
        return f64::NAN;
    }
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut unique = pos.clone();
    unique.dedup();
    if unique.len() < 3 {
        return f64::NAN;
    }

    let mut best_alpha = f64::NAN;
    let mut best_ks = f64::INFINITY;
    // Candidate cutoffs: every unique value except the top two (keep a tail).
    for &xmin in &unique[..unique.len() - 2] {
        let tail: Vec<f64> = pos.iter().copied().filter(|&x| x >= xmin).collect();
        if tail.len() < 5 {
            continue;
        }
        let alpha = mle_alpha(&tail, xmin);
        if !alpha.is_finite() {
            continue;
        }
        let n = tail.len() as f64;
        let mut ks = 0.0f64;
        for (k, &x) in tail.iter().enumerate() {
            let cdf_emp = (k + 1) as f64 / n;
            let cdf_fit = 1.0 - (x / xmin).powf(-(alpha - 1.0));
            ks = ks.max((cdf_emp - cdf_fit).abs());
        }
        if ks < best_ks {
            best_ks = ks;
            best_alpha = alpha;
        }
    }

    if !best_alpha.is_finite() {
        best_alpha = mle_alpha(&pos, unique[0]);
        if !best_alpha.is_finite() {
            return f64::NAN;
        }
    }
    -best_alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poisson_like(n_steps: usize, n_neurons: usize) -> Raster {
        // Deterministic pseudo-random raster to exercise the binning paths.
        let mut data = vec![0.0f32; n_steps * n_neurons];
        let mut z = 12345u64;
        for x in data.iter_mut() {
            z = z
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (z >> 40) % 100 < 5 {
                *x = 1.0;
            }
        }
        Raster {
            n_steps,
            n_neurons,
            data,
        }
    }

    #[test]
    fn branching_ratio_of_steady_activity_is_near_one() {
        let r = poisson_like(20_000, 50);
        let sigma = branching_ratio(&r, 0.5, 4.0);
        assert!(sigma.is_finite());
        assert!((0.5..1.5).contains(&sigma), "sigma = {sigma}");
    }

    #[test]
    fn empty_and_single_bin_are_nan() {
        let empty = Raster {
            n_steps: 0,
            n_neurons: 0,
            data: vec![],
        };
        assert!(branching_ratio(&empty, 0.5, 4.0).is_nan());
    }

    #[test]
    fn avalanches_detected() {
        let r = poisson_like(20_000, 50);
        let (sizes, durations) = avalanche_distributions(&r, 0.5, 4.0);
        assert!(!sizes.is_empty());
        assert_eq!(sizes.len(), durations.len());
    }

    #[test]
    fn mle_recovers_power_law_exponent() {
        // Continuous power law alpha = 2.5 via inverse CDF, discretised.
        let mut z = 999u64;
        let mut data = Vec::new();
        for _ in 0..20_000 {
            z = z
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((z >> 11) as f64) / (1u64 << 53) as f64;
            let x = (1.0 - u).powf(-1.0 / (2.5 - 1.0));
            data.push(x.floor());
        }
        let est = estimate_power_law_exponent(&data);
        assert!(est.is_finite() && est < 0.0);
        assert!((est + 2.5).abs() < 0.4, "estimate {est}, want ~ -2.5");
    }

    #[test]
    fn too_few_values_is_nan() {
        assert!(estimate_power_law_exponent(&[1.0, 2.0]).is_nan());
        assert!(estimate_power_law_exponent(&[5.0; 10]).is_nan());
    }
}
