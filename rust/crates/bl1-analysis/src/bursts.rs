//! Network burst detection and statistics (Wagenaar et al. 2006).
//!
//! Bursts are detected by threshold-crossing on the population spike count: a
//! burst begins when the count exceeds `mean + threshold_std · std` and ends
//! when it falls back to the mean.

use bl1_core::Raster;

/// A detected network burst.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Burst {
    pub start_ms: f32,
    pub end_ms: f32,
    pub n_spikes: u64,
    /// Fraction of neurons that fired at least once during the burst.
    pub fraction_recruited: f32,
}

/// Detect network bursts in a spike raster.
///
/// `threshold_std` sets the onset threshold in standard deviations above the
/// mean population count; only bursts lasting at least `min_duration_ms` are
/// kept.
pub fn detect_bursts(
    raster: &Raster,
    dt_ms: f32,
    threshold_std: f32,
    min_duration_ms: f32,
) -> Vec<Burst> {
    let (t, n) = (raster.n_steps, raster.n_neurons);
    if t == 0 || n == 0 {
        return Vec::new();
    }

    let pop_count: Vec<f32> = (0..t)
        .map(|row| raster.row(row).iter().sum::<f32>())
        .collect();
    let mean = pop_count.iter().sum::<f32>() / t as f32;
    let var = pop_count.iter().map(|&c| (c - mean).powi(2)).sum::<f32>() / t as f32;
    let std = var.sqrt();
    if std < 1e-12 {
        return Vec::new();
    }

    let onset = mean + threshold_std * std;
    let offset = mean;
    let min_steps = ((min_duration_ms / dt_ms).round() as usize).max(1);

    let mut bursts = Vec::new();
    let mut in_burst = false;
    let mut start = 0usize;

    let close = |start: usize, end: usize, bursts: &mut Vec<Burst>| {
        if end - start >= min_steps {
            let mut n_spikes = 0u64;
            let mut active = vec![false; n];
            for row in start..end {
                for (j, &s) in raster.row(row).iter().enumerate() {
                    if s != 0.0 {
                        n_spikes += 1;
                        active[j] = true;
                    }
                }
            }
            let recruited = active.iter().filter(|&&a| a).count() as f32 / n as f32;
            bursts.push(Burst {
                start_ms: start as f32 * dt_ms,
                end_ms: end as f32 * dt_ms,
                n_spikes,
                fraction_recruited: recruited,
            });
        }
    };

    for (row, &c) in pop_count.iter().enumerate() {
        if !in_burst {
            if c > onset {
                in_burst = true;
                start = row;
            }
        } else if c <= offset {
            close(start, row, &mut bursts);
            in_burst = false;
        }
    }
    if in_burst {
        close(start, t, &mut bursts);
    }
    bursts
}

/// Summary statistics over a set of detected bursts.
#[derive(Debug, Clone, Copy)]
pub struct BurstStatistics {
    pub n_bursts: usize,
    pub ibi_mean_ms: f32,
    pub ibi_cv: f32,
    pub duration_mean_ms: f32,
    pub recruitment_mean: f32,
    /// Bursts per minute over the full recording of `total_time_ms`.
    pub burst_rate_per_min: f32,
}

/// Compute burst statistics. `total_time_ms` is the full recording length,
/// used for the (edge-inclusive) burst rate matching the reference model.
pub fn burst_statistics(bursts: &[Burst], total_time_ms: f32) -> BurstStatistics {
    let n = bursts.len();
    let total_min = total_time_ms / 60_000.0;
    let burst_rate_per_min = if total_min > 0.0 {
        n as f32 / total_min
    } else {
        0.0
    };

    if n == 0 {
        return BurstStatistics {
            n_bursts: 0,
            ibi_mean_ms: f32::NAN,
            ibi_cv: f32::NAN,
            duration_mean_ms: f32::NAN,
            recruitment_mean: f32::NAN,
            burst_rate_per_min,
        };
    }

    let duration_mean = bursts.iter().map(|b| b.end_ms - b.start_ms).sum::<f32>() / n as f32;
    let recruitment_mean = bursts.iter().map(|b| b.fraction_recruited).sum::<f32>() / n as f32;

    let (ibi_mean, ibi_cv) = if n >= 2 {
        let ibis: Vec<f32> = bursts
            .windows(2)
            .map(|w| w[1].start_ms - w[0].start_ms)
            .collect();
        let m = ibis.iter().sum::<f32>() / ibis.len() as f32;
        let v = ibis.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / ibis.len() as f32;
        let cv = if m > 0.0 { v.sqrt() / m } else { f32::NAN };
        (m, cv)
    } else {
        (f32::NAN, f32::NAN)
    };

    BurstStatistics {
        n_bursts: n,
        ibi_mean_ms: ibi_mean,
        ibi_cv,
        duration_mean_ms: duration_mean,
        recruitment_mean,
        burst_rate_per_min,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a raster with periodic synchronous bursts.
    fn bursty(n_steps: usize, n_neurons: usize, period: usize, width: usize) -> Raster {
        let mut data = vec![0.0f32; n_steps * n_neurons];
        for t in 0..n_steps {
            if t % period < width {
                for j in 0..n_neurons {
                    data[t * n_neurons + j] = 1.0;
                }
            }
        }
        Raster {
            n_steps,
            n_neurons,
            data,
        }
    }

    #[test]
    fn detects_periodic_bursts() {
        let r = bursty(10_000, 30, 1000, 40);
        let bursts = detect_bursts(&r, 0.5, 2.0, 5.0);
        // ~10 bursts expected (one per period).
        assert!(
            (8..=10).contains(&bursts.len()),
            "got {} bursts",
            bursts.len()
        );
        assert!(bursts[0].fraction_recruited > 0.9);
    }

    #[test]
    fn silent_raster_has_no_bursts() {
        let r = Raster {
            n_steps: 1000,
            n_neurons: 10,
            data: vec![0.0; 10_000],
        };
        assert!(detect_bursts(&r, 0.5, 2.0, 5.0).is_empty());
    }

    #[test]
    fn statistics_are_reasonable() {
        let r = bursty(20_000, 30, 2000, 40);
        let bursts = detect_bursts(&r, 0.5, 2.0, 5.0);
        let stats = burst_statistics(&bursts, 20_000.0 * 0.5);
        assert!(stats.n_bursts >= 8);
        // Period 2000 steps * 0.5 ms = 1000 ms between burst onsets.
        assert!(
            (stats.ibi_mean_ms - 1000.0).abs() < 50.0,
            "ibi={}",
            stats.ibi_mean_ms
        );
        assert!(stats.burst_rate_per_min > 0.0);
    }
}
