//! `bl1-mea` — virtual multi-electrode array.
//!
//! Provides the standard CL1 64-channel and MaxOne HD-MEA layouts, a
//! neuron→electrode detection map, spike detection, and a point-source LFP
//! approximation. Neuron positions are 2-D `[x, y]` coordinates in micrometres.

use bl1_core::CsrMatrix;

/// A 2-D position in micrometres.
pub type Position = [f32; 2];

/// An electrode-array layout.
#[derive(Debug, Clone)]
pub struct MeaConfig {
    pub name: String,
    /// Electrode centre positions (µm).
    pub positions: Vec<Position>,
    /// Radius (µm) within which a neuron's spikes are detected by an electrode.
    pub detection_radius: f32,
    /// Radius (µm) within which stimulation reaches a neuron.
    pub activation_radius: f32,
}

impl MeaConfig {
    /// Number of electrodes.
    pub fn n_electrodes(&self) -> usize {
        self.positions.len()
    }

    /// Build a square grid layout.
    fn grid(name: &str, rows: usize, cols: usize, spacing: f32, det: f32, act: f32) -> Self {
        let mut positions = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                positions.push([c as f32 * spacing, r as f32 * spacing]);
            }
        }
        Self {
            name: name.to_string(),
            positions,
            detection_radius: det,
            activation_radius: act,
        }
    }

    /// CL1 64-channel array: 8×8 grid, 200 µm spacing.
    pub fn cl1_64ch() -> Self {
        Self::grid("CL1-64ch", 8, 8, 200.0, 100.0, 75.0)
    }

    /// MaxOne HD-MEA: 120×220 grid (26 400 electrodes), 17.5 µm spacing.
    pub fn maxone_hd() -> Self {
        Self::grid("MaxOne-HD", 120, 220, 17.5, 17.5, 17.5)
    }
}

/// Build the neuron→electrode detection map as a CSR matrix of shape
/// `(n_electrodes, n_neurons)`: entry `(e, i) = 1` when neuron `i` lies within
/// the detection radius of electrode `e`.
pub fn build_neuron_electrode_map(config: &MeaConfig, neurons: &[Position]) -> CsrMatrix {
    let r2 = config.detection_radius * config.detection_radius;
    let mut triplets = Vec::new();
    for (e, ep) in config.positions.iter().enumerate() {
        for (i, np) in neurons.iter().enumerate() {
            let dx = ep[0] - np[0];
            let dy = ep[1] - np[1];
            if dx * dx + dy * dy < r2 {
                triplets.push((e, i, 1.0));
            }
        }
    }
    CsrMatrix::from_triplets(config.n_electrodes(), neurons.len(), triplets)
}

/// Count, per electrode, how many nearby neurons fired this step
/// (`map @ spikes`).
pub fn detect_spikes(map: &CsrMatrix, spikes: &[f32]) -> Vec<f32> {
    map.matvec(spikes)
}

/// Point-source extracellular potential at each electrode (µV-scale, arbitrary
/// units): `V_e = 1/(4π σ) · Σ_n I_n / d_n`, with the distance clamped to a
/// 1 µm minimum. `currents` are per-neuron membrane currents.
pub fn compute_lfp(
    config: &MeaConfig,
    neurons: &[Position],
    currents: &[f32],
    sigma: f32,
) -> Vec<f32> {
    let k = 1.0 / (4.0 * std::f32::consts::PI * sigma);
    config
        .positions
        .iter()
        .map(|ep| {
            let mut acc = 0.0f32;
            for (np, &i_n) in neurons.iter().zip(currents) {
                let dx = ep[0] - np[0];
                let dy = ep[1] - np[1];
                let d = (dx * dx + dy * dy).sqrt().max(1.0);
                acc += i_n / d;
            }
            k * acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl1_has_64_electrodes() {
        let mea = MeaConfig::cl1_64ch();
        assert_eq!(mea.n_electrodes(), 64);
    }

    #[test]
    fn hd_mea_has_26400_electrodes() {
        assert_eq!(MeaConfig::maxone_hd().n_electrodes(), 26_400);
    }

    #[test]
    fn detection_map_links_nearby_neurons() {
        let mea = MeaConfig::cl1_64ch();
        // A neuron right on electrode 0 (position [0,0]) must be detected there.
        let neurons = vec![[0.0, 0.0], [10_000.0, 10_000.0]];
        let map = build_neuron_electrode_map(&mea, &neurons);
        let counts = detect_spikes(&map, &[1.0, 1.0]);
        assert_eq!(counts[0], 1.0, "neuron 0 detected at electrode 0");
        // The far-away neuron is out of range of every electrode.
        let total: f32 = counts.iter().sum();
        assert_eq!(total, 1.0);
    }

    #[test]
    fn lfp_decays_with_distance() {
        let mea = MeaConfig {
            name: "test".into(),
            positions: vec![[0.0, 0.0], [1000.0, 0.0]],
            detection_radius: 100.0,
            activation_radius: 75.0,
        };
        let neurons = vec![[0.0, 0.0]];
        let lfp = compute_lfp(&mea, &neurons, &[1.0], 0.3);
        assert!(
            lfp[0].abs() > lfp[1].abs(),
            "closer electrode sees larger LFP"
        );
    }
}
