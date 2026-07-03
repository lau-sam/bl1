//! YAML experiment configuration, compatible with the project's `configs/*.yaml`.
//!
//! Unknown fields are ignored, so files carrying extra sections load fine.

use serde::Deserialize;

/// Top-level configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub culture: CultureParams,
    #[serde(default)]
    pub simulation: SimParams,
    #[serde(default)]
    pub background: Background,
    #[serde(default)]
    pub synapses: Synapses,
    #[serde(default)]
    pub stp: StpConfig,
    #[serde(default)]
    pub burst_detection: BurstDetection,
}

impl Config {
    /// Parse a configuration from a YAML string.
    pub fn from_yaml_str(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }
}

/// Culture / network parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct CultureParams {
    pub n_neurons: usize,
    #[serde(default = "d_ei_ratio")]
    pub ei_ratio: f32,
    #[serde(default = "d_substrate")]
    pub substrate_um: [f32; 2],
    #[serde(default = "d_lambda")]
    pub lambda_um: f32,
    #[serde(default = "d_p_max")]
    pub p_max: f32,
    #[serde(default = "d_g_exc")]
    pub g_exc: f32,
    #[serde(default = "d_g_inh")]
    pub g_inh: f32,
}

/// Simulation timing.
#[derive(Debug, Clone, Deserialize)]
pub struct SimParams {
    #[serde(default = "d_dt")]
    pub dt_ms: f32,
    #[serde(default = "d_duration")]
    pub duration_ms: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            dt_ms: d_dt(),
            duration_ms: d_duration(),
        }
    }
}

/// Background tonic drive (per-neuron Gaussian current each step).
#[derive(Debug, Clone, Deserialize)]
pub struct Background {
    #[serde(default = "d_bg_mean")]
    pub mean: f32,
    #[serde(default = "d_bg_std")]
    pub std: f32,
}

impl Default for Background {
    fn default() -> Self {
        Self {
            mean: d_bg_mean(),
            std: d_bg_std(),
        }
    }
}

/// Receptor split of the excitatory/inhibitory conductances.
#[derive(Debug, Clone, Deserialize)]
pub struct Synapses {
    #[serde(default = "d_nmda_ratio")]
    pub nmda_ratio: f32,
    #[serde(default)]
    pub gaba_b_ratio: f32,
}

impl Default for Synapses {
    fn default() -> Self {
        Self {
            nmda_ratio: d_nmda_ratio(),
            gaba_b_ratio: 0.0,
        }
    }
}

/// Short-term-plasticity configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct StpConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "d_stp_exc")]
    pub excitatory: StpRegime,
    #[serde(default = "d_stp_inh")]
    pub inhibitory: StpRegime,
}

impl Default for StpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            excitatory: d_stp_exc(),
            inhibitory: d_stp_inh(),
        }
    }
}

/// One STP regime (release probability and time constants).
#[derive(Debug, Clone, Deserialize)]
pub struct StpRegime {
    #[serde(rename = "U")]
    pub u: f32,
    pub tau_rec_ms: f32,
    pub tau_fac_ms: f32,
}

/// Burst-detection thresholds.
#[derive(Debug, Clone, Deserialize)]
pub struct BurstDetection {
    #[serde(default = "d_threshold_std")]
    pub threshold_std: f32,
    #[serde(default = "d_min_duration")]
    pub min_duration_ms: f32,
}

impl Default for BurstDetection {
    fn default() -> Self {
        Self {
            threshold_std: d_threshold_std(),
            min_duration_ms: d_min_duration(),
        }
    }
}

// --- Defaults (Wagenaar-calibrated where applicable) -----------------------
fn d_ei_ratio() -> f32 {
    0.8
}
fn d_substrate() -> [f32; 2] {
    [3000.0, 3000.0]
}
fn d_lambda() -> f32 {
    200.0
}
fn d_p_max() -> f32 {
    0.21
}
fn d_g_exc() -> f32 {
    0.12
}
fn d_g_inh() -> f32 {
    0.36
}
fn d_dt() -> f32 {
    0.5
}
fn d_duration() -> f32 {
    60_000.0
}
fn d_bg_mean() -> f32 {
    1.0
}
fn d_bg_std() -> f32 {
    3.0
}
fn d_nmda_ratio() -> f32 {
    0.37
}
fn d_threshold_std() -> f32 {
    1.5
}
fn d_min_duration() -> f32 {
    50.0
}
fn d_stp_exc() -> StpRegime {
    StpRegime {
        u: 0.30,
        tau_rec_ms: 800.0,
        tau_fac_ms: 0.001,
    }
}
fn d_stp_inh() -> StpRegime {
    StpRegime {
        u: 0.04,
        tau_rec_ms: 100.0,
        tau_fac_ms: 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wagenaar_style_config() {
        let yaml = r#"
culture:
  n_neurons: 500
  ei_ratio: 0.8
  neuron_model: izhikevich
  substrate_um: [3000, 3000]
  lambda_um: 200.0
  p_max: 0.21
  g_exc: 0.12
  g_inh: 0.36
simulation:
  dt_ms: 0.5
  duration_ms: 5000
synapses:
  nmda_ratio: 0.37
stp:
  enabled: true
  excitatory:
    U: 0.30
    tau_rec_ms: 800.0
    tau_fac_ms: 0.001
  inhibitory:
    U: 0.04
    tau_rec_ms: 100.0
    tau_fac_ms: 1000.0
burst_detection:
  threshold_std: 1.5
  min_duration_ms: 50.0
"#;
        let cfg = Config::from_yaml_str(yaml).unwrap();
        assert_eq!(cfg.culture.n_neurons, 500);
        assert_eq!(cfg.culture.p_max, 0.21);
        assert!(cfg.stp.enabled);
        assert_eq!(cfg.stp.excitatory.u, 0.30);
        assert_eq!(cfg.burst_detection.threshold_std, 1.5);
        assert_eq!(cfg.synapses.nmda_ratio, 0.37);
    }

    #[test]
    fn minimal_config_uses_defaults() {
        let cfg = Config::from_yaml_str("culture:\n  n_neurons: 100\n").unwrap();
        assert_eq!(cfg.culture.n_neurons, 100);
        assert_eq!(cfg.culture.ei_ratio, 0.8);
        assert_eq!(cfg.simulation.dt_ms, 0.5);
        assert_eq!(cfg.burst_detection.threshold_std, 1.5);
        assert!(!cfg.stp.enabled);
    }
}
