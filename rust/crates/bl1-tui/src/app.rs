//! Application state and simulation driving for the TUI.

use std::fs;
use std::path::Path;

use bl1_analysis::{
    avalanche_distributions, branching_ratio, burst_statistics, detect_bursts,
    estimate_power_law_exponent,
};
use bl1_core::simulate;
use bl1_sim::{Config, Culture};

/// A selectable configuration: a display name and its raw YAML.
pub struct ConfigEntry {
    pub name: String,
    pub yaml: String,
}

/// Results of a completed preview run.
pub struct RunResult {
    pub n_neurons: usize,
    pub n_exc: usize,
    pub dt_ms: f32,
    pub duration_ms: f32,
    pub n_bursts: usize,
    pub mean_fr_hz: f64,
    pub burst_rate_per_min: f32,
    pub ibi_mean_ms: f32,
    pub ibi_cv: f32,
    pub recruitment: f32,
    pub branching_ratio: f64,
    pub avalanche_size_exp: f64,
    /// Raster kept for display (already small: capped neurons × preview steps).
    pub raster: bl1_core::Raster,
    pub n_exc_rows: usize,
}

/// Whole-application state.
pub struct App {
    pub configs: Vec<ConfigEntry>,
    pub selected: usize,
    pub neuron_cap: usize,
    pub preview_ms: f32,
    pub seed: u64,
    pub result: Option<RunResult>,
    pub status: String,
    pub should_quit: bool,
}

impl App {
    /// Build the app, discovering `*.yaml` files under `config_dir` (if any)
    /// and always offering two built-in presets.
    pub fn new(config_dir: Option<&Path>) -> Self {
        let mut configs = builtin_presets();
        if let Some(dir) = config_dir
            && let Ok(entries) = fs::read_dir(dir)
        {
            let mut files: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            files.sort();
            for path in files {
                if path.extension().and_then(|s| s.to_str()) == Some("yaml")
                    && let Ok(yaml) = fs::read_to_string(&path)
                {
                    let name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("config")
                        .to_string();
                    configs.push(ConfigEntry { name, yaml });
                }
            }
        }
        Self {
            configs,
            selected: 0,
            neuron_cap: 400,
            preview_ms: 2000.0,
            seed: 1,
            result: None,
            status: "Select a config and press Enter / r to run a preview.".to_string(),
            should_quit: false,
        }
    }

    pub fn select_next(&mut self) {
        if !self.configs.is_empty() {
            self.selected = (self.selected + 1) % self.configs.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.configs.is_empty() {
            self.selected = (self.selected + self.configs.len() - 1) % self.configs.len();
        }
    }

    pub fn increase_neurons(&mut self) {
        self.neuron_cap = (self.neuron_cap * 2).min(5000);
    }

    pub fn decrease_neurons(&mut self) {
        self.neuron_cap = (self.neuron_cap / 2).max(50);
    }

    pub fn increase_duration(&mut self) {
        self.preview_ms = (self.preview_ms * 2.0).min(20_000.0);
    }

    pub fn decrease_duration(&mut self) {
        self.preview_ms = (self.preview_ms / 2.0).max(500.0);
    }

    pub fn reseed(&mut self) {
        self.seed = self.seed.wrapping_add(1);
    }

    /// Run a capped preview simulation of the selected config.
    pub fn run_selected(&mut self) {
        let Some(entry) = self.configs.get(self.selected) else {
            self.status = "No configuration selected.".to_string();
            return;
        };
        let mut config = match Config::from_yaml_str(&entry.yaml) {
            Ok(c) => c,
            Err(e) => {
                self.status = format!("Config parse error: {e}");
                return;
            }
        };

        // Cap size for interactive responsiveness.
        config.culture.n_neurons = config.culture.n_neurons.min(self.neuron_cap).max(1);
        let dt = config.simulation.dt_ms.max(0.01);
        let duration_ms = self.preview_ms;
        let n_steps = ((duration_ms / dt).round() as usize).max(1);

        let culture = Culture::build(&config, self.seed);
        let drive = culture.background_current(n_steps, self.seed.wrapping_mul(2654435761));
        let mut state = culture.make_sim_state();
        let raster = simulate(&culture.network, &mut state, &drive, n_steps, dt);

        let thr = config.burst_detection.threshold_std;
        let min_dur = config.burst_detection.min_duration_ms;
        let bursts = detect_bursts(&raster, dt, thr, min_dur);
        let bstats = burst_statistics(&bursts, n_steps as f32 * dt);
        let sigma = branching_ratio(&raster, dt, 4.0);
        let (sizes, _dur) = avalanche_distributions(&raster, dt, 4.0);
        let size_exp = estimate_power_law_exponent(&sizes);

        self.result = Some(RunResult {
            n_neurons: culture.n_neurons(),
            n_exc: culture.n_exc,
            dt_ms: dt,
            duration_ms,
            n_bursts: bstats.n_bursts,
            mean_fr_hz: raster.mean_firing_rate_hz(dt),
            burst_rate_per_min: bstats.burst_rate_per_min,
            ibi_mean_ms: bstats.ibi_mean_ms,
            ibi_cv: bstats.ibi_cv,
            recruitment: bstats.recruitment_mean,
            branching_ratio: sigma,
            avalanche_size_exp: size_exp,
            n_exc_rows: culture.n_exc,
            raster,
        });
        self.status = format!(
            "Ran {} neurons for {:.0} ms (seed {}). {} bursts detected.",
            culture.n_neurons(),
            duration_ms,
            self.seed,
            bstats.n_bursts
        );
    }
}

/// Two always-available presets so the TUI is useful without any config files.
fn builtin_presets() -> Vec<ConfigEntry> {
    vec![
        ConfigEntry {
            name: "[preset] quick-200".to_string(),
            yaml: "culture:\n  n_neurons: 200\n  substrate_um: [800, 800]\n  p_max: 0.3\n  g_exc: 0.12\n  g_inh: 0.36\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n".to_string(),
        },
        ConfigEntry {
            name: "[preset] wagenaar-like".to_string(),
            yaml: "culture:\n  n_neurons: 1000\n  ei_ratio: 0.8\n  substrate_um: [3000, 3000]\n  lambda_um: 200.0\n  p_max: 0.21\n  g_exc: 0.12\n  g_inh: 0.36\nsimulation:\n  dt_ms: 0.5\nsynapses:\n  nmda_ratio: 0.37\nstp:\n  enabled: true\n  excitatory:\n    U: 0.30\n    tau_rec_ms: 800.0\n    tau_fac_ms: 0.001\n  inhibitory:\n    U: 0.04\n    tau_rec_ms: 100.0\n    tau_fac_ms: 1000.0\nburst_detection:\n  threshold_std: 1.5\n  min_duration_ms: 50.0\n".to_string(),
        },
    ]
}
