//! The `Culture` factory: turn a [`Config`] into a ready-to-simulate network.

use crate::config::Config;
use crate::connectivity::build_connectivity;
use crate::placement::{Position, place_neurons};
use bl1_core::{Network, SimState, StpParams, SynapseState, build_population};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// A constructed culture: the simulable network plus the metadata needed to
/// build matching simulation state and drive.
pub struct Culture {
    pub network: Network,
    pub positions: Vec<Position>,
    pub n_exc: usize,
    pub stp_params: Option<StpParams>,
    pub nmda_ratio: f32,
    pub gaba_b_ratio: f32,
    pub dt_ms: f32,
    pub bg_mean: f32,
    pub bg_std: f32,
}

impl Culture {
    /// Build a culture from `config`, seeding all randomness with `seed` for
    /// reproducibility.
    pub fn build(config: &Config, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        let c = &config.culture;

        let pop = build_population(c.n_neurons, c.ei_ratio);
        let positions = place_neurons(&mut rng, c.n_neurons, c.substrate_um);
        let (w_exc, w_inh) = build_connectivity(
            &mut rng,
            &positions,
            &pop.is_excitatory,
            c.lambda_um,
            c.p_max,
            c.g_exc,
            c.g_inh,
        );

        let stp_params = if config.stp.enabled {
            let e = &config.stp.excitatory;
            let i = &config.stp.inhibitory;
            let n = c.n_neurons;
            let mut u = Vec::with_capacity(n);
            let mut tau_rec = Vec::with_capacity(n);
            let mut tau_fac = Vec::with_capacity(n);
            for &exc in &pop.is_excitatory {
                let r = if exc { e } else { i };
                u.push(r.u);
                tau_rec.push(r.tau_rec_ms);
                tau_fac.push(r.tau_fac_ms);
            }
            Some(StpParams {
                u,
                tau_rec,
                tau_fac,
            })
        } else {
            None
        };

        let network = Network {
            w_exc,
            w_inh,
            izh: pop.params,
            is_excitatory: pop.is_excitatory,
        };

        Self {
            network,
            positions,
            n_exc: pop.n_exc,
            stp_params,
            nmda_ratio: config.synapses.nmda_ratio,
            gaba_b_ratio: config.synapses.gaba_b_ratio,
            dt_ms: config.simulation.dt_ms,
            bg_mean: config.background.mean,
            bg_std: config.background.std,
        }
    }

    /// Number of neurons.
    pub fn n_neurons(&self) -> usize {
        self.network.n_neurons()
    }

    /// Build a fresh simulation state with the configured STP and receptor split.
    pub fn make_sim_state(&self) -> SimState {
        let mut state = SimState::new(&self.network, self.stp_params.clone());
        state.syn =
            SynapseState::with_receptor_split(self.n_neurons(), self.nmda_ratio, self.gaba_b_ratio);
        state
    }

    /// Generate `n_steps × n_neurons` of Gaussian background drive
    /// (`mean ± std`), seeded independently for reproducibility.
    pub fn background_current(&self, n_steps: usize, seed: u64) -> Vec<f32> {
        let n = self.n_neurons();
        let mut rng = Pcg64::seed_from_u64(seed);
        let mut out = vec![0.0f32; n_steps * n];
        for v in out.iter_mut() {
            *v = self.bg_mean + self.bg_std * gaussian(&mut rng);
        }
        out
    }
}

/// Standard normal sample via the Box-Muller transform.
fn gaussian<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bl1_core::simulate;

    fn small_config() -> Config {
        Config::from_yaml_str(
            "culture:\n  n_neurons: 200\n  substrate_um: [800, 800]\n  p_max: 0.3\nsimulation:\n  dt_ms: 0.5\n  duration_ms: 1000\nstp:\n  enabled: true\n",
        )
        .unwrap()
    }

    #[test]
    fn build_is_reproducible() {
        let cfg = small_config();
        let a = Culture::build(&cfg, 42);
        let b = Culture::build(&cfg, 42);
        assert_eq!(a.network.w_exc.nnz(), b.network.w_exc.nnz());
        assert_eq!(a.positions, b.positions);
    }

    #[test]
    fn culture_simulates_and_spikes() {
        let cfg = small_config();
        let culture = Culture::build(&cfg, 7);
        let n_steps = 2000;
        let drive = culture.background_current(n_steps, 100);
        let mut state = culture.make_sim_state();
        let raster = simulate(&culture.network, &mut state, &drive, n_steps, culture.dt_ms);
        assert!(raster.total_spikes() > 0.0, "culture should be active");
        assert!(raster.data.iter().all(|&x| x == 0.0 || x == 1.0));
    }

    #[test]
    fn stp_disabled_yields_no_stp_params() {
        let cfg = Config::from_yaml_str("culture:\n  n_neurons: 50\n").unwrap();
        let culture = Culture::build(&cfg, 1);
        assert!(culture.stp_params.is_none());
    }
}
