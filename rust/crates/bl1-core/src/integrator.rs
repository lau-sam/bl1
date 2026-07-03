//! Time-stepping loop for the forward simulation.
//!
//! Per-step ordering matches the reference `jax.lax.scan` body exactly:
//! 1. compute synaptic current from the *previous* step's conductances,
//! 2. add external drive and advance the neurons,
//! 3. propagate the new spikes (optionally scaled by short-term plasticity)
//!    through `W_exc` / `W_inh` and update the conductances.

use crate::neuron::{IzhParams, NeuronState, izhikevich_step};
use crate::plasticity::{StpParams, StpState};
use crate::sparse::CsrMatrix;
use crate::synapse::{SynapseState, synaptic_current_into};

/// A network ready to simulate. Excitatory presynaptic connections live in
/// `w_exc` (feeding AMPA + NMDA), inhibitory in `w_inh` (feeding GABA).
#[derive(Debug, Clone)]
pub struct Network {
    pub w_exc: CsrMatrix,
    pub w_inh: CsrMatrix,
    pub izh: IzhParams,
    pub is_excitatory: Vec<bool>,
}

impl Network {
    /// Number of neurons.
    pub fn n_neurons(&self) -> usize {
        self.izh.a.len()
    }
}

/// Binary spike raster of shape `(n_steps, n_neurons)` stored row-major.
#[derive(Debug, Clone)]
pub struct Raster {
    pub n_steps: usize,
    pub n_neurons: usize,
    pub data: Vec<f32>,
}

impl Raster {
    /// Row `t` (all neurons at timestep `t`).
    pub fn row(&self, t: usize) -> &[f32] {
        &self.data[t * self.n_neurons..(t + 1) * self.n_neurons]
    }

    /// Total number of spikes in the raster.
    pub fn total_spikes(&self) -> f64 {
        self.data.iter().map(|&x| x as f64).sum()
    }

    /// Mean firing rate (Hz) across all neurons over the recording.
    pub fn mean_firing_rate_hz(&self, dt_ms: f32) -> f64 {
        if self.n_neurons == 0 || self.n_steps == 0 {
            return 0.0;
        }
        let total_time_s = self.n_steps as f64 * dt_ms as f64 / 1000.0;
        self.total_spikes() / (self.n_neurons as f64 * total_time_s)
    }
}

/// Simulation state carried across steps.
pub struct SimState {
    pub neuron: NeuronState,
    pub syn: SynapseState,
    pub stp: Option<(StpParams, StpState)>,
}

impl SimState {
    /// Fresh state for `network` with optional short-term plasticity.
    pub fn new(network: &Network, stp_params: Option<StpParams>) -> Self {
        let n = network.n_neurons();
        let stp = stp_params.map(|p| {
            let s = StpState::init(&p);
            (p, s)
        });
        Self {
            neuron: NeuronState::resting(&network.izh),
            syn: SynapseState::zeros(n),
            stp,
        }
    }
}

/// Run `n_steps` of the network under external current `i_external`
/// (`n_steps × n_neurons`, row-major) and return the spike raster.
///
/// `state` is advanced in place so the caller can inspect the final state.
pub fn simulate(
    network: &Network,
    state: &mut SimState,
    i_external: &[f32],
    n_steps: usize,
    dt: f32,
) -> Raster {
    let n = network.n_neurons();
    assert_eq!(
        i_external.len(),
        n_steps * n,
        "i_external must be n_steps × n_neurons"
    );
    let mut raster = Raster {
        n_steps,
        n_neurons: n,
        data: vec![0.0; n_steps * n],
    };
    let mut i_syn = vec![0.0f32; n];
    let mut i_total = vec![0.0f32; n];
    let mut exc_input = vec![0.0f32; n];
    let mut inh_input = vec![0.0f32; n];

    for t in 0..n_steps {
        // 1. Synaptic current from previous conductances.
        synaptic_current_into(&state.syn, &state.neuron.v, &mut i_syn);
        // 2. Add external drive, advance neurons.
        let i_ext_row = &i_external[t * n..(t + 1) * n];
        for j in 0..n {
            i_total[j] = i_syn[j] + i_ext_row[j];
        }
        izhikevich_step(&mut state.neuron, &network.izh, &i_total, dt);
        raster.data[t * n..(t + 1) * n].copy_from_slice(&state.neuron.spikes);

        // 3. Propagate spikes (scaled by STP if enabled) into conductances.
        // The raster already holds the binary spikes captured above; the drive
        // vector may be a non-binary STP release scale.
        let stp_scale;
        let drive: &[f32] = match &mut state.stp {
            Some((params, stp)) => {
                stp_scale = stp.step(params, &state.neuron.spikes, dt);
                &stp_scale
            }
            None => &state.neuron.spikes,
        };
        network.w_exc.matvec_into(drive, &mut exc_input);
        network.w_inh.matvec_into(drive, &mut inh_input);
        state.syn.step(&exc_input, &inh_input, dt);
    }
    raster
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_rs() -> Network {
        Network {
            w_exc: CsrMatrix::from_triplets(1, 1, vec![]),
            w_inh: CsrMatrix::from_triplets(1, 1, vec![]),
            izh: IzhParams {
                a: vec![0.02],
                b: vec![0.2],
                c: vec![-65.0],
                d: vec![8.0],
            },
            is_excitatory: vec![true],
        }
    }

    #[test]
    fn silent_without_drive() {
        let net = single_rs();
        let mut st = SimState::new(&net, None);
        let raster = simulate(&net, &mut st, &vec![0.0; 400], 400, 0.5);
        assert_eq!(raster.total_spikes(), 0.0);
    }

    #[test]
    fn fires_under_constant_drive() {
        let net = single_rs();
        let mut st = SimState::new(&net, None);
        let raster = simulate(&net, &mut st, &vec![10.0; 400], 400, 0.5);
        assert!(raster.total_spikes() > 0.0);
        assert!(raster.mean_firing_rate_hz(0.5) > 0.0);
    }

    #[test]
    fn recurrent_excitation_propagates() {
        // Two excitatory neurons, 0 -> 1 with a strong weight. Drive neuron 0;
        // neuron 1 must eventually receive input and fire.
        let net = Network {
            w_exc: CsrMatrix::from_triplets(2, 2, vec![(1, 0, 5.0)]),
            w_inh: CsrMatrix::from_triplets(2, 2, vec![]),
            izh: IzhParams {
                a: vec![0.02, 0.02],
                b: vec![0.2, 0.2],
                c: vec![-65.0, -65.0],
                d: vec![8.0, 8.0],
            },
            is_excitatory: vec![true, true],
        };
        let mut st = SimState::new(&net, None);
        let mut i_ext = vec![0.0f32; 2 * 800];
        for t in 0..800 {
            i_ext[t * 2] = 10.0; // drive only neuron 0
        }
        let raster = simulate(&net, &mut st, &i_ext, 800, 0.5);
        let n1_spikes: f32 = (0..800).map(|t| raster.row(t)[1]).sum();
        assert!(n1_spikes > 0.0, "downstream neuron never fired");
    }

    #[test]
    fn stp_runs_and_records_binary_spikes() {
        let net = single_rs();
        let mut st = SimState::new(&net, Some(StpParams::excitatory(1)));
        let raster = simulate(&net, &mut st, &vec![10.0; 400], 400, 0.5);
        // Raster stays binary even with STP scaling on the drive path.
        assert!(raster.data.iter().all(|&x| x == 0.0 || x == 1.0));
        assert!(raster.total_spikes() > 0.0);
    }
}
