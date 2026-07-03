//! Short-term (Tsodyks-Markram) and spike-timing-dependent plasticity.
//!
//! Short-term plasticity produces a per-presynaptic multiplicative `scale`
//! applied to outgoing spikes; STDP modifies the excitatory weight matrix in
//! place from pre/post spike traces (Morrison et al. 2008).

use crate::sparse::CsrMatrix;

/// Per-presynaptic short-term-plasticity parameters.
#[derive(Debug, Clone)]
pub struct StpParams {
    /// Baseline release probability `U`.
    pub u: Vec<f32>,
    /// Recovery time constant (ms).
    pub tau_rec: Vec<f32>,
    /// Facilitation time constant (ms). Use a tiny positive value (e.g. 1e-3)
    /// for purely depressing synapses to avoid division by zero.
    pub tau_fac: Vec<f32>,
}

impl StpParams {
    /// Depressing excitatory defaults: `U = 0.5`, `tau_rec = 800`, `tau_fac ≈ 0`.
    pub fn excitatory(n: usize) -> Self {
        Self {
            u: vec![0.5; n],
            tau_rec: vec![800.0; n],
            tau_fac: vec![0.001; n],
        }
    }

    /// Facilitating inhibitory defaults: `U = 0.04`, `tau_rec = 100`, `tau_fac = 1000`.
    pub fn inhibitory(n: usize) -> Self {
        Self {
            u: vec![0.04; n],
            tau_rec: vec![100.0; n],
            tau_fac: vec![1000.0; n],
        }
    }
}

/// STP dynamic state: available resources `x` and utilisation `u`.
#[derive(Debug, Clone)]
pub struct StpState {
    pub x: Vec<f32>,
    pub u: Vec<f32>,
}

impl StpState {
    /// Initialise `x = 1`, `u = U`.
    pub fn init(params: &StpParams) -> Self {
        Self {
            x: vec![1.0; params.u.len()],
            u: params.u.clone(),
        }
    }

    /// Advance one step and return the per-presynaptic release `scale`
    /// (`u * x` on spiking neurons, 0 elsewhere) to multiply outgoing spikes.
    ///
    /// Recovery of `x` and decay of `u` toward baseline happen first, then the
    /// spike applies facilitation/depression.
    pub fn step(&mut self, params: &StpParams, spikes: &[f32], dt: f32) -> Vec<f32> {
        let n = self.x.len();
        let mut scale = vec![0.0f32; n];
        for i in 0..n {
            let decay_x = (-dt / params.tau_rec[i]).exp();
            let x_rec = 1.0 - (1.0 - self.x[i]) * decay_x;
            let decay_u = (-dt / params.tau_fac[i]).exp();
            let u_dec = params.u[i] + (self.u[i] - params.u[i]) * decay_u;

            if spikes[i] != 0.0 {
                let u_spike = u_dec + params.u[i] * (1.0 - u_dec);
                scale[i] = u_spike * x_rec;
                self.u[i] = u_spike;
                self.x[i] = x_rec * (1.0 - u_spike);
            } else {
                self.u[i] = u_dec;
                self.x[i] = x_rec;
            }
        }
        scale
    }
}

/// STDP parameters (trace-based, Morrison et al. 2008). `A_minus > A_plus`
/// gives a slight depression bias for stability.
#[derive(Debug, Clone)]
pub struct StdpParams {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_max: f32,
    pub w_min: f32,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.005,
            a_minus: 0.00525,
            tau_plus: 20.0,
            tau_minus: 50.0,
            w_max: 0.1,
            w_min: 0.0,
        }
    }
}

/// STDP eligibility traces, one entry per neuron.
#[derive(Debug, Clone)]
pub struct StdpState {
    pub pre_trace: Vec<f32>,
    pub post_trace: Vec<f32>,
}

impl StdpState {
    pub fn zeros(n: usize) -> Self {
        Self {
            pre_trace: vec![0.0; n],
            post_trace: vec![0.0; n],
        }
    }

    /// Decay traces, deposit the current spikes, and update the stored weights
    /// of `w_exc` (rows = postsynaptic `j`, cols = presynaptic `i`) in place.
    ///
    /// For each connection `(j, i)`:
    /// `dW = spikes[j]·pre_trace[i]  −  post_trace[j]·spikes[i]`,
    /// clamped to `[w_min, w_max]`. Sparsity (existing entries) is preserved.
    pub fn step(&mut self, params: &StdpParams, spikes: &[f32], w_exc: &mut CsrMatrix, dt: f32) {
        let decay_pre = (-dt / params.tau_plus).exp();
        let decay_post = (-dt / params.tau_minus).exp();
        for t in &mut self.pre_trace {
            *t *= decay_pre;
        }
        for t in &mut self.post_trace {
            *t *= decay_post;
        }
        for (i, &s) in spikes.iter().enumerate() {
            if s != 0.0 {
                self.pre_trace[i] += params.a_plus;
                self.post_trace[i] += params.a_minus;
            }
        }
        for j in 0..w_exc.n_rows {
            let start = w_exc.indptr[j];
            let end = w_exc.indptr[j + 1];
            for k in start..end {
                let i = w_exc.indices[k];
                let ltp = spikes[j] * self.pre_trace[i];
                let ltd = self.post_trace[j] * spikes[i];
                let w = w_exc.data[k] + (ltp - ltd);
                w_exc.data[k] = w.clamp(params.w_min, params.w_max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depressing_synapse_scale_shrinks_on_repeated_spikes() {
        let params = StpParams::excitatory(1);
        let mut state = StpState::init(&params);
        // Two spikes 5 ms apart: the second release should be weaker.
        let first = state.step(&params, &[1.0], 0.5)[0];
        for _ in 0..9 {
            state.step(&params, &[0.0], 0.5);
        }
        let second = state.step(&params, &[1.0], 0.5)[0];
        assert!(first > 0.0);
        assert!(second < first, "depression: {second} !< {first}");
    }

    #[test]
    fn no_spike_gives_zero_scale() {
        let params = StpParams::excitatory(2);
        let mut state = StpState::init(&params);
        let scale = state.step(&params, &[0.0, 0.0], 0.5);
        assert_eq!(scale, vec![0.0, 0.0]);
    }

    #[test]
    fn stdp_potentiates_when_post_follows_pre() {
        // Connection (post=0, pre=1). Pre fires, then post fires one step later:
        // pre_trace[1] is positive when post spikes -> LTP.
        let params = StdpParams::default();
        let mut w = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 0.05)]);
        let mut st = StdpState::zeros(2);
        st.step(&params, &[0.0, 1.0], &mut w, 0.5); // pre (neuron 1) fires
        st.step(&params, &[1.0, 0.0], &mut w, 0.5); // post (neuron 0) fires
        assert!(
            w.data[0] > 0.05,
            "weight should potentiate, got {}",
            w.data[0]
        );
    }

    #[test]
    fn stdp_respects_weight_bounds() {
        let params = StdpParams {
            w_max: 0.06,
            ..Default::default()
        };
        let mut w = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 0.059)]);
        let mut st = StdpState::zeros(2);
        for _ in 0..50 {
            st.step(&params, &[0.0, 1.0], &mut w, 0.5);
            st.step(&params, &[1.0, 0.0], &mut w, 0.5);
        }
        assert!(w.data[0] <= 0.06 + 1e-6, "clamped to w_max");
    }
}
