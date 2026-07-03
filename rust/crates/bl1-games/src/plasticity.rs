//! Reward-modulated STDP (three-factor rule; Izhikevich 2007).
//!
//! Plain STDP is Hebbian and outcome-blind, so it has no reason to learn to
//! *win*. The three-factor rule adds a global neuromodulator (a dopamine-like
//! reward signal): coincident pre/post spikes lay down a slowly decaying
//! **eligibility trace** on each synapse, and the weight only changes when a
//! reward arrives — `dW = lr · reward · eligibility`. A hit delivers a positive
//! reward, so synapses whose recent activity led to a hit are strengthened. The
//! eligibility trace bridges the delay between action and outcome (the "distal
//! reward problem"). Miss-punishment is supported but off by default (see
//! [`Reward`]): net-negative reward erodes the working circuit before the
//! culture reliably hits.

use bl1_core::CsrMatrix;

/// Parameters for the eligibility trace and reward-gated weight update.
#[derive(Debug, Clone)]
pub struct ThreeFactorParams {
    /// Presynaptic trace increment on a spike.
    pub a_plus: f32,
    /// Postsynaptic trace increment on a spike.
    pub a_minus: f32,
    /// Presynaptic trace time constant (ms).
    pub tau_plus: f32,
    /// Postsynaptic trace time constant (ms).
    pub tau_minus: f32,
    /// Eligibility-trace time constant (ms) — bridges the action→reward delay.
    pub tau_elig: f32,
    /// Weight-update gain.
    pub learning_rate: f32,
    pub w_max: f32,
    pub w_min: f32,
}

impl Default for ThreeFactorParams {
    fn default() -> Self {
        Self {
            a_plus: 1.0,
            a_minus: 1.0,
            tau_plus: 20.0,
            tau_minus: 40.0,
            tau_elig: 1000.0,
            learning_rate: 0.008,
            w_max: 0.5,
            w_min: 0.0,
        }
    }
}

/// A dopamine-like scalar reward that jumps on game events and decays over time.
#[derive(Debug, Clone)]
pub struct Reward {
    pub level: f32,
    pub tau_ms: f32,
    pub hit_amp: f32,
    pub miss_amp: f32,
}

impl Default for Reward {
    fn default() -> Self {
        Self {
            level: 0.0,
            tau_ms: 200.0,
            hit_amp: 1.0,
            // Reward-only by default: punishing every miss net-depresses the
            // working tracking circuit while the culture still misses more than
            // it hits. Reinforcing hits alone preserves the reflex and can only
            // strengthen what already works. (A reward-prediction-error baseline
            // that safely re-enables punishment is future tuning.)
            miss_amp: 0.0,
        }
    }
}

impl Reward {
    /// Decay the reward level one step toward zero.
    pub fn decay(&mut self, dt: f32) {
        self.level *= (-dt / self.tau_ms).exp();
    }

    /// Add a positive (hit) or negative (miss) reward pulse.
    pub fn reward(&mut self) {
        self.level += self.hit_amp;
    }

    pub fn punish(&mut self) {
        self.level -= self.miss_amp;
    }
}

/// Reward-modulated STDP state: neuron traces plus a per-synapse eligibility
/// trace parallel to `w_exc.data`.
pub struct ThreeFactorStdp {
    pub params: ThreeFactorParams,
    pre_trace: Vec<f32>,
    post_trace: Vec<f32>,
    /// One eligibility value per stored `w_exc` entry (same order as `data`).
    eligibility: Vec<f32>,
}

impl ThreeFactorStdp {
    /// `n_neurons` traces and `nnz` eligibility slots (= `w_exc.data.len()`).
    pub fn new(n_neurons: usize, nnz: usize, params: ThreeFactorParams) -> Self {
        Self {
            params,
            pre_trace: vec![0.0; n_neurons],
            post_trace: vec![0.0; n_neurons],
            eligibility: vec![0.0; nnz],
        }
    }

    /// Advance one step: decay traces, deposit current spikes, accumulate the
    /// STDP kernel into each synapse's eligibility, and — when `reward` is
    /// non-zero — nudge the weights by `lr · reward · eligibility` (clamped).
    pub fn step(&mut self, spikes: &[f32], w_exc: &mut CsrMatrix, reward: f32, dt: f32) {
        let p = &self.params;
        let decay_pre = (-dt / p.tau_plus).exp();
        let decay_post = (-dt / p.tau_minus).exp();
        let decay_elig = (-dt / p.tau_elig).exp();

        for t in &mut self.pre_trace {
            *t *= decay_pre;
        }
        for t in &mut self.post_trace {
            *t *= decay_post;
        }
        for (i, &s) in spikes.iter().enumerate() {
            if s != 0.0 {
                self.pre_trace[i] += p.a_plus;
                self.post_trace[i] += p.a_minus;
            }
        }

        let apply = reward != 0.0;
        for j in 0..w_exc.n_rows {
            let start = w_exc.indptr[j];
            let end = w_exc.indptr[j + 1];
            for k in start..end {
                let i = w_exc.indices[k];
                let kernel = spikes[j] * self.pre_trace[i] - self.post_trace[j] * spikes[i];
                let e = self.eligibility[k] * decay_elig + kernel;
                self.eligibility[k] = e;
                if apply {
                    let w = w_exc.data[k] + p.learning_rate * reward * e;
                    w_exc.data[k] = w.clamp(p.w_min, p.w_max);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_decays_toward_zero() {
        let mut r = Reward::default();
        r.reward();
        let a = r.level;
        r.decay(50.0);
        assert!(r.level < a && r.level > 0.0);
    }

    #[test]
    fn positive_reward_potentiates_eligible_synapse() {
        // Connection (post=0, pre=1). Drive pre then post to build eligibility,
        // then deliver a positive reward: the weight should grow.
        let params = ThreeFactorParams::default();
        let mut w = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 0.05)]);
        let mut tf = ThreeFactorStdp::new(2, w.data.len(), params);
        tf.step(&[0.0, 1.0], &mut w, 0.0, 0.5); // pre fires, no reward yet
        tf.step(&[1.0, 0.0], &mut w, 0.0, 0.5); // post fires -> eligibility up
        let before = w.data[0];
        tf.step(&[0.0, 0.0], &mut w, 1.0, 0.5); // reward arrives
        assert!(
            w.data[0] > before,
            "reward should potentiate: {before} -> {}",
            w.data[0]
        );
    }

    #[test]
    fn negative_reward_depresses_eligible_synapse() {
        let params = ThreeFactorParams::default();
        let mut w = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 0.2)]);
        let mut tf = ThreeFactorStdp::new(2, w.data.len(), params);
        tf.step(&[0.0, 1.0], &mut w, 0.0, 0.5);
        tf.step(&[1.0, 0.0], &mut w, 0.0, 0.5);
        let before = w.data[0];
        tf.step(&[0.0, 0.0], &mut w, -1.0, 0.5); // punishment
        assert!(
            w.data[0] < before,
            "punishment should depress: {before} -> {}",
            w.data[0]
        );
    }

    #[test]
    fn no_reward_leaves_weights_unchanged() {
        let params = ThreeFactorParams::default();
        let mut w = CsrMatrix::from_triplets(2, 2, vec![(0, 1, 0.1)]);
        let mut tf = ThreeFactorStdp::new(2, w.data.len(), params);
        tf.step(&[0.0, 1.0], &mut w, 0.0, 0.5);
        tf.step(&[1.0, 0.0], &mut w, 0.0, 0.5);
        assert_eq!(w.data[0], 0.1, "no reward -> no weight change");
    }
}
