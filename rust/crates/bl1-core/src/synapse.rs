//! Conductance-based synapses: AMPA, GABA_A (single-exponential) and NMDA,
//! GABA_B (dual-exponential with peak normalisation), plus the NMDA Mg²⁺
//! voltage-dependent block. Constants follow the reference Python model.

/// AMPA decay time constant (ms).
pub const TAU_AMPA: f32 = 2.0;
/// AMPA reversal potential (mV).
pub const E_AMPA: f32 = 0.0;
/// GABA_A decay time constant (ms).
pub const TAU_GABA_A: f32 = 6.0;
/// GABA_A reversal potential (mV).
pub const E_GABA_A: f32 = -75.0;

/// NMDA rise/decay time constants (ms) and reversal (mV).
pub const TAU_NMDA_RISE: f32 = 2.0;
pub const TAU_NMDA_DECAY: f32 = 100.0;
pub const E_NMDA: f32 = 0.0;
/// Extracellular Mg²⁺ concentration (mM) for the NMDA block.
pub const MG_CONC: f32 = 1.0;

/// GABA_B rise/decay time constants (ms) and reversal (mV).
pub const TAU_GABA_B_RISE: f32 = 45.0;
pub const TAU_GABA_B_DECAY: f32 = 170.0;
pub const E_GABA_B: f32 = -95.0;

/// Peak-normalisation factor for a dual-exponential conductance so that its
/// peak amplitude per unit input equals 1.
fn dual_exp_norm(tau_rise: f32, tau_decay: f32) -> f32 {
    let t_peak = tau_decay * tau_rise / (tau_decay - tau_rise) * (tau_decay / tau_rise).ln();
    let peak = (-t_peak / tau_decay).exp() - (-t_peak / tau_rise).exp();
    1.0 / peak
}

/// Per-receptor conductance state, one entry per postsynaptic neuron.
#[derive(Debug, Clone)]
pub struct SynapseState {
    pub g_ampa: Vec<f32>,
    pub g_gaba_a: Vec<f32>,
    pub g_nmda_rise: Vec<f32>,
    pub g_nmda_decay: Vec<f32>,
    pub g_gaba_b_rise: Vec<f32>,
    pub g_gaba_b_decay: Vec<f32>,
    nmda_norm: f32,
    gaba_b_norm: f32,
}

impl SynapseState {
    /// Zero-initialised conductances for `n` neurons.
    pub fn zeros(n: usize) -> Self {
        Self {
            g_ampa: vec![0.0; n],
            g_gaba_a: vec![0.0; n],
            g_nmda_rise: vec![0.0; n],
            g_nmda_decay: vec![0.0; n],
            g_gaba_b_rise: vec![0.0; n],
            g_gaba_b_decay: vec![0.0; n],
            nmda_norm: dual_exp_norm(TAU_NMDA_RISE, TAU_NMDA_DECAY),
            gaba_b_norm: dual_exp_norm(TAU_GABA_B_RISE, TAU_GABA_B_DECAY),
        }
    }

    pub fn len(&self) -> usize {
        self.g_ampa.len()
    }

    pub fn is_empty(&self) -> bool {
        self.g_ampa.is_empty()
    }

    /// Advance all receptor conductances one step of `dt` ms.
    ///
    /// `exc_input` is the excitatory presynaptic drive `W_exc @ drive` (feeds
    /// AMPA + NMDA); `inh_input` is the inhibitory drive `W_inh @ drive`
    /// (feeds GABA_A + GABA_B).
    pub fn step(&mut self, exc_input: &[f32], inh_input: &[f32], dt: f32) {
        let decay_ampa = (-dt / TAU_AMPA).exp();
        let decay_gaba_a = (-dt / TAU_GABA_A).exp();
        let decay_nmda_r = (-dt / TAU_NMDA_RISE).exp();
        let decay_nmda_d = (-dt / TAU_NMDA_DECAY).exp();
        let decay_gaba_b_r = (-dt / TAU_GABA_B_RISE).exp();
        let decay_gaba_b_d = (-dt / TAU_GABA_B_DECAY).exp();
        for j in 0..self.len() {
            // Single-exponential receptors.
            self.g_ampa[j] = self.g_ampa[j] * decay_ampa + exc_input[j];
            self.g_gaba_a[j] = self.g_gaba_a[j] * decay_gaba_a + inh_input[j];
            // Dual-exponential receptors.
            self.g_nmda_rise[j] =
                self.g_nmda_rise[j] * decay_nmda_r + exc_input[j] * self.nmda_norm;
            self.g_nmda_decay[j] =
                self.g_nmda_decay[j] * decay_nmda_d + exc_input[j] * self.nmda_norm;
            self.g_gaba_b_rise[j] =
                self.g_gaba_b_rise[j] * decay_gaba_b_r + inh_input[j] * self.gaba_b_norm;
            self.g_gaba_b_decay[j] =
                self.g_gaba_b_decay[j] * decay_gaba_b_d + inh_input[j] * self.gaba_b_norm;
        }
    }
}

/// NMDA Mg²⁺ voltage-dependent block `B(v) = 1 / (1 + (Mg/3.57) e^{-0.062 v})`.
pub fn nmda_mg_block(v: f32) -> f32 {
    1.0 / (1.0 + (MG_CONC / 3.57) * (-0.062 * v).exp())
}

/// Total synaptic current per neuron given the current conductances and
/// membrane potentials, written into `out`.
///
/// `I = g_ampa (E_AMPA - v) + g_nmda B(v) (E_NMDA - v)
///      + g_gaba_a (E_GABA_A - v) + g_gaba_b (E_GABA_B - v)`
pub fn synaptic_current_into(syn: &SynapseState, v: &[f32], out: &mut [f32]) {
    for j in 0..syn.len() {
        let vj = v[j];
        let g_nmda = syn.g_nmda_decay[j] - syn.g_nmda_rise[j];
        let g_gaba_b = syn.g_gaba_b_decay[j] - syn.g_gaba_b_rise[j];
        out[j] = syn.g_ampa[j] * (E_AMPA - vj)
            + g_nmda * nmda_mg_block(vj) * (E_NMDA - vj)
            + syn.g_gaba_a[j] * (E_GABA_A - vj)
            + g_gaba_b * (E_GABA_B - vj);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mg_block_is_monotonic_in_voltage() {
        // Depolarisation relieves the Mg²⁺ block, so B increases with v.
        assert!(nmda_mg_block(-70.0) < nmda_mg_block(0.0));
        assert!(nmda_mg_block(0.0) < nmda_mg_block(40.0));
        // Bounded in (0, 1).
        assert!(nmda_mg_block(-80.0) > 0.0 && nmda_mg_block(40.0) < 1.0);
    }

    #[test]
    fn dual_exp_norm_gives_unit_peak() {
        // Drive a single input and integrate; the peak of (decay - rise)
        // should reach ~1.0 with the normalisation factor.
        let mut s = SynapseState::zeros(1);
        s.step(&[1.0], &[0.0], 0.5); // one unit of excitatory input
        let mut peak = 0.0f32;
        for _ in 0..2000 {
            s.step(&[0.0], &[0.0], 0.5);
            let g = s.g_nmda_decay[0] - s.g_nmda_rise[0];
            peak = peak.max(g);
        }
        assert!((peak - 1.0).abs() < 0.05, "normalised NMDA peak = {peak}");
    }

    #[test]
    fn ampa_decays_exponentially() {
        let mut s = SynapseState::zeros(1);
        s.g_ampa[0] = 1.0;
        s.step(&[0.0], &[0.0], TAU_AMPA); // one time-constant
        assert!((s.g_ampa[0] - (-1.0f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn excitatory_current_is_inward_at_rest() {
        let mut s = SynapseState::zeros(1);
        s.g_ampa[0] = 0.5;
        let mut i = [0.0f32];
        synaptic_current_into(&s, &[-65.0], &mut i);
        assert!(i[0] > 0.0, "AMPA drives depolarising current at rest");
    }
}
