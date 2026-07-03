//! Spiking neuron models: Izhikevich (2003) and Adaptive Exponential IF
//! (Brette & Gerstner 2005).
//!
//! Both use forward (semi-implicit) Euler with a default step of `dt = 0.5 ms`,
//! matching the reference Python implementation. The recovery/adaptation
//! variable is updated from the *previous* membrane potential.

/// Spike threshold (mV) at which an Izhikevich neuron fires and resets.
pub const IZH_V_PEAK: f32 = 30.0;
/// Resting potential (mV) used to initialise Izhikevich state.
pub const IZH_V_REST: f32 = -65.0;

/// Per-neuron Izhikevich parameters.
#[derive(Debug, Clone)]
pub struct IzhParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
}

/// Izhikevich neuron state: membrane potential `v`, recovery `u`, and last
/// spike indicator `spikes` (0.0 / 1.0).
#[derive(Debug, Clone)]
pub struct NeuronState {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub spikes: Vec<f32>,
}

impl NeuronState {
    /// Initialise `n` neurons at rest: `v = -65`, `u = b * v`, `spikes = 0`.
    pub fn resting(params: &IzhParams) -> Self {
        let n = params.a.len();
        let v = vec![IZH_V_REST; n];
        let u: Vec<f32> = params.b.iter().map(|&b| b * IZH_V_REST).collect();
        Self {
            v,
            u,
            spikes: vec![0.0; n],
        }
    }

    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.v.len()
    }

    /// Whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

/// Advance the Izhikevich population by one step of `dt` ms under input
/// current `i_ext` (per neuron). Updates `state` in place.
///
/// ```text
/// v' = v + dt (0.04 v^2 + 5 v + 140 - u + I)
/// u' = u + dt a (b v - u)          // uses the old v
/// if v' >= 30:  spike; v' = c; u' += d
/// ```
pub fn izhikevich_step(state: &mut NeuronState, p: &IzhParams, i_ext: &[f32], dt: f32) {
    debug_assert_eq!(i_ext.len(), state.len());
    for (j, &ie) in i_ext.iter().enumerate() {
        let v = state.v[j];
        let u = state.u[j];
        let mut v_new = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u + ie);
        let mut u_new = u + dt * p.a[j] * (p.b[j] * v - u);
        if v_new >= IZH_V_PEAK {
            state.spikes[j] = 1.0;
            v_new = p.c[j];
            u_new += p.d[j];
        } else {
            state.spikes[j] = 0.0;
        }
        state.v[j] = v_new;
        state.u[j] = u_new;
    }
}

/// Per-neuron AdEx parameters (Brette & Gerstner 2005). Units: `c` in pF,
/// `g_l` in nS, potentials in mV, `a`/currents in pA/nS, times in ms.
#[derive(Debug, Clone)]
pub struct AdExParams {
    pub c: Vec<f32>,
    pub g_l: Vec<f32>,
    pub e_l: Vec<f32>,
    pub delta_t: Vec<f32>,
    pub v_t: Vec<f32>,
    pub v_reset: Vec<f32>,
    pub v_peak: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub tau_w: Vec<f32>,
}

/// AdEx state: membrane potential `v`, adaptation current `w`, spike flag.
#[derive(Debug, Clone)]
pub struct AdExState {
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub spikes: Vec<f32>,
}

impl AdExState {
    /// Initialise at `v = E_L`, `w = 0`.
    pub fn resting(params: &AdExParams) -> Self {
        Self {
            v: params.e_l.clone(),
            w: vec![0.0; params.e_l.len()],
            spikes: vec![0.0; params.e_l.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

/// Advance the AdEx population by one step of `dt` ms.
///
/// ```text
/// dv = (-g_L (v - E_L) + g_L dT exp((v - V_T)/dT) - w + I) / C
/// dw = (a (v - E_L) - w) / tau_w
/// if v' >= V_peak: spike; v' = V_reset; w' += b
/// ```
/// The exponential argument is clamped to `[-20, 20]` to prevent overflow.
pub fn adex_step(state: &mut AdExState, p: &AdExParams, i_ext: &[f32], dt: f32) {
    debug_assert_eq!(i_ext.len(), state.len());
    for (j, &ie) in i_ext.iter().enumerate() {
        let v = state.v[j];
        let w = state.w[j];
        let exp_arg = ((v - p.v_t[j]) / p.delta_t[j]).clamp(-20.0, 20.0);
        let exp_term = p.g_l[j] * p.delta_t[j] * exp_arg.exp();
        let dv = (-p.g_l[j] * (v - p.e_l[j]) + exp_term - w + ie) / p.c[j];
        let mut v_new = v + dt * dv;
        let dw = (p.a[j] * (v - p.e_l[j]) - w) / p.tau_w[j];
        let mut w_new = w + dt * dw;
        if v_new >= p.v_peak[j] {
            state.spikes[j] = 1.0;
            v_new = p.v_reset[j];
            w_new += p.b[j];
        } else {
            state.spikes[j] = 0.0;
        }
        state.v[j] = v_new;
        state.w[j] = w_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs_params(n: usize) -> IzhParams {
        IzhParams {
            a: vec![0.02; n],
            b: vec![0.2; n],
            c: vec![-65.0; n],
            d: vec![8.0; n],
        }
    }

    #[test]
    fn resting_state_is_stable_without_input() {
        let p = rs_params(1);
        let mut s = NeuronState::resting(&p);
        for _ in 0..200 {
            izhikevich_step(&mut s, &p, &[0.0], 0.5);
        }
        // With no drive the RS neuron settles at its stable subthreshold fixed
        // point (v ≈ -70 mV, the lower root of 0.04v²+5v+140-0.2v=0) and never
        // spikes.
        assert_eq!(s.spikes[0], 0.0);
        assert!(s.v[0] > -80.0 && s.v[0] < -60.0, "v drifted to {}", s.v[0]);
    }

    #[test]
    fn strong_input_makes_rs_neuron_fire() {
        let p = rs_params(1);
        let mut s = NeuronState::resting(&p);
        let mut fired = false;
        for _ in 0..200 {
            izhikevich_step(&mut s, &p, &[10.0], 0.5);
            if s.spikes[0] == 1.0 {
                fired = true;
                assert_eq!(s.v[0], -65.0, "reset to c after spike");
                break;
            }
        }
        assert!(fired, "RS neuron should fire under 10 pA drive");
    }

    #[test]
    fn adex_fires_under_drive() {
        let p = AdExParams {
            c: vec![281.0],
            g_l: vec![30.0],
            e_l: vec![-70.6],
            delta_t: vec![2.0],
            v_t: vec![-50.4],
            v_reset: vec![-70.6],
            v_peak: vec![20.0],
            a: vec![4.0],
            b: vec![80.5],
            tau_w: vec![144.0],
        };
        let mut s = AdExState::resting(&p);
        let mut fired = false;
        for _ in 0..2000 {
            adex_step(&mut s, &p, &[600.0], 0.5);
            if s.spikes[0] == 1.0 {
                fired = true;
                break;
            }
        }
        assert!(fired, "AdEx neuron should fire under 600 pA drive");
    }
}
