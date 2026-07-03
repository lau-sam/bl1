//! Feedback stimulation under the free-energy-principle protocol (Kagan 2022).
//!
//! A **hit** delivers *predictable* stimulation (a uniform, low-entropy pulse
//! across the culture); a **miss** delivers *unpredictable* stimulation (a
//! high-entropy random pulse). The culture is hypothesised to reorganise to
//! minimise unpredictable input — i.e. to avoid missing.

use crate::pong::Event;
use rand::Rng;

/// Feedback amplitudes for the FEP protocol.
pub struct FeedbackProtocol {
    /// Amplitude of the predictable (hit) pulse, applied to every neuron.
    pub predictable_amp: f32,
    /// Inclusive range of the unpredictable (miss) pulse per neuron.
    pub unpredictable_min: f32,
    pub unpredictable_max: f32,
}

impl Default for FeedbackProtocol {
    fn default() -> Self {
        Self {
            predictable_amp: 2.0,
            unpredictable_min: -5.0,
            unpredictable_max: 15.0,
        }
    }
}

impl FeedbackProtocol {
    /// Feedback current (one value per neuron) for `event`.
    /// `Event::None` produces no stimulation.
    pub fn current<R: Rng>(&self, event: Event, n_neurons: usize, rng: &mut R) -> Vec<f32> {
        match event {
            Event::Hit => vec![self.predictable_amp; n_neurons],
            Event::Miss => (0..n_neurons)
                .map(|_| rng.random_range(self.unpredictable_min..self.unpredictable_max))
                .collect(),
            Event::None => vec![0.0; n_neurons],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn hit_is_uniform_low_entropy() {
        let fb = FeedbackProtocol::default();
        let mut rng = Pcg64::seed_from_u64(1);
        let c = fb.current(Event::Hit, 16, &mut rng);
        assert!(c.iter().all(|&x| x == 2.0), "predictable pulse is uniform");
    }

    #[test]
    fn miss_is_variable_high_entropy() {
        let fb = FeedbackProtocol::default();
        let mut rng = Pcg64::seed_from_u64(1);
        let c = fb.current(Event::Miss, 64, &mut rng);
        let mean = c.iter().sum::<f32>() / c.len() as f32;
        let var = c.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / c.len() as f32;
        assert!(var > 1.0, "unpredictable pulse should vary, var={var}");
    }

    #[test]
    fn none_is_silent() {
        let fb = FeedbackProtocol::default();
        let mut rng = Pcg64::seed_from_u64(1);
        assert!(
            fb.current(Event::None, 8, &mut rng)
                .iter()
                .all(|&x| x == 0.0)
        );
    }
}
