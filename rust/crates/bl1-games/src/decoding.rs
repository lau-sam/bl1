//! Motor decoding: spikes in two neuron regions → a paddle action.
//!
//! Rate coding over an "up" region and a "down" region. Spike counts are
//! accumulated over the decode window, converted to a mean per-neuron firing
//! rate, and compared against a baseline (Kagan 2022 two-region readout).

use crate::pong::Action;

/// Maps region firing rates to a paddle action.
pub struct MotorDecoder {
    /// Neuron indices whose activity votes "up".
    pub up_neurons: Vec<usize>,
    /// Neuron indices whose activity votes "down".
    pub down_neurons: Vec<usize>,
    /// Minimum mean rate (Hz) for a region to drive the paddle.
    pub baseline_rate_hz: f32,
    /// The winning region must lead the other by this margin (Hz) to move —
    /// prevents a near-tie from biasing the paddle in one direction.
    pub margin_hz: f32,
}

impl MotorDecoder {
    pub fn new(up_neurons: Vec<usize>, down_neurons: Vec<usize>) -> Self {
        Self {
            up_neurons,
            down_neurons,
            baseline_rate_hz: 20.0,
            margin_hz: 2.0,
        }
    }

    /// Decode an action from `window_counts` (spikes per neuron accumulated over
    /// the decode window of `window_ms`). Returns the action and the two region
    /// rates (Hz) for logging. The paddle only moves when one region is both
    /// above baseline and leads the other by `margin_hz` (differential readout).
    pub fn decode(&self, window_counts: &[f32], window_ms: f32) -> (Action, f32, f32) {
        let window_s = (window_ms / 1000.0).max(1e-6);
        let up_rate = region_rate(window_counts, &self.up_neurons, window_s);
        let down_rate = region_rate(window_counts, &self.down_neurons, window_s);

        let action = if up_rate > self.baseline_rate_hz && up_rate - down_rate > self.margin_hz {
            Action::Up
        } else if down_rate > self.baseline_rate_hz && down_rate - up_rate > self.margin_hz {
            Action::Down
        } else {
            Action::Stay
        };
        (action, up_rate, down_rate)
    }
}

/// Mean per-neuron firing rate (Hz) of a region over the window.
fn region_rate(window_counts: &[f32], neurons: &[usize], window_s: f32) -> f32 {
    if neurons.is_empty() {
        return 0.0;
    }
    let total: f32 = neurons.iter().map(|&i| window_counts[i]).sum();
    total / neurons.len() as f32 / window_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stronger_up_region_moves_up() {
        let dec = MotorDecoder::new(vec![0, 1], vec![2, 3]);
        // Over a 100 ms window, up neurons fire a lot, down neurons don't.
        let counts = vec![10.0, 10.0, 0.0, 0.0];
        let (action, up, down) = dec.decode(&counts, 100.0);
        assert_eq!(action, Action::Up);
        assert!(up > down);
    }

    #[test]
    fn quiet_regions_stay() {
        let dec = MotorDecoder::new(vec![0, 1], vec![2, 3]);
        let counts = vec![0.0, 0.0, 0.0, 0.0];
        let (action, _, _) = dec.decode(&counts, 100.0);
        assert_eq!(action, Action::Stay);
    }
}
