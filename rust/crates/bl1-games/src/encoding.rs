//! Sensory encoding: ball position → stimulation of sensory electrodes.
//!
//! Mixed place + rate code (Kagan 2022):
//! - **place**: the ball's vertical position selects one of `n_channels`
//!   sensory electrodes (with a neighbour near band boundaries);
//! - **rate**: the ball's horizontal proximity to the paddle sets the pulse
//!   frequency, from `freq_min` (far) to `freq_max` (close).
//!
//! A phase accumulator gates pulses so stimulation fires at the coded rate.

/// Stimulation emitted for one game step.
pub struct Stimulus {
    /// Sensory electrodes to stimulate this step (empty when no pulse fires).
    pub electrodes: Vec<usize>,
    /// Stimulation amplitude (mV-scale current), 0 when no pulse fires.
    pub amplitude: f32,
}

/// Ball → sensory-electrode encoder with per-encoder pulse phase.
pub struct SensoryEncoder {
    /// Electrode index for each of the `n_channels` vertical bands.
    pub channels: Vec<usize>,
    pub freq_min_hz: f32,
    pub freq_max_hz: f32,
    pub amplitude: f32,
    phase: f32,
}

impl SensoryEncoder {
    pub fn new(channels: Vec<usize>) -> Self {
        Self {
            channels,
            freq_min_hz: 4.0,
            freq_max_hz: 40.0,
            // Comparable to the tonic drive scale (bg ~ N(1, 3)); the Izhikevich
            // neuron fires around I ≈ 5–15, so a pulse of ~10 recruits the
            // targeted band without saturating the whole culture.
            amplitude: 10.0,
            phase: 0.0,
        }
    }

    /// Advance the pulse phase by `dt_game_s` and return the stimulus for this
    /// game step: which sensory electrodes fire (place code on `ball_y`) at what
    /// amplitude, gated by a frequency set from `ball_x` proximity (rate code).
    pub fn encode(&mut self, ball_x: f32, ball_y: f32, dt_game_s: f32) -> Stimulus {
        let n = self.channels.len().max(1);
        let freq =
            self.freq_min_hz + (self.freq_max_hz - self.freq_min_hz) * ball_x.clamp(0.0, 1.0);

        // Advance phase; a pulse fires when it crosses an integer boundary.
        self.phase += freq * dt_game_s;
        let fire = self.phase >= 1.0;
        if fire {
            self.phase -= self.phase.floor();
        }
        if !fire {
            return Stimulus {
                electrodes: Vec::new(),
                amplitude: 0.0,
            };
        }

        // Place code: pick the band and, near a boundary, its neighbour.
        let y = ball_y.clamp(0.0, 1.0);
        let band_f = (y * n as f32).min(n as f32 - 1.0);
        let band = band_f.floor() as usize;
        let mut electrodes = vec![self.channels[band]];
        let frac = band_f - band as f32;
        if frac < 0.25 && band > 0 {
            electrodes.push(self.channels[band - 1]);
        } else if frac > 0.75 && band + 1 < n {
            electrodes.push(self.channels[band + 1]);
        }

        Stimulus {
            electrodes,
            amplitude: self.amplitude,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_frequency_fires_more_often() {
        let mut enc = SensoryEncoder::new((0..8).collect());
        let dt = 0.02; // 20 ms game step
        // Ball far (x=0) -> low freq; ball near (x=1) -> high freq.
        let count_far = (0..100)
            .filter(|_| !enc.encode(0.0, 0.5, dt).electrodes.is_empty())
            .count();
        enc.phase = 0.0;
        let count_near = (0..100)
            .filter(|_| !enc.encode(1.0, 0.5, dt).electrodes.is_empty())
            .count();
        assert!(count_near > count_far, "{count_near} !> {count_far}");
    }

    #[test]
    fn place_code_maps_y_to_band() {
        let mut enc = SensoryEncoder::new((0..8).collect());
        enc.freq_min_hz = 1000.0; // force a pulse every step
        enc.freq_max_hz = 1000.0;
        let top = enc.encode(0.5, 0.99, 0.02);
        assert!(top.electrodes.contains(&7), "top band -> electrode 7");
        enc.phase = 0.0;
        let bottom = enc.encode(0.5, 0.01, 0.02);
        assert!(bottom.electrodes.contains(&0), "bottom band -> electrode 0");
    }
}
