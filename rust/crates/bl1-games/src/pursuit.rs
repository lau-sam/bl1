//! Smooth-pursuit Pong via reward-modulated Hebbian learning (node perturbation
//! / REINFORCE), a more robust alternative to spike-correlation R-STDP.
//!
//! The spiking culture provides the *representation*: a sensory population S,
//! driven by a Gaussian bump at the ball's position, produces a spike-rate
//! feature vector `x`. A linear readout maps `x` to the paddle target through a
//! Gaussian policy `target ~ N(w·x + b, σ)`. Exploration = sampling the policy;
//! learning = a three-factor rule
//!
//! ```text
//!   Δw_i = η · (R − R̄) · (target − μ) · x_i,   Δb = η · (R − R̄) · (target − μ)
//! ```
//!
//! which ascends expected reward along the sampled perturbation. `R` is the
//! dense tracking reward, `R̄` a per-position baseline. Unlike pairwise STDP,
//! this has an unbiased reward gradient, so it converges on the tracking map.
//!
//! Status: this **works** — on the spiking culture it reaches ~50 % hit rate
//! (vs ~16 % for a static paddle) with a consistent upward learning trend
//! across seeds. Two ingredients were essential: (1) population averaging per
//! band, and (2) **sum-1 normalisation** of the feature vector, so the learning
//! signal `Δw ∝ x` stays well-scaled regardless of how sparsely the culture
//! fires. The paddle is placed at the decoded target (direct control); adding
//! realistic smooth-pursuit paddle dynamics is a further step.

use bl1_core::{IzhParams, NeuronState, build_population, izhikevich_step};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::closed_loop::RunLog;
use crate::pong::{Action, Event, Pong};

/// Parameters for the pursuit (REINFORCE) agent.
#[derive(Debug, Clone)]
pub struct PursuitParams {
    /// Number of sensory bands (place-code resolution).
    pub n_input: usize,
    /// Neurons per band — averaging over a population gives a clean rate code
    /// from noisy individual spike trains.
    pub per_band: usize,
    pub window_steps: usize,
    pub dt: f32,
    pub input_amp: f32,
    pub input_sigma: f32,
    pub learning_rate: f32,
    pub reward_alpha: f32,
    /// Policy exploration std (paddle-target units), decaying over training.
    pub explore0: f32,
    pub explore_min: f32,
    pub explore_decay_steps: usize,
    pub ball_speed: f32,
}

impl Default for PursuitParams {
    fn default() -> Self {
        Self {
            n_input: 16,
            per_band: 32,
            window_steps: 40,
            dt: 0.5,
            input_amp: 12.0,
            input_sigma: 1.5,
            learning_rate: 0.05,
            reward_alpha: 0.1,
            explore0: 0.3,
            explore_min: 0.05,
            explore_decay_steps: 3000,
            ball_speed: 0.03,
        }
    }
}

/// Reward-modulated Hebbian pursuit agent.
pub struct PursuitAgent {
    p: PursuitParams,
    s_params: IzhParams,
    s: NeuronState,
    /// Readout weights and bias for the Gaussian policy mean.
    w: Vec<f32>,
    b: f32,
    /// Per-position reward baseline (one bin per input neuron).
    baseline: Vec<f32>,
    pong: Pong,
    rng: Pcg64,
}

impl PursuitAgent {
    pub fn new(p: PursuitParams, seed: u64) -> Self {
        let s_params = build_population(p.n_input * p.per_band, 1.0).params;
        let s = NeuronState::resting(&s_params);
        Self {
            w: vec![0.0; p.n_input],
            b: 0.5, // start with a centred paddle target
            baseline: vec![0.0; p.n_input],
            s,
            s_params,
            pong: Pong {
                ball_speed: p.ball_speed,
                ..Pong::default()
            },
            rng: Pcg64::seed_from_u64(seed.wrapping_add(0x2545F491)),
            p,
        }
    }

    /// Play `n_game_steps` and return the learning log.
    // Hot loops index parallel per-band arrays (currents, features, weights).
    #[allow(clippy::needless_range_loop)]
    pub fn run(&mut self, n_game_steps: usize) -> RunLog {
        let p = self.p.clone();
        let ni = p.n_input;
        let mut game = self.pong.reset(&mut self.rng);

        let mut log = RunLog {
            rally_lengths: Vec::new(),
            events: Vec::new(),
            hits: 0,
            misses: 0,
            population_rate_hz: Vec::with_capacity(n_game_steps),
        };

        let pb = p.per_band;
        let ns = ni * pb;
        let mut s_current = vec![0.0f32; ns];
        let mut x = vec![0.0f32; ni]; // one feature per band

        for step in 0..n_game_steps {
            // Sensory encoding: Gaussian bump at the ball's Y, applied to every
            // neuron of each band.
            let center = game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0);
            for b in 0..ni {
                let d = b as f32 - center;
                let cur = p.input_amp * (-(d * d) / (2.0 * p.input_sigma * p.input_sigma)).exp();
                for k in 0..pb {
                    s_current[b * pb + k] = cur;
                }
            }

            // Run the spiking population; feature = mean spike rate per band
            // (averaging over `per_band` neurons denoises the rate code).
            x.iter_mut().for_each(|v| *v = 0.0);
            let mut total = 0.0f32;
            for _ in 0..p.window_steps {
                izhikevich_step(&mut self.s, &self.s_params, &s_current, p.dt);
                for b in 0..ni {
                    for k in 0..pb {
                        let sp = self.s.spikes[b * pb + k];
                        x[b] += sp;
                        total += sp;
                    }
                }
            }
            // Normalise to a sum-1 population code: this makes the feature scale
            // independent of the absolute firing rate, so the learning signal
            // (Δw ∝ x) is well-scaled regardless of how sparsely the culture
            // fires. Without this, tiny rates give a vanishing gradient.
            let sum: f32 = x.iter().sum();
            if sum > 1e-6 {
                for v in x.iter_mut() {
                    *v /= sum;
                }
            }
            log.population_rate_hz
                .push(total / ns as f32 / (p.window_steps as f32 * p.dt / 1000.0));

            // Gaussian policy: mean μ = w·x + b, sample the paddle target.
            let mu: f32 = self.b + self.w.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
            let frac = (step as f32 / p.explore_decay_steps as f32).min(1.0);
            let sigma = p.explore0 + (p.explore_min - p.explore0) * frac;
            let target = (mu + sigma * gaussian(&mut self.rng)).clamp(0.0, 1.0);
            let perturb = target - mu;

            // Dense tracking reward and per-position baseline.
            let reward = 1.0 - 2.0 * (target - game.ball_y).abs();
            let bin = (game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0)).round() as usize;
            let rpe = reward - self.baseline[bin];
            self.baseline[bin] =
                (1.0 - p.reward_alpha) * self.baseline[bin] + p.reward_alpha * reward;

            // Three-factor update: Δw = η (R−R̄) · perturbation · x.
            let g = p.learning_rate * rpe * perturb;
            for i in 0..ni {
                self.w[i] += g * x[i];
            }
            self.b += g;

            // Advance the game: place the paddle directly at the decoded target
            // (isolates the readout from paddle-lag dynamics), then step.
            let mut g = game;
            g.paddle_y = target;
            let (next, event) = self.pong.step(&g, Action::Stay, &mut self.rng);
            match event {
                Event::Hit => {
                    log.hits += 1;
                    log.events.push((step, event));
                }
                Event::Miss => {
                    log.misses += 1;
                    log.rally_lengths.push(game.rally_length);
                    log.events.push((step, event));
                }
                Event::None => {}
            }
            game = next;
        }

        log
    }
}

/// Standard normal via Box-Muller.
fn gaussian<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pursuit_runs_and_scores() {
        let mut a = PursuitAgent::new(PursuitParams::default(), 1);
        let log = a.run(300);
        assert!(log.hits + log.misses > 0);
        assert_eq!(log.population_rate_hz.len(), 300);
    }

    #[test]
    fn pursuit_learns_to_track() {
        // The agent should clearly beat the ~16% static-paddle baseline and
        // improve over the run (second half better than the first).
        let mut a = PursuitAgent::new(PursuitParams::default(), 1);
        let log = a.run(6000);
        assert!(
            log.hit_rate() > 0.40,
            "expected learned tracking > 40%, got {:.1}%",
            log.hit_rate() * 100.0
        );
        assert!(
            log.improvement() > 0.0,
            "expected positive learning trend, got {:+.1} pts",
            log.improvement() * 100.0
        );
    }
}
