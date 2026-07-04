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
//!
//! The agent exposes a single-step API ([`PursuitAgent::step`]) plus observable
//! state so a live UI can advance training frame by frame and watch it learn.

use bl1_core::{IzhParams, NeuronState, build_population, izhikevich_step};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::closed_loop::RunLog;
use crate::pong::{Action, Event, Pong, PongState};

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

/// Reward-modulated Hebbian pursuit agent with an incremental training step.
pub struct PursuitAgent {
    p: PursuitParams,
    s_params: IzhParams,
    s: NeuronState,
    /// Readout weights and bias for the Gaussian policy mean.
    w: Vec<f32>,
    b: f32,
    /// Per-position reward baseline (one bin per band).
    baseline: Vec<f32>,
    pong: Pong,
    rng: Pcg64,

    // --- persistent play/learning state (advanced one step at a time) ---
    game: PongState,
    step_idx: usize,
    hits: u32,
    misses: u32,
    events: Vec<(usize, Event)>,
    rally_lengths: Vec<u32>,
    pop_rate: Vec<f32>,

    // --- observable snapshots for a live UI ---
    features: Vec<f32>,
    /// EMA-smoothed features for display (raw per-frame rates flicker badly).
    disp: Vec<f32>,
    last_target: f32,
    last_sigma: f32,
    last_reward: f32,

    // scratch buffers
    s_current: Vec<f32>,
}

impl PursuitAgent {
    pub fn new(p: PursuitParams, seed: u64) -> Self {
        let s_params = build_population(p.n_input * p.per_band, 1.0).params;
        let s = NeuronState::resting(&s_params);
        let mut rng = Pcg64::seed_from_u64(seed.wrapping_add(0x2545F491));
        let pong = Pong {
            ball_speed: p.ball_speed,
            ..Pong::default()
        };
        let game = pong.reset(&mut rng);
        Self {
            w: vec![0.0; p.n_input],
            b: 0.5, // start with a centred paddle target
            baseline: vec![0.0; p.n_input],
            features: vec![0.0; p.n_input],
            disp: vec![0.0; p.n_input],
            s_current: vec![0.0; p.n_input * p.per_band],
            s,
            s_params,
            pong,
            rng,
            game,
            step_idx: 0,
            hits: 0,
            misses: 0,
            events: Vec::new(),
            rally_lengths: Vec::new(),
            pop_rate: Vec::new(),
            last_target: 0.5,
            last_sigma: p.explore0,
            last_reward: 0.0,
            p,
        }
    }

    /// Advance training by one game step; returns the event produced.
    // Hot loops index parallel per-band arrays (currents, features, weights).
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) -> Event {
        let p = &self.p;
        let ni = p.n_input;
        let pb = p.per_band;

        // Sensory encoding: Gaussian bump at the ball's Y, applied to every
        // neuron of each band.
        let center = self.game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0);
        for b in 0..ni {
            let d = b as f32 - center;
            let cur = p.input_amp * (-(d * d) / (2.0 * p.input_sigma * p.input_sigma)).exp();
            for k in 0..pb {
                self.s_current[b * pb + k] = cur;
            }
        }

        // Run the spiking population; feature = mean spike rate per band.
        self.features.iter_mut().for_each(|v| *v = 0.0);
        let mut total = 0.0f32;
        for _ in 0..p.window_steps {
            izhikevich_step(&mut self.s, &self.s_params, &self.s_current, p.dt);
            for b in 0..ni {
                for k in 0..pb {
                    let sp = self.s.spikes[b * pb + k];
                    self.features[b] += sp;
                    total += sp;
                }
            }
        }
        // Sum-1 normalisation: keeps the feature scale (and hence Δw ∝ x)
        // independent of the absolute firing rate.
        let sum: f32 = self.features.iter().sum();
        if sum > 1e-6 {
            for v in self.features.iter_mut() {
                *v /= sum;
            }
        }
        // Smooth the display features so the bump doesn't flicker frame-to-frame.
        for i in 0..ni {
            self.disp[i] = 0.85 * self.disp[i] + 0.15 * self.features[i];
        }
        let ns = (ni * pb) as f32;
        self.pop_rate
            .push(total / ns / (p.window_steps as f32 * p.dt / 1000.0));

        // Gaussian policy: mean μ = w·x + b, sample the paddle target.
        let mu: f32 = self.b
            + self
                .w
                .iter()
                .zip(&self.features)
                .map(|(w, xi)| w * xi)
                .sum::<f32>();
        let frac = (self.step_idx as f32 / p.explore_decay_steps as f32).min(1.0);
        let sigma = p.explore0 + (p.explore_min - p.explore0) * frac;
        let target = (mu + sigma * gaussian(&mut self.rng)).clamp(0.0, 1.0);
        let perturb = target - mu;

        // Dense tracking reward and per-position baseline.
        let reward = 1.0 - 2.0 * (target - self.game.ball_y).abs();
        let bin = (self.game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0)).round() as usize;
        let rpe = reward - self.baseline[bin];
        self.baseline[bin] = (1.0 - p.reward_alpha) * self.baseline[bin] + p.reward_alpha * reward;

        // Three-factor update: Δw = η (R−R̄) · perturbation · x.
        let g = p.learning_rate * rpe * perturb;
        for i in 0..ni {
            self.w[i] += g * self.features[i];
        }
        self.b += g;

        // Place the paddle at the decoded target (direct control), then step.
        let mut g_state = self.game;
        g_state.paddle_y = target;
        let (next, event) = self.pong.step(&g_state, Action::Stay, &mut self.rng);
        match event {
            Event::Hit => {
                self.hits += 1;
                self.events.push((self.step_idx, event));
            }
            Event::Miss => {
                self.misses += 1;
                self.rally_lengths.push(self.game.rally_length);
                self.events.push((self.step_idx, event));
            }
            Event::None => {}
        }
        self.game = next;
        self.last_target = target;
        self.last_sigma = sigma;
        self.last_reward = reward;
        self.step_idx += 1;
        event
    }

    /// Play `n_game_steps` and return the accumulated learning log.
    pub fn run(&mut self, n_game_steps: usize) -> RunLog {
        for _ in 0..n_game_steps {
            self.step();
        }
        RunLog {
            rally_lengths: self.rally_lengths.clone(),
            events: self.events.clone(),
            hits: self.hits,
            misses: self.misses,
            population_rate_hz: self.pop_rate.clone(),
        }
    }

    // --- observable state for a live UI ---

    pub fn game(&self) -> &PongState {
        &self.game
    }
    /// Smoothed sensory features, for display (flicker-free) — deliberately the
    /// EMA `disp`, not the raw per-frame `features` used for learning.
    #[allow(clippy::misnamed_getters)]
    pub fn features(&self) -> &[f32] {
        &self.disp
    }
    pub fn step_idx(&self) -> usize {
        self.step_idx
    }
    pub fn hits(&self) -> u32 {
        self.hits
    }
    pub fn misses(&self) -> u32 {
        self.misses
    }
    pub fn last_target(&self) -> f32 {
        self.last_target
    }
    pub fn sigma(&self) -> f32 {
        self.last_sigma
    }

    /// Overall hit rate so far.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// The most recent `n` outcomes in time order (oldest→newest); `true` = hit.
    pub fn recent_outcomes(&self, n: usize) -> Vec<bool> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..]
            .iter()
            .map(|(_, e)| *e == Event::Hit)
            .collect()
    }

    /// Hit rate over the most recent `n` events (recent skill).
    pub fn recent_hit_rate(&self, n: usize) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }
        let start = self.events.len().saturating_sub(n);
        let slice = &self.events[start..];
        slice.iter().filter(|(_, e)| *e == Event::Hit).count() as f32 / slice.len() as f32
    }

    /// Hit rate per consecutive block of `block` events — the learning curve.
    pub fn hit_rate_curve(&self, block: usize) -> Vec<f32> {
        if block == 0 {
            return Vec::new();
        }
        self.events
            .chunks(block)
            .map(|c| c.iter().filter(|(_, e)| *e == Event::Hit).count() as f32 / c.len() as f32)
            .collect()
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

    #[test]
    fn single_step_advances_state() {
        let mut a = PursuitAgent::new(PursuitParams::default(), 3);
        a.step();
        assert_eq!(a.step_idx(), 1);
        assert_eq!(a.features().len(), 16);
    }
}
