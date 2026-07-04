//! Reservoir-computing Pong: the **real recurrent culture** as the substrate.
//!
//! Unlike [`crate::pursuit::PursuitAgent`], which drives a feed-forward bank of
//! Izhikevich neurons, this agent uses a full [`bl1_sim::Culture`] — recurrent,
//! distance-wired, conductance-based synapses with short-term plasticity — as a
//! fixed *reservoir*. The ball's height is injected as a Gaussian current bump
//! into a spatial band of the culture; the recurrent dynamics transform it into
//! a high-dimensional spike pattern. We read a place code (mean spike rate per
//! vertical band of neurons) and train **only a linear readout** on it with the
//! same reward-modulated node-perturbation (REINFORCE) rule that works for the
//! feed-forward agent.
//!
//! This is honest reservoir computing: the recurrent weights and STP are never
//! touched — the *culture* is the fixed nonlinear substrate, and learning lives
//! entirely in the readout. It's a harder representation than the feed-forward
//! bank (the culture bursts synchronously, which smears the place code), so it
//! tracks less sharply, but it's the genuine recurrent network playing Pong.
//!
//! The API mirrors [`PursuitAgent`] ([`ReservoirAgent::step`] + observable
//! state) so the same live UI can drive either substrate.

use std::fs;
use std::path::Path;

use anyhow::Result;
use bl1_core::{SimState, simulate};
use bl1_sim::{Config, Culture};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::closed_loop::RunLog;
use crate::pong::{Action, Event, Pong, PongState};
use crate::pursuit::{Brain, PaddleControl};

/// Parameters for the reservoir (recurrent-culture) pursuit agent.
#[derive(Debug, Clone)]
pub struct ReservoirParams {
    /// Number of neurons in the recurrent culture (the reservoir size).
    pub n_neurons: usize,
    /// Number of vertical bands: sensory place-code resolution and readout dim.
    pub n_input: usize,
    /// Sub-steps of the culture simulated per game frame (the neural window).
    pub window_steps: usize,
    /// Sensory current amplitude at the bump centre.
    pub input_amp: f32,
    /// Bump width across bands (Gaussian σ, in band units).
    pub input_sigma: f32,
    pub learning_rate: f32,
    pub reward_alpha: f32,
    pub explore0: f32,
    pub explore_min: f32,
    pub explore_decay_steps: usize,
    pub ball_speed: f32,
    /// How the paddle follows the decoded target (shared with feed-forward).
    pub control: PaddleControl,
    pub paddle_accel: f32,
    pub paddle_damping: f32,
    pub paddle_max_speed: f32,
}

impl Default for ReservoirParams {
    fn default() -> Self {
        Self {
            n_neurons: 400,
            n_input: 16,
            window_steps: 40,
            input_amp: 10.0,
            input_sigma: 1.5,
            learning_rate: 0.05,
            reward_alpha: 0.1,
            explore0: 0.3,
            explore_min: 0.05,
            explore_decay_steps: 3000,
            ball_speed: 0.03,
            control: PaddleControl::Direct,
            paddle_accel: 0.5,
            paddle_damping: 0.6,
            paddle_max_speed: 0.08,
        }
    }
}

/// Culture config for the reservoir, deterministic in `n_neurons`. A fixed
/// geometry keeps a saved brain reproducible from `(seed, n_neurons)` alone.
fn reservoir_config(n_neurons: usize) -> Config {
    let yaml = format!(
        "culture:\n  n_neurons: {n_neurons}\n  substrate_um: [1000, 1000]\n  p_max: 0.2\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n"
    );
    Config::from_yaml_str(&yaml).expect("reservoir config is valid")
}

/// A recurrent-culture reservoir with a trained linear readout.
pub struct ReservoirAgent {
    p: ReservoirParams,
    culture: Culture,
    state: SimState,
    dt: f32,
    /// Band index for each neuron (by vertical position).
    band_of: Vec<usize>,
    /// Neuron count per band, for rate normalisation.
    band_size: Vec<f32>,

    // readout (Gaussian policy mean μ = w·x + b) and per-position baseline
    w: Vec<f32>,
    b: f32,
    baseline: Vec<f32>,

    seed: u64,
    pong: Pong,
    rng: Pcg64,

    // persistent play/learning state
    game: PongState,
    paddle_vy: f32,
    step_idx: usize,
    hits: u32,
    misses: u32,
    events: Vec<(usize, Event)>,
    rally_lengths: Vec<u32>,
    pop_rate: Vec<f32>,

    // observable snapshots for a live UI
    features: Vec<f32>,
    disp: Vec<f32>,
    last_target: f32,
    last_sigma: f32,
    last_reward: f32,

    // scratch buffers
    drive: Vec<f32>,
    sensory: Vec<f32>,
}

impl ReservoirAgent {
    pub fn new(p: ReservoirParams, seed: u64) -> Self {
        let config = reservoir_config(p.n_neurons);
        let culture = Culture::build(&config, seed);
        let state = culture.make_sim_state();
        let n = culture.n_neurons();
        let dt = culture.dt_ms.max(0.01);

        // Bin neurons into vertical bands by their Y position on the substrate.
        let height = config.culture.substrate_um[1].max(1.0);
        let ni = p.n_input;
        let mut band_of = vec![0usize; n];
        let mut band_size = vec![0.0f32; ni];
        for (i, pos) in culture.positions.iter().enumerate() {
            let frac = (pos[1] / height).clamp(0.0, 1.0);
            let band = ((frac * ni as f32) as usize).min(ni - 1);
            band_of[i] = band;
            band_size[band] += 1.0;
        }
        // Guard against empty bands (small cultures) — avoid divide-by-zero.
        for s in band_size.iter_mut() {
            if *s < 1.0 {
                *s = 1.0;
            }
        }

        let mut rng = Pcg64::seed_from_u64(seed.wrapping_add(0x2545F491));
        let pong = Pong {
            ball_speed: p.ball_speed,
            ..Pong::default()
        };
        let game = pong.reset(&mut rng);

        Self {
            w: vec![0.0; ni],
            b: 0.5,
            baseline: vec![0.0; ni],
            features: vec![0.0; ni],
            disp: vec![0.0; ni],
            drive: vec![0.0; n],
            sensory: vec![0.0; n],
            band_of,
            band_size,
            culture,
            state,
            dt,
            seed,
            pong,
            rng,
            game,
            paddle_vy: 0.0,
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
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) -> Event {
        let p = &self.p;
        let ni = p.n_input;
        let n = self.culture.n_neurons();

        // Sensory encoding: Gaussian bump over bands at the ball's Y, applied as
        // a constant current to every neuron in each band across the window.
        let center = self.game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0);
        self.sensory.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            let bnd = self.band_of[i] as f32;
            let d = bnd - center;
            self.sensory[i] =
                p.input_amp * (-(d * d) / (2.0 * p.input_sigma * p.input_sigma)).exp();
        }

        // Run the recurrent culture; feature = mean spike rate per band. The
        // reservoir (weights + STP) is fixed — only the readout below learns.
        self.features.iter_mut().for_each(|v| *v = 0.0);
        let mut total = 0.0f32;
        for _ in 0..p.window_steps {
            for j in 0..n {
                self.drive[j] = self.culture.bg_mean
                    + self.culture.bg_std * gaussian(&mut self.rng)
                    + self.sensory[j];
            }
            let _ = simulate(
                &self.culture.network,
                &mut self.state,
                &self.drive,
                1,
                self.dt,
            );
            let spikes = &self.state.neuron.spikes;
            for j in 0..n {
                let sp = spikes[j];
                self.features[self.band_of[j]] += sp;
                total += sp;
            }
        }
        // Per-band mean rate (divide by band population), then sum-1 normalise so
        // Δw ∝ x stays well-scaled regardless of absolute firing / burst state.
        for b in 0..ni {
            self.features[b] /= self.band_size[b];
        }
        let sum: f32 = self.features.iter().sum();
        if sum > 1e-6 {
            for v in self.features.iter_mut() {
                *v /= sum;
            }
        }
        for i in 0..ni {
            self.disp[i] = 0.85 * self.disp[i] + 0.15 * self.features[i];
        }
        self.pop_rate
            .push(total / n as f32 / (p.window_steps as f32 * self.dt / 1000.0));

        // Gaussian policy: μ = w·x + b, sample the paddle target.
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

        // Actuator: direct snap or inertial smooth pursuit (shared with pursuit).
        let paddle = match p.control {
            PaddleControl::Direct => target,
            PaddleControl::SmoothPursuit => {
                let err = target - self.game.paddle_y;
                self.paddle_vy = (self.paddle_vy + p.paddle_accel * err) * p.paddle_damping;
                self.paddle_vy = self
                    .paddle_vy
                    .clamp(-p.paddle_max_speed, p.paddle_max_speed);
                (self.game.paddle_y + self.paddle_vy).clamp(0.0, 1.0)
            }
        };

        // Dense tracking reward on the actual paddle position, per-position
        // baseline, and the three-factor readout update: Δw = η (R−R̄) · perturb · x.
        let reward = 1.0 - 2.0 * (paddle - self.game.ball_y).abs();
        let bin = (self.game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0)).round() as usize;
        let rpe = reward - self.baseline[bin];
        self.baseline[bin] = (1.0 - p.reward_alpha) * self.baseline[bin] + p.reward_alpha * reward;
        let g = p.learning_rate * rpe * perturb;
        for i in 0..ni {
            self.w[i] += g * self.features[i];
        }
        self.b += g;

        // Drive the paddle and step the game.
        let mut g_state = self.game;
        g_state.paddle_y = paddle;
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

    // --- observable state for a live UI (mirrors PursuitAgent) ---

    pub fn game(&self) -> &PongState {
        &self.game
    }
    /// Smoothed sensory features for display — deliberately the EMA `disp`, not
    /// the raw per-frame `features` used for learning.
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
    pub fn control(&self) -> PaddleControl {
        self.p.control
    }
    pub fn sigma(&self) -> f32 {
        self.last_sigma
    }
    pub fn n_neurons(&self) -> usize {
        self.culture.n_neurons()
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    // --- portable brain (save / load / share) ---

    pub fn brain(&self) -> Brain {
        Brain {
            version: 1,
            mode: match self.p.control {
                PaddleControl::Direct => "reservoir".to_string(),
                PaddleControl::SmoothPursuit => "reservoir-smooth".to_string(),
            },
            n_input: self.p.n_input,
            per_band: 0,
            seed: self.seed,
            w: self.w.clone(),
            b: self.b,
            baseline: self.baseline.clone(),
            step_idx: self.step_idx,
            hits: self.hits,
            misses: self.misses,
            culture_neurons: self.p.n_neurons,
        }
    }

    pub fn from_brain(brain: &Brain) -> Self {
        let control = if brain.mode.contains("smooth") {
            PaddleControl::SmoothPursuit
        } else {
            PaddleControl::Direct
        };
        let params = ReservoirParams {
            n_neurons: if brain.culture_neurons > 0 {
                brain.culture_neurons
            } else {
                ReservoirParams::default().n_neurons
            },
            n_input: brain.n_input,
            control,
            ..ReservoirParams::default()
        };
        let mut a = ReservoirAgent::new(params, brain.seed);
        a.w = brain.w.clone();
        a.b = brain.b;
        a.baseline = brain.baseline.clone();
        a.step_idx = brain.step_idx;
        a.hits = brain.hits;
        a.misses = brain.misses;
        a
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, serde_yaml::to_string(&self.brain())?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let brain: Brain = serde_yaml::from_str(&fs::read_to_string(path)?)?;
        Ok(Self::from_brain(&brain))
    }

    pub fn recent_outcomes(&self, n: usize) -> Vec<bool> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..]
            .iter()
            .map(|(_, e)| *e == Event::Hit)
            .collect()
    }

    pub fn recent_hit_rate(&self, n: usize) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }
        let start = self.events.len().saturating_sub(n);
        let slice = &self.events[start..];
        slice.iter().filter(|(_, e)| *e == Event::Hit).count() as f32 / slice.len() as f32
    }

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
    fn reservoir_runs_and_scores() {
        let p = ReservoirParams {
            n_neurons: 200,
            ..ReservoirParams::default()
        };
        let mut a = ReservoirAgent::new(p, 1);
        let log = a.run(200);
        assert!(log.hits + log.misses > 0);
        assert_eq!(log.population_rate_hz.len(), 200);
        assert_eq!(a.features().len(), 16);
    }

    #[test]
    fn reservoir_learns_to_track() {
        // The recurrent culture is the fixed substrate; only the readout learns.
        // It should clearly beat the ~16% static-paddle baseline and improve.
        // (Kept small — 200 neurons — so the recurrent sim stays test-fast; the
        // default 400-neuron reservoir reaches ~50% over a longer run.)
        let p = ReservoirParams {
            n_neurons: 200,
            ..ReservoirParams::default()
        };
        let mut a = ReservoirAgent::new(p, 1);
        let log = a.run(3000);
        assert!(
            log.hit_rate() > 0.30,
            "expected reservoir tracking > 30%, got {:.1}%",
            log.hit_rate() * 100.0
        );
        assert!(
            log.improvement() > 0.0,
            "expected positive learning trend, got {:+.1} pts",
            log.improvement() * 100.0
        );
    }

    #[test]
    fn reservoir_brain_roundtrips() {
        let p = ReservoirParams {
            n_neurons: 200,
            ..ReservoirParams::default()
        };
        let mut a = ReservoirAgent::new(p, 5);
        a.run(100);
        let saved = a.brain();
        assert_eq!(saved.mode, "reservoir");
        assert_eq!(saved.culture_neurons, 200);
        let b = ReservoirAgent::from_brain(&saved);
        assert_eq!(b.hits(), a.hits());
        assert_eq!(b.n_neurons(), 200);
        assert_eq!(b.brain().w, a.brain().w);
    }
}
