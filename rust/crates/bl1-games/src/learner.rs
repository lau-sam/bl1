//! The generic reward-modulated learner shared by every game and substrate.
//!
//! One rule, node perturbation (REINFORCE), plays every game here. A [`Substrate`]
//! turns the scalar an [`Environment`] exposes into a normalised place-code
//! feature vector `x`; a linear readout maps `x` to an actuator target through a
//! Gaussian policy `target ~ N(w·x + b, σ)`; exploration is sampling the policy,
//! and learning is the three-factor update
//!
//! ```text
//!   Δw_i = η · (R − R̄) · (target − μ) · x_i,   Δb = η · (R − R̄) · (target − μ)
//! ```
//!
//! which ascends expected reward along the sampled perturbation. `R` is the
//! dense tracking reward the environment returns, `R̄` a per-position baseline.
//! The substrate and environment are swappable; this file owns everything they
//! share — the readout, the exploration schedule, the actuator, the event log,
//! and the portable [`Brain`] snapshot — so a new game or substrate never has to
//! re-implement any of it.

use std::fs;
use std::path::Path;

use anyhow::Result;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

use crate::closed_loop::RunLog;
use crate::env::{Environment, EnvView, GameKind};
use crate::pong::{Event, PongEnv};
use crate::substrate::{CultureReservoir, FeedForwardBank, Substrate, SubstrateKind, gaussian};

/// How the decoded target drives the actuator (paddle Y / view bearing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleControl {
    /// The actuator teleports to the decoded target every frame — the culture's
    /// output *is* the position. Easy: no actuator to fight.
    Direct,
    /// The actuator chases the target with inertia (spring pull + damping + a
    /// speed cap): realistic smooth pursuit. It lags and overshoots, so the
    /// culture must *lead* a fast target instead of snapping onto it.
    SmoothPursuit,
}

impl PaddleControl {
    /// Short human label for a live UI.
    pub fn label(self) -> &'static str {
        match self {
            PaddleControl::Direct => "direct",
            PaddleControl::SmoothPursuit => "smooth-pursuit",
        }
    }
}

/// Which substrate to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateSpec {
    /// Independent Izhikevich bank, `per_band` neurons per band.
    FeedForward { per_band: usize },
    /// Recurrent culture reservoir of `n_neurons`.
    Reservoir { n_neurons: usize },
}

impl SubstrateSpec {
    fn kind(self) -> SubstrateKind {
        match self {
            SubstrateSpec::FeedForward { .. } => SubstrateKind::FeedForward,
            SubstrateSpec::Reservoir { .. } => SubstrateKind::Reservoir,
        }
    }
}

/// Which game to play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvSpec {
    Pong,
}

impl EnvSpec {
    fn kind(self) -> GameKind {
        match self {
            EnvSpec::Pong => GameKind::Pong,
        }
    }
}

/// Shared learning + actuator hyperparameters. Defaults are the calibrated values
/// that make the feed-forward Pong agent reach ~50 % (see [`Learner::build`],
/// which adjusts the few substrate-specific ones).
#[derive(Debug, Clone)]
pub struct LearnParams {
    /// Number of sensory bands (place-code resolution = readout dimensionality).
    pub n_input: usize,
    /// Sub-steps of the neural window simulated per game frame.
    pub window_steps: usize,
    /// Feed-forward window integration step (ms); the reservoir uses its own dt.
    pub dt: f32,
    /// Sensory current amplitude at the bump centre.
    pub input_amp: f32,
    /// Bump width across bands (Gaussian σ, in band units).
    pub input_sigma: f32,
    pub learning_rate: f32,
    pub reward_alpha: f32,
    /// Policy exploration std, decaying over training.
    pub explore0: f32,
    pub explore_min: f32,
    pub explore_decay_steps: usize,
    /// Target drift speed (ball speed / enemy bearing speed).
    pub target_speed: f32,
    /// How the actuator follows the decoded target.
    pub control: PaddleControl,
    /// Spring pull toward the target per frame (smooth-pursuit only).
    pub paddle_accel: f32,
    /// Velocity retention per frame — `< 1` bleeds off momentum (friction).
    pub paddle_damping: f32,
    /// Hard cap on actuator speed per frame (smooth-pursuit only).
    pub paddle_max_speed: f32,
}

impl Default for LearnParams {
    fn default() -> Self {
        Self {
            n_input: 16,
            window_steps: 40,
            dt: 0.5,
            input_amp: 12.0,
            input_sigma: 1.5,
            learning_rate: 0.05,
            reward_alpha: 0.1,
            explore0: 0.3,
            explore_min: 0.05,
            explore_decay_steps: 3000,
            target_speed: 0.03,
            control: PaddleControl::Direct,
            paddle_accel: 0.5,
            paddle_damping: 0.6,
            paddle_max_speed: 0.08,
        }
    }
}

/// A portable, shareable snapshot of a trained brain: the learned readout plus
/// the identity (mode = game+substrate+control, seed, shape) so it reconstructs
/// exactly. A tiny YAML file — copy it to hand your trained culture to someone
/// else, who can load it and keep training from where you left off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brain {
    pub version: u32,
    /// `game-substrate[-smooth]` — parsed back into the full configuration.
    pub mode: String,
    pub n_input: usize,
    /// Feed-forward neurons per band (0 for reservoir brains).
    pub per_band: usize,
    pub seed: u64,
    pub w: Vec<f32>,
    pub b: f32,
    pub baseline: Vec<f32>,
    pub step_idx: usize,
    pub hits: u32,
    pub misses: u32,
    /// Reservoir substrate size (0 for feed-forward brains). Defaulted so older
    /// files still load.
    #[serde(default)]
    pub culture_neurons: usize,
}

/// The persisted `mode` tag encoding `(game, substrate, control)`. Pong tags are
/// the historical strings so v0.1.0 brains keep loading.
fn mode_tag(env: EnvSpec, sub: SubstrateKind, control: PaddleControl) -> String {
    let smooth = control == PaddleControl::SmoothPursuit;
    match (env, sub) {
        (EnvSpec::Pong, SubstrateKind::FeedForward) => {
            if smooth { "pursuit-smooth" } else { "pursuit-feedforward" }
        }
        (EnvSpec::Pong, SubstrateKind::Reservoir) => {
            if smooth { "reservoir-smooth" } else { "reservoir" }
        }
    }
    .to_string()
}

/// Parse a `mode` tag back into its three orthogonal choices.
fn parse_mode(tag: &str) -> (EnvSpec, SubstrateKind, PaddleControl) {
    let env = EnvSpec::Pong;
    let sub = if tag.contains("reservoir") {
        SubstrateKind::Reservoir
    } else {
        SubstrateKind::FeedForward
    };
    let control = if tag.contains("smooth") {
        PaddleControl::SmoothPursuit
    } else {
        PaddleControl::Direct
    };
    (env, sub, control)
}

/// The generic node-perturbation learner: a swappable substrate + environment
/// behind a fixed readout + learning rule.
pub struct Learner {
    substrate: Box<dyn Substrate>,
    env: Box<dyn Environment>,
    p: LearnParams,
    env_spec: EnvSpec,
    per_band: usize,
    n_neurons: usize,
    seed: u64,

    // readout (Gaussian policy mean μ = w·x + b) and per-position baseline
    w: Vec<f32>,
    b: f32,
    baseline: Vec<f32>,

    rng: Pcg64,

    // persistent play/learning state
    paddle_vy: f32,
    step_idx: usize,
    hits: u32,
    misses: u32,
    streak: u32,
    events: Vec<(usize, Event)>,
    rally_lengths: Vec<u32>,
    pop_rate: Vec<f32>,

    // observable snapshots for a live UI
    bump: Vec<f32>,
    disp: Vec<f32>,
    last_target: f32,
    last_sigma: f32,
    last_reward: f32,
}

impl Learner {
    /// Build a learner from the three orthogonal choices, applying the calibrated
    /// defaults (only `input_amp` and the substrate size differ by substrate).
    pub fn build(env: EnvSpec, sub: SubstrateSpec, control: PaddleControl, seed: u64) -> Self {
        let mut p = LearnParams {
            control,
            ..LearnParams::default()
        };
        // The reservoir place code is smeared by bursts, so it wants a slightly
        // gentler drive than the crisp feed-forward bank.
        if sub.kind() == SubstrateKind::Reservoir {
            p.input_amp = 10.0;
        }
        Self::with_params(env, sub, p, seed)
    }

    /// Build with explicit hyperparameters (advanced / testing).
    pub fn with_params(env: EnvSpec, sub: SubstrateSpec, p: LearnParams, seed: u64) -> Self {
        let ni = p.n_input;
        // The RNG for policy + environment. The substrate keeps its own identity
        // from the raw seed; this offset stream is shared by the policy sample,
        // the reservoir's background noise, and environment re-spawns.
        let mut rng = Pcg64::seed_from_u64(seed.wrapping_add(0x2545F491));

        let (substrate, per_band, n_neurons): (Box<dyn Substrate>, usize, usize) = match sub {
            SubstrateSpec::FeedForward { per_band } => (
                Box::new(FeedForwardBank::new(ni, per_band, p.window_steps, p.dt)),
                per_band,
                0,
            ),
            SubstrateSpec::Reservoir { n_neurons } => (
                Box::new(CultureReservoir::new(n_neurons, ni, p.window_steps, seed)),
                0,
                n_neurons,
            ),
        };

        let environment: Box<dyn Environment> = match env {
            EnvSpec::Pong => Box::new(PongEnv::new(p.target_speed, &mut rng)),
        };

        Self {
            substrate,
            env: environment,
            env_spec: env,
            per_band,
            n_neurons,
            seed,
            w: vec![0.0; ni],
            b: 0.5, // start with a centred actuator target
            baseline: vec![0.0; ni],
            rng,
            paddle_vy: 0.0,
            step_idx: 0,
            hits: 0,
            misses: 0,
            streak: 0,
            events: Vec::new(),
            rally_lengths: Vec::new(),
            pop_rate: Vec::new(),
            bump: vec![0.0; ni],
            disp: vec![0.0; ni],
            last_target: 0.5,
            last_sigma: p.explore0,
            last_reward: 0.0,
            p,
        }
    }

    /// Advance training by one game step; returns the event produced.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) -> Event {
        let ni = self.p.n_input;

        // Sensory encoding: a Gaussian bump over bands at the sensed position.
        let track = self.env.sensory_position().clamp(0.0, 1.0);
        let center = track * (ni as f32 - 1.0);
        for b in 0..ni {
            let d = b as f32 - center;
            self.bump[b] =
                self.p.input_amp * (-(d * d) / (2.0 * self.p.input_sigma * self.p.input_sigma)).exp();
        }

        // Run the substrate; copy out the (small) normalised feature vector so the
        // borrow ends before we mutate the readout.
        let x: Vec<f32> = self.substrate.encode(&self.bump, &mut self.rng).to_vec();
        self.pop_rate.push(self.substrate.population_rate_hz());
        for i in 0..ni {
            self.disp[i] = 0.85 * self.disp[i] + 0.15 * x[i];
        }

        // Gaussian policy: mean μ = w·x + b, sample the actuator target.
        let mu: f32 = self.b + self.w.iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
        let frac = (self.step_idx as f32 / self.p.explore_decay_steps as f32).min(1.0);
        let sigma = self.p.explore0 + (self.p.explore_min - self.p.explore0) * frac;
        let target = (mu + sigma * gaussian(&mut self.rng)).clamp(0.0, 1.0);
        let perturb = target - mu;

        // Actuator: direct snap, or inertial smooth pursuit relative to where the
        // actuator currently is.
        let pos = match self.p.control {
            PaddleControl::Direct => target,
            PaddleControl::SmoothPursuit => {
                let cur = self.env.actuator_position();
                let err = target - cur;
                self.paddle_vy = (self.paddle_vy + self.p.paddle_accel * err) * self.p.paddle_damping;
                self.paddle_vy = self
                    .paddle_vy
                    .clamp(-self.p.paddle_max_speed, self.p.paddle_max_speed);
                (cur + self.paddle_vy).clamp(0.0, 1.0)
            }
        };

        // Advance the game; it returns the dense reward + the outcome.
        let (reward, event) = self.env.step(pos, &mut self.rng);

        // Per-position baseline + three-factor readout update: Δw = η(R−R̄)·perturb·x.
        let bin = (track * (ni as f32 - 1.0)).round() as usize;
        let rpe = reward - self.baseline[bin];
        self.baseline[bin] = (1.0 - self.p.reward_alpha) * self.baseline[bin] + self.p.reward_alpha * reward;
        let g = self.p.learning_rate * rpe * perturb;
        for i in 0..ni {
            self.w[i] += g * x[i];
        }
        self.b += g;

        match event {
            Event::Hit => {
                self.hits += 1;
                self.streak += 1;
                self.events.push((self.step_idx, event));
            }
            Event::Miss => {
                self.misses += 1;
                self.rally_lengths.push(self.streak);
                self.streak = 0;
                self.events.push((self.step_idx, event));
            }
            Event::None => {}
        }
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

    pub fn view(&self) -> EnvView<'_> {
        self.env.view()
    }
    /// Smoothed sensory features for display (flicker-free EMA, not the raw
    /// per-frame features used for learning).
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
    pub fn control(&self) -> PaddleControl {
        self.p.control
    }
    pub fn substrate_label(&self) -> &'static str {
        self.substrate.kind().label()
    }
    pub fn game_kind(&self) -> GameKind {
        self.env_spec.kind()
    }

    /// Overall hit/kill rate so far.
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

    // --- portable brain (save / load / share) ---

    /// Snapshot the trained brain for saving or sharing.
    pub fn brain(&self) -> Brain {
        Brain {
            version: 1,
            mode: mode_tag(self.env_spec, self.substrate.kind(), self.p.control),
            n_input: self.p.n_input,
            per_band: self.per_band,
            seed: self.seed,
            w: self.w.clone(),
            b: self.b,
            baseline: self.baseline.clone(),
            step_idx: self.step_idx,
            hits: self.hits,
            misses: self.misses,
            culture_neurons: self.n_neurons,
        }
    }

    /// Reconstruct a learner from a saved brain, ready to continue training.
    pub fn from_brain(brain: &Brain) -> Self {
        let (env, sub_kind, control) = parse_mode(&brain.mode);
        let sub = match sub_kind {
            SubstrateKind::FeedForward => SubstrateSpec::FeedForward {
                per_band: if brain.per_band > 0 { brain.per_band } else { 32 },
            },
            SubstrateKind::Reservoir => SubstrateSpec::Reservoir {
                n_neurons: if brain.culture_neurons > 0 {
                    brain.culture_neurons
                } else {
                    400
                },
            },
        };
        let mut p = LearnParams {
            n_input: brain.n_input,
            control,
            ..LearnParams::default()
        };
        if sub_kind == SubstrateKind::Reservoir {
            p.input_amp = 10.0;
        }
        let mut a = Self::with_params(env, sub, p, brain.seed);
        a.w = brain.w.clone();
        a.b = brain.b;
        a.baseline = brain.baseline.clone();
        a.step_idx = brain.step_idx;
        a.hits = brain.hits;
        a.misses = brain.misses;
        a
    }

    /// Write the brain to a YAML file (creating parent dirs).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, serde_yaml::to_string(&self.brain())?)?;
        Ok(())
    }

    /// Load a shared brain file and rebuild the learner.
    pub fn load(path: &Path) -> Result<Self> {
        let brain: Brain = serde_yaml::from_str(&fs::read_to_string(path)?)?;
        Ok(Self::from_brain(&brain))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pong_ff(control: PaddleControl) -> Learner {
        Learner::build(EnvSpec::Pong, SubstrateSpec::FeedForward { per_band: 32 }, control, 1)
    }

    #[test]
    fn pursuit_runs_and_scores() {
        let mut a = pong_ff(PaddleControl::Direct);
        let log = a.run(300);
        assert!(log.hits + log.misses > 0);
        assert_eq!(log.population_rate_hz.len(), 300);
    }

    #[test]
    fn pursuit_learns_to_track() {
        // Feed-forward direct control should clearly beat the ~16% static-paddle
        // baseline and improve over the run.
        let mut a = pong_ff(PaddleControl::Direct);
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
    fn smooth_pursuit_learns_with_inertia() {
        // Inertial paddle is strictly harder than direct control: it settles
        // ~25–33%, still clearly above the ~16% baseline, trending up.
        let mut a = pong_ff(PaddleControl::SmoothPursuit);
        let log = a.run(6000);
        assert!(
            log.hit_rate() > 0.25,
            "expected smooth-pursuit tracking > 25%, got {:.1}%",
            log.hit_rate() * 100.0
        );
        assert!(log.improvement() > 0.0);
    }

    #[test]
    fn reservoir_learns_to_track() {
        // The recurrent culture is the fixed substrate; only the readout learns.
        // Kept small (200 neurons) so the recurrent sim stays test-fast.
        let mut a = Learner::build(
            EnvSpec::Pong,
            SubstrateSpec::Reservoir { n_neurons: 200 },
            PaddleControl::Direct,
            1,
        );
        let log = a.run(3000);
        assert!(
            log.hit_rate() > 0.30,
            "expected reservoir tracking > 30%, got {:.1}%",
            log.hit_rate() * 100.0
        );
        assert!(log.improvement() > 0.0);
    }

    #[test]
    fn single_step_advances_state() {
        let mut a = pong_ff(PaddleControl::Direct);
        a.step();
        assert_eq!(a.step_idx(), 1);
        assert_eq!(a.features().len(), 16);
    }

    #[test]
    fn brain_roundtrip_preserves_learned_state() {
        let mut a = pong_ff(PaddleControl::Direct);
        a.run(500);
        let saved = a.brain();
        assert_eq!(saved.mode, "pursuit-feedforward");
        let b = Learner::from_brain(&saved);
        assert_eq!(b.hits(), a.hits());
        assert_eq!(b.misses(), a.misses());
        assert_eq!(b.step_idx(), a.step_idx());
        assert_eq!(b.brain().w, a.brain().w);
        assert_eq!(b.brain().seed, a.brain().seed);
    }

    #[test]
    fn smooth_brain_roundtrips_control_mode() {
        let mut a = pong_ff(PaddleControl::SmoothPursuit);
        a.run(200);
        assert_eq!(a.brain().mode, "pursuit-smooth");
        let restored = Learner::from_brain(&a.brain());
        assert_eq!(restored.control(), PaddleControl::SmoothPursuit);
    }

    #[test]
    fn reservoir_brain_roundtrips() {
        let mut a = Learner::build(
            EnvSpec::Pong,
            SubstrateSpec::Reservoir { n_neurons: 200 },
            PaddleControl::Direct,
            5,
        );
        a.run(100);
        let saved = a.brain();
        assert_eq!(saved.mode, "reservoir");
        assert_eq!(saved.culture_neurons, 200);
        let b = Learner::from_brain(&saved);
        assert_eq!(b.hits(), a.hits());
        assert_eq!(b.brain().w, a.brain().w);
    }
}
