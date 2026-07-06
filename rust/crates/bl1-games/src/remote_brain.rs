//! A game-agnostic **brain server**: the culture as a controller for an external
//! game (e.g. real DOOM through ViZDoom) over a simple IPC protocol.
//!
//! The 1-D [`crate::Learner`] owns its game loop; a real engine can't be squeezed
//! behind that. [`RemoteBrain`] inverts control instead — the *game* drives, and
//! each frame it hands the culture an observation vector + the reward earned by
//! the previous action, and gets back one continuous value per action head:
//!
//! ```text
//!   game ── obs[n_input], reward ──▶ RemoteBrain ── actions[n_heads] ──▶ game
//! ```
//!
//! Under the hood it is the same recipe as [`crate::Learner`]: the observation
//! drives a [`Substrate`] (feed-forward bank or recurrent-culture reservoir), a
//! per-head linear readout samples a Gaussian policy, and reward-modulated node
//! perturbation trains the readouts online. Reward is one step delayed (it scores
//! the *previous* action), so the brain applies last frame's update before acting
//! on the new observation. Multiple heads (turn / move / shoot) each learn their
//! own readout from the shared culture representation.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

use crate::substrate::{Substrate, gaussian};

/// A portable snapshot of a remote brain's learned readout + the culture identity
/// needed to reconstruct it, so a real-DOOM session can be saved and resumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteBrainState {
    pub version: u32,
    /// `feedforward` or `reservoir` — must match on load.
    pub substrate: String,
    pub n_input: usize,
    pub n_heads: usize,
    /// Feed-forward neurons per band (0 for reservoir).
    #[serde(default)]
    pub per_band: usize,
    /// Reservoir size (0 for feed-forward).
    #[serde(default)]
    pub n_neurons: usize,
    pub seed: u64,
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub baseline: f32,
    pub step_idx: usize,
}

/// Tunables for the remote brain (shared learning + exploration schedule).
///
/// The learning rule (reward-modulated node perturbation on a linear readout)
/// follows Wunderlich et al. 2019, *Front. Neurosci.* 13:260. Which knobs are
/// paper-grounded and which are engineering defaults is called out per field —
/// see also the README "Honesty note on hyperparameters".
#[derive(Debug, Clone)]
pub struct BrainParams {
    /// Observation length = substrate input bands.
    pub n_input: usize,
    /// Number of action heads (independent readouts).
    pub n_heads: usize,
    /// Scales the incoming observation into a sensory current. Engineering scale.
    pub input_amp: f32,
    /// Readout step size (Wunderlich uses β=0.125; our default is lower).
    pub learning_rate: f32,
    /// EWMA rate for the reward baseline (Wunderlich uses γ=0.5; ours is slower).
    pub reward_alpha: f32,
    /// Initial exploration noise σ. NOT paper-derived: Wunderlich and DishBrain
    /// both use *constant* exploration, no annealing.
    pub explore0: f32,
    /// Exploration floor. Engineering default; a floor of 0 lets the policy
    /// freeze (never explores again) — the reservoir's stuck-at-0 failure mode.
    pub explore_min: f32,
    /// Steps over which σ decays from `explore0` to `explore_min`. Engineering
    /// default; the whole annealing schedule has no basis in the cited papers.
    pub explore_decay_steps: usize,
}

impl Default for BrainParams {
    fn default() -> Self {
        Self {
            n_input: 32,
            n_heads: 3,
            input_amp: 12.0,
            learning_rate: 0.05,
            reward_alpha: 0.05,
            explore0: 0.3,
            explore_min: 0.05,
            explore_decay_steps: 5000,
        }
    }
}

/// The culture as a multi-head controller, learning online by node perturbation.
pub struct RemoteBrain {
    p: BrainParams,
    substrate: Box<dyn Substrate>,
    rng: Pcg64,

    /// Readout weights per head (`n_heads × n_input`) and biases.
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    /// Running reward baseline (global EMA — obs is high-dimensional).
    baseline: f32,

    // one-step-delayed-reward memory: the features and perturbations that
    // produced the previous action, to be credited by the next reward.
    prev_x: Option<Vec<f32>>,
    prev_perturb: Vec<f32>,

    step_idx: usize,
    scratch_bump: Vec<f32>,
}

impl RemoteBrain {
    pub fn new(p: BrainParams, substrate: Box<dyn Substrate>, seed: u64) -> Self {
        assert_eq!(
            substrate.n_bands(),
            p.n_input,
            "substrate bands must match observation length"
        );
        let rng = Pcg64::seed_from_u64(seed.wrapping_add(0x2545F491));
        Self {
            w: vec![vec![0.0; p.n_input]; p.n_heads],
            b: vec![0.0; p.n_heads],
            baseline: 0.0,
            prev_x: None,
            prev_perturb: vec![0.0; p.n_heads],
            step_idx: 0,
            scratch_bump: vec![0.0; p.n_input],
            substrate,
            rng,
            p,
        }
    }

    /// Credit the previous action with `reward`, then observe `obs` and return one
    /// sampled action per head. Actions are raw policy samples (roughly centred on
    /// 0.5, clamped to `[0, 1]`); the caller maps them to game buttons.
    #[allow(clippy::needless_range_loop)]
    pub fn act(&mut self, obs: &[f32], reward: f32) -> Vec<f32> {
        let ni = self.p.n_input;

        // 1. Reward-modulated update for the previous action (node perturbation).
        if let Some(px) = self.prev_x.take() {
            let rpe = reward - self.baseline;
            self.baseline =
                (1.0 - self.p.reward_alpha) * self.baseline + self.p.reward_alpha * reward;
            for m in 0..self.p.n_heads {
                let g = self.p.learning_rate * rpe * self.prev_perturb[m];
                for i in 0..ni {
                    self.w[m][i] += g * px[i];
                }
                self.b[m] += g;
            }
        }

        // 2. Encode the observation through the culture (obs → sensory current).
        for i in 0..ni {
            self.scratch_bump[i] = self.p.input_amp * obs.get(i).copied().unwrap_or(0.0);
        }
        let x: Vec<f32> = self.substrate.encode(&self.scratch_bump, &mut self.rng).to_vec();

        // 3. Sample each head's Gaussian policy; remember the perturbation.
        let frac = (self.step_idx as f32 / self.p.explore_decay_steps as f32).min(1.0);
        let sigma = self.p.explore0 + (self.p.explore_min - self.p.explore0) * frac;
        let mut actions = vec![0.0f32; self.p.n_heads];
        for m in 0..self.p.n_heads {
            let mu: f32 = self.b[m] + self.w[m].iter().zip(&x).map(|(w, xi)| w * xi).sum::<f32>();
            let a = mu + sigma * gaussian(&mut self.rng);
            self.prev_perturb[m] = a - mu;
            actions[m] = a.clamp(0.0, 1.0);
        }

        self.prev_x = Some(x);
        self.step_idx += 1;
        actions
    }

    pub fn population_rate_hz(&self) -> f32 {
        self.substrate.population_rate_hz()
    }
    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    /// The learned readout, for saving: `(weights, biases, baseline, step_idx)`.
    pub fn readout(&self) -> (Vec<Vec<f32>>, Vec<f32>, f32, usize) {
        (self.w.clone(), self.b.clone(), self.baseline, self.step_idx)
    }

    /// Load a readout back in, if it matches this brain's shape. Returns `false`
    /// (and changes nothing) on a dimension mismatch.
    pub fn set_readout(
        &mut self,
        w: Vec<Vec<f32>>,
        b: Vec<f32>,
        baseline: f32,
        step_idx: usize,
    ) -> bool {
        if w.len() != self.p.n_heads
            || b.len() != self.p.n_heads
            || w.iter().any(|row| row.len() != self.p.n_input)
        {
            return false;
        }
        self.w = w;
        self.b = b;
        self.baseline = baseline;
        self.step_idx = step_idx;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::FeedForwardBank;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    /// Synthetic single-head tracking task: the observation is a bump at a random
    /// target position; the correct action is that position. Reward (one step
    /// delayed) is dense tracking. The brain should learn to map obs → target.
    #[test]
    fn brain_learns_a_tracking_map() {
        let n = 12;
        let p = BrainParams {
            n_input: n,
            n_heads: 1,
            explore_decay_steps: 3000,
            ..BrainParams::default()
        };
        let substrate = Box::new(FeedForwardBank::new(n, 16, 40, 0.5));
        let mut brain = RemoteBrain::new(p, substrate, 1);
        let mut env_rng = Pcg64::seed_from_u64(7);

        let mut obs = vec![0.0f32; n];
        let set_bump = |obs: &mut [f32], target: f32| {
            let center = target * (n as f32 - 1.0);
            for (i, o) in obs.iter_mut().enumerate() {
                let d = i as f32 - center;
                *o = (-(d * d) / (2.0 * 1.5 * 1.5)).exp();
            }
        };

        let mut prev_action = 0.5f32;
        let mut prev_target = 0.5f32;
        let mut target = env_rng.random::<f32>();
        set_bump(&mut obs, target);

        let mut rewards = Vec::new();
        for _ in 0..6000 {
            let reward = 1.0 - 2.0 * (prev_action - prev_target).abs();
            rewards.push(reward);
            let a = brain.act(&obs, reward);
            prev_action = a[0];
            prev_target = target;
            target = env_rng.random::<f32>();
            set_bump(&mut obs, target);
        }

        let half = rewards.len() / 2;
        let first: f32 = rewards[..half].iter().sum::<f32>() / half as f32;
        let second: f32 = rewards[half..].iter().sum::<f32>() / (rewards.len() - half) as f32;
        assert!(
            second > first,
            "expected the brain to improve tracking reward: first {first:.3} → second {second:.3}"
        );
        assert!(
            second > 0.3,
            "expected decent learned tracking, got mean reward {second:.3}"
        );
    }

    #[test]
    fn readout_roundtrips_and_rejects_mismatch() {
        let mk = || RemoteBrain::new(
            BrainParams { n_input: 8, n_heads: 3, ..BrainParams::default() },
            Box::new(FeedForwardBank::new(8, 8, 40, 0.5)),
            1,
        );
        let mut a = mk();
        let obs = vec![0.3f32; 8];
        for _ in 0..20 {
            a.act(&obs, 0.5);
        }
        let (w, b, baseline, step) = a.readout();
        let mut fresh = mk();
        assert!(fresh.set_readout(w.clone(), b.clone(), baseline, step));
        assert_eq!(fresh.readout().0, w);
        // Wrong head count is rejected, state untouched.
        assert!(!fresh.set_readout(vec![vec![0.0; 8]], b, baseline, step));
    }

    #[test]
    fn heads_are_independent() {
        let n = 8;
        let p = BrainParams {
            n_input: n,
            n_heads: 3,
            ..BrainParams::default()
        };
        let substrate = Box::new(FeedForwardBank::new(n, 8, 40, 0.5));
        let mut brain = RemoteBrain::new(p, substrate, 2);
        let obs = vec![0.5f32; n];
        let a = brain.act(&obs, 0.0);
        assert_eq!(a.len(), 3);
        for v in a {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
