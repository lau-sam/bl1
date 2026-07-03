//! A reward-modulated STDP agent that learns Pong by smooth pursuit.
//!
//! This is the *learning* architecture, following the proven neuromorphic
//! recipe (Wunderlich et al. 2019, BrainScaleS-2) rather than relying on a
//! recurrent culture's dynamics for credit assignment:
//!
//! - a **sensory** population S place-codes the ball's vertical position;
//! - a **motor** population M codes the paddle target position;
//! - a dedicated **plastic feed-forward projection** `W_sm` (M×S) is the only
//!   thing that learns — R-STDP shapes it directly, so credit assignment is
//!   local rather than routed through recurrent dynamics;
//! - a **dense, graded reward** every step (how well the decoded paddle tracks
//!   the ball), with a **per-position baseline** `R̄` so the update follows the
//!   reward-prediction error `(R − R̄)`;
//! - **exploration** via decaying motor noise and **homeostasis** via row-sum
//!   normalisation, both intended to help R-STDP converge.
//!
//! Neurons are the same biophysical Izhikevich units as the rest of BL-1.
//!
//! Status: this is the correct, literature-grounded architecture, but it does
//! **not** yet converge to Pong play in practice — the readout stays near the
//! centre and the hit rate hovers around chance. Making R-STDP actually learn
//! here (careful causal-correlation measurement, weight init, long training,
//! hyperparameter search) is an open research problem, kept as scaffolding.

use bl1_core::{IzhParams, NeuronState, build_population, izhikevich_step};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::closed_loop::RunLog;
use crate::pong::{Action, Event, Pong};

/// Tunable parameters for the R-STDP agent.
#[derive(Debug, Clone)]
pub struct AgentParams {
    pub n_input: usize,
    pub n_output: usize,
    /// Neural sub-steps simulated per game step.
    pub window_steps: usize,
    pub dt: f32,
    /// Peak current of the sensory Gaussian bump, and its width (in neuron units).
    pub input_amp: f32,
    pub input_sigma: f32,
    /// Initial mean feed-forward weight (per S→M synapse).
    pub w_init: f32,
    pub w_min: f32,
    pub w_max: f32,
    /// R-STDP learning rate and trace/eligibility time constants (ms).
    pub learning_rate: f32,
    pub tau_trace: f32,
    pub tau_elig: f32,
    /// Reward-baseline EMA rate.
    pub reward_alpha: f32,
    /// Exploration motor-noise current: decays linearly from `..0` to `..min`.
    pub explore_std0: f32,
    pub explore_std_min: f32,
    pub explore_decay_steps: usize,
    pub ball_speed: f32,
}

impl Default for AgentParams {
    fn default() -> Self {
        Self {
            n_input: 16,
            n_output: 16,
            window_steps: 20,
            dt: 0.5,
            input_amp: 12.0,
            input_sigma: 1.5,
            w_init: 2.0,
            w_min: 0.0,
            w_max: 20.0,
            learning_rate: 0.05,
            tau_trace: 20.0,
            // Short eligibility: the reward is dense and immediate, so credit
            // must stay local to the current ball position. A long trace (for
            // distal/sparse reward) would smear credit across many positions.
            tau_elig: 30.0,
            reward_alpha: 0.5,
            explore_std0: 8.0,
            explore_std_min: 1.0,
            explore_decay_steps: 2000,
            ball_speed: 0.03,
        }
    }
}

/// The R-STDP Pong agent: two Izhikevich populations plus a plastic S→M matrix.
pub struct RstdpAgent {
    p: AgentParams,
    s_params: IzhParams,
    m_params: IzhParams,
    s: NeuronState,
    m: NeuronState,
    /// Feed-forward weights, row-major `[out * n_input + in]`.
    w_sm: Vec<f32>,
    /// Target row sum for homeostatic normalisation.
    row_target: f32,
    s_trace: Vec<f32>,
    m_trace: Vec<f32>,
    elig: Vec<f32>,
    /// Reward baseline per ball-position bin (one bin per input neuron).
    baseline: Vec<f32>,
    pong: Pong,
    rng: Pcg64,
}

impl RstdpAgent {
    pub fn new(p: AgentParams, seed: u64) -> Self {
        let s_params = build_population(p.n_input, 1.0).params;
        let m_params = build_population(p.n_output, 1.0).params;
        let s = NeuronState::resting(&s_params);
        let m = NeuronState::resting(&m_params);
        let w_sm = vec![p.w_init; p.n_output * p.n_input];
        let row_target = p.w_init * p.n_input as f32;
        let pong = Pong {
            ball_speed: p.ball_speed,
            ..Pong::default()
        };
        Self {
            s_trace: vec![0.0; p.n_input],
            m_trace: vec![0.0; p.n_output],
            elig: vec![0.0; p.n_output * p.n_input],
            baseline: vec![0.0; p.n_input],
            s,
            m,
            s_params,
            m_params,
            w_sm,
            row_target,
            pong,
            rng: Pcg64::seed_from_u64(seed.wrapping_add(0x51ED270B)),
            p,
        }
    }

    /// Play `n_game_steps` and return the learning log.
    // The hot loops index several parallel arrays (weights, currents, counts,
    // traces) by the same index, where explicit indexing is clearer than zips.
    #[allow(clippy::needless_range_loop)]
    pub fn run(&mut self, n_game_steps: usize) -> RunLog {
        let p = self.p.clone();
        let (ni, no) = (p.n_input, p.n_output);
        let decay_trace = (-p.dt / p.tau_trace).exp();
        let decay_elig = (-p.dt / p.tau_elig).exp();

        let mut game = self.pong.reset(&mut self.rng);
        let mut paddle_target = 0.5f32;

        let mut log = RunLog {
            rally_lengths: Vec::new(),
            events: Vec::new(),
            hits: 0,
            misses: 0,
            population_rate_hz: Vec::with_capacity(n_game_steps),
        };

        let mut s_current = vec![0.0f32; ni];
        let mut m_current = vec![0.0f32; no];
        let mut m_counts = vec![0.0f32; no];

        for step in 0..n_game_steps {
            // --- Sensory encoding: Gaussian bump centred on the ball's Y. ---
            let center = game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0);
            for (i, c) in s_current.iter_mut().enumerate() {
                let d = i as f32 - center;
                *c = p.input_amp * (-(d * d) / (2.0 * p.input_sigma * p.input_sigma)).exp();
            }

            // Exploration noise decays over training.
            let frac = (step as f32 / p.explore_decay_steps as f32).min(1.0);
            let expl = p.explore_std0 + (p.explore_std_min - p.explore_std0) * frac;

            m_counts.iter_mut().for_each(|c| *c = 0.0);
            let mut m_total = 0.0f32;

            for _ in 0..p.window_steps {
                izhikevich_step(&mut self.s, &self.s_params, &s_current, p.dt);

                // Motor input = W_sm · s_spikes + exploration noise.
                for j in 0..no {
                    let row = &self.w_sm[j * ni..(j + 1) * ni];
                    let mut inp = 0.0;
                    for i in 0..ni {
                        inp += row[i] * self.s.spikes[i];
                    }
                    m_current[j] = inp + expl * gaussian(&mut self.rng);
                }
                izhikevich_step(&mut self.m, &self.m_params, &m_current, p.dt);

                // Decay traces, deposit spikes, accumulate eligibility.
                for t in &mut self.s_trace {
                    *t *= decay_trace;
                }
                for t in &mut self.m_trace {
                    *t *= decay_trace;
                }
                for i in 0..ni {
                    if self.s.spikes[i] != 0.0 {
                        self.s_trace[i] += 1.0;
                    }
                }
                for j in 0..no {
                    if self.m.spikes[j] != 0.0 {
                        self.m_trace[j] += 1.0;
                    }
                    m_counts[j] += self.m.spikes[j];
                    m_total += self.m.spikes[j];
                }
                for j in 0..no {
                    let post = self.m.spikes[j];
                    let post_tr = self.m_trace[j];
                    for i in 0..ni {
                        let kernel = post * self.s_trace[i] - post_tr * self.s.spikes[i];
                        let e = &mut self.elig[j * ni + i];
                        *e = *e * decay_elig + kernel;
                    }
                }
            }
            log.population_rate_hz
                .push(m_total / no as f32 / (p.window_steps as f32 * p.dt / 1000.0));

            // --- Decode paddle target = centre of mass of motor activity. ---
            let mass: f32 = m_counts.iter().sum();
            if mass > 0.0 {
                let com: f32 = m_counts
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| j as f32 * c)
                    .sum::<f32>()
                    / mass;
                paddle_target = com / (no as f32 - 1.0);
            }

            // --- Per-output-neuron graded reward (Wunderlich): each motor neuron
            //     is rewarded by how close its coded position is to the ball, so
            //     the scalar reward is high only when firing is CONCENTRATED at
            //     the right position — spread or mis-placed firing is penalised.
            //     This is the pressure that forms the S→M diagonal; a COM-only
            //     reward tolerates spread and never learns. ---
            let mut r_acc = 0.0;
            for j in 0..no {
                let pos_j = j as f32 / (no as f32 - 1.0);
                let rw = 1.0 - 2.0 * (pos_j - game.ball_y).abs();
                r_acc += rw * m_counts[j];
            }
            let reward = if mass > 1e-6 { r_acc / mass } else { 0.0 };
            let bin = (game.ball_y.clamp(0.0, 1.0) * (ni as f32 - 1.0)).round() as usize;
            let rpe = reward - self.baseline[bin];
            self.baseline[bin] =
                (1.0 - p.reward_alpha) * self.baseline[bin] + p.reward_alpha * reward;

            // --- R-STDP: Δw = β (R − R̄) · eligibility, then homeostasis. ---
            for j in 0..no {
                for i in 0..ni {
                    let k = j * ni + i;
                    let w = self.w_sm[k] + p.learning_rate * rpe * self.elig[k];
                    self.w_sm[k] = w.clamp(p.w_min, p.w_max);
                }
                // Renormalise this output neuron's incoming weights to keep total
                // drive bounded while letting learning redistribute it.
                let row = &mut self.w_sm[j * ni..(j + 1) * ni];
                let sum: f32 = row.iter().sum();
                if sum > 1e-6 {
                    let scale = self.row_target / sum;
                    for w in row.iter_mut() {
                        *w *= scale;
                    }
                }
            }

            // --- Advance the game: paddle pursues the decoded target. ---
            let dead = 0.02;
            let action = if paddle_target > game.paddle_y + dead {
                Action::Up
            } else if paddle_target < game.paddle_y - dead {
                Action::Down
            } else {
                Action::Stay
            };
            let (next, event) = self.pong.step(&game, action, &mut self.rng);
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
    fn agent_runs_and_scores() {
        let mut a = RstdpAgent::new(AgentParams::default(), 1);
        let log = a.run(300);
        assert!(log.hits + log.misses > 0, "should produce events");
        assert_eq!(log.population_rate_hz.len(), 300);
    }

    #[test]
    fn agent_is_reproducible() {
        let mut a = RstdpAgent::new(AgentParams::default(), 7);
        let mut b = RstdpAgent::new(AgentParams::default(), 7);
        let la = a.run(200);
        let lb = b.run(200);
        assert_eq!(la.hits, lb.hits);
        assert_eq!(la.misses, lb.misses);
    }
}
