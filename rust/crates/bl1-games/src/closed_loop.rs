//! The DishBrain closed loop: a cultured network learns to play Pong.
//!
//! Each game step runs one neural window: decode a paddle action from the motor
//! regions, advance the game, deliver FEP feedback for the event, encode the new
//! ball position as sensory stimulation, simulate the window, and update weights
//! with reward-modulated STDP at every sub-step (hit rewards, miss punishes).
//! Weights change online, so the culture can reorganise toward hitting.

use bl1_core::{CsrMatrix, SimState, simulate};
use bl1_mea::{MeaConfig, Position, build_neuron_electrode_map};
use bl1_sim::{Config, Culture};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

use crate::decoding::MotorDecoder;
use crate::encoding::SensoryEncoder;
use crate::feedback::FeedbackProtocol;
use crate::plasticity::{Reward, ThreeFactorParams, ThreeFactorStdp};
use crate::pong::{Event, Pong, PongState};

/// Tunable parameters for one closed-loop experiment.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Duration of one game step's neural window (ms).
    pub game_step_ms: f32,
    /// Number of vertical sensory bands (= number of sensory electrodes).
    pub n_sensory: usize,
    /// Motor-decode baseline rate (Hz).
    pub baseline_rate_hz: f32,
    /// Ball speed (field fractions per game step); larger = more rallies per run.
    pub ball_speed: f32,
    /// Sensory stimulation amplitude (current units, comparable to the drive scale).
    pub sensory_amplitude: f32,
    /// Reward-modulated STDP parameters.
    pub plasticity: ThreeFactorParams,
    /// Dopamine-like reward signal (hit/miss amplitudes + decay).
    pub reward: Reward,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            game_step_ms: 50.0,
            n_sensory: 8,
            baseline_rate_hz: 3.0,
            ball_speed: 0.03,
            // Tuned by the sweep: lower sensory amplitude avoids saturating the
            // culture and preserves the differential motor signal.
            sensory_amplitude: 8.0,
            plasticity: ThreeFactorParams::default(),
            reward: Reward::default(),
        }
    }
}

/// Metrics collected across a run — the reproducible learning signal.
pub struct RunLog {
    /// Completed rally lengths, in order (one entry per miss).
    pub rally_lengths: Vec<u32>,
    /// `(game_step, event)` for every hit/miss.
    pub events: Vec<(usize, Event)>,
    pub hits: u32,
    pub misses: u32,
    /// Mean population firing rate (Hz) per game step.
    pub population_rate_hz: Vec<f32>,
}

impl RunLog {
    /// Overall hit rate over the whole run.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Learning score: second-half hit rate minus first-half hit rate over the
    /// run's events. Positive = the culture improved. More robust to per-block
    /// noise than eyeballing the curve; average it across seeds when tuning.
    pub fn improvement(&self) -> f32 {
        let n = self.events.len();
        if n < 2 {
            return 0.0;
        }
        let mid = n / 2;
        let rate = |slice: &[(usize, Event)]| {
            if slice.is_empty() {
                0.0
            } else {
                slice.iter().filter(|(_, e)| *e == Event::Hit).count() as f32 / slice.len() as f32
            }
        };
        rate(&self.events[mid..]) - rate(&self.events[..mid])
    }

    /// Hit rate within each consecutive block of `block` events — the learning
    /// curve. A rising series means the culture is improving.
    pub fn hit_rate_curve(&self, block: usize) -> Vec<f32> {
        if block == 0 {
            return Vec::new();
        }
        let mut curve = Vec::new();
        for chunk in self.events.chunks(block) {
            let hits = chunk.iter().filter(|(_, e)| *e == Event::Hit).count();
            curve.push(hits as f32 / chunk.len() as f32);
        }
        curve
    }
}

/// A ready-to-run closed-loop Pong experiment.
pub struct ClosedLoop {
    culture: Culture,
    state: SimState,
    pong: Pong,
    game: PongState,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    feedback: FeedbackProtocol,
    plasticity: ThreeFactorStdp,
    reward: Reward,
    /// Neuron indices reachable by each electrode (for sensory injection).
    stim_neurons: Vec<Vec<usize>>,
    rng: Pcg64,
    dt: f32,
    steps_per_game: usize,
    n: usize,
}

impl ClosedLoop {
    /// Build a closed loop from a culture [`Config`], seeding all randomness.
    pub fn new(config: &Config, cfg: &LoopConfig, seed: u64) -> Self {
        let culture = Culture::build(config, seed);
        let state = culture.make_sim_state();
        let n = culture.n_neurons();
        let dt = culture.dt_ms.max(0.01);
        let steps_per_game = ((cfg.game_step_ms / dt).round() as usize).max(1);

        // Build an electrode grid that tiles the culture substrate.
        let mea = tiled_mea(config.culture.substrate_um, 8, 8);
        let cols = 8usize;

        // Sensory electrodes: the leftmost column, one per vertical band.
        let sensory: Vec<usize> = (0..cfg.n_sensory).map(|r| r * cols).collect();

        // Stimulation map (injection reaches neurons within the activation
        // radius of each electrode).
        let mut stim_mea = mea.clone();
        stim_mea.detection_radius = mea.activation_radius;
        let stim_map = build_neuron_electrode_map(&stim_mea, &culture.positions);
        let stim_neurons: Vec<Vec<usize>> = (0..mea.n_electrodes())
            .map(|e| csr_row(&stim_map, e))
            .collect();

        // Motor readout = the sensory-driven neurons themselves, split into the
        // top ("up") and bottom ("down") bands. Reading the same population that
        // sensory stimulation targets keeps the differential signal strong
        // instead of averaging it away over the whole culture.
        let half = cfg.n_sensory / 2;
        let up_e: Vec<usize> = sensory[half..].to_vec();
        let down_e: Vec<usize> = sensory[..half].to_vec();
        let up_neurons = union_rows(&stim_map, &up_e);
        let down_neurons = union_rows(&stim_map, &down_e);

        let mut decoder = MotorDecoder::new(up_neurons, down_neurons);
        decoder.baseline_rate_hz = cfg.baseline_rate_hz;
        decoder.margin_hz = 1.0;

        let mut rng = Pcg64::seed_from_u64(seed.wrapping_add(0x9E3779B9));
        let pong = Pong {
            ball_speed: cfg.ball_speed,
            ..Pong::default()
        };
        let game = pong.reset(&mut rng);

        let nnz = culture.network.w_exc.data.len();
        let plasticity = ThreeFactorStdp::new(n, nnz, cfg.plasticity.clone());

        let mut encoder = SensoryEncoder::new(sensory);
        encoder.amplitude = cfg.sensory_amplitude;

        Self {
            culture,
            state,
            pong,
            game,
            encoder,
            decoder,
            feedback: FeedbackProtocol::default(),
            plasticity,
            reward: cfg.reward.clone(),
            stim_neurons,
            rng,
            dt,
            steps_per_game,
            n,
        }
    }

    /// Run `n_game_steps` game steps and return the collected metrics.
    pub fn run(&mut self, n_game_steps: usize) -> RunLog {
        let window_ms = self.steps_per_game as f32 * self.dt;
        let game_step_s = window_ms / 1000.0;
        let n = self.n;

        let mut window_counts = vec![0.0f32; n];
        let mut drive = vec![0.0f32; n];
        let mut sensory = vec![0.0f32; n];

        let mut log = RunLog {
            rally_lengths: Vec::new(),
            events: Vec::new(),
            hits: 0,
            misses: 0,
            population_rate_hz: Vec::with_capacity(n_game_steps),
        };

        for step in 0..n_game_steps {
            // 1. Decode an action from the previous window's activity.
            let (action, _up, _down) = self.decoder.decode(&window_counts, window_ms);

            // 2. Advance the game.
            let prev = self.game;
            let (next, event) = self.pong.step(&prev, action, &mut self.rng);
            self.game = next;
            match event {
                Event::Hit => {
                    log.hits += 1;
                    log.events.push((step, event));
                    self.reward.reward(); // dopamine pulse: consolidate eligible synapses
                }
                Event::Miss => {
                    log.misses += 1;
                    log.rally_lengths.push(prev.rally_length);
                    log.events.push((step, event));
                    self.reward.punish(); // negative reward: depress eligible synapses
                }
                Event::None => {}
            }

            // 3. Feedback current for the event (applied to the first sub-step).
            let feedback = self.feedback.current(event, n, &mut self.rng);

            // 4. Encode the new ball position as sensory stimulation (constant
            //    across the window when a pulse fires this step).
            let stim = self.encoder.encode(next.ball_x, next.ball_y, game_step_s);
            sensory.iter_mut().for_each(|v| *v = 0.0);
            for &e in &stim.electrodes {
                for &neuron in &self.stim_neurons[e] {
                    sensory[neuron] += stim.amplitude;
                }
            }

            // 5. Simulate the window one sub-step at a time. Reward-modulated
            //    STDP accumulates eligibility every step and, while the reward
            //    signal is non-zero (just after an event), consolidates it into
            //    the weights. The reward decays across sub-steps.
            window_counts.iter_mut().for_each(|v| *v = 0.0);
            let mut total = 0.0f32;
            for sub in 0..self.steps_per_game {
                for j in 0..n {
                    drive[j] = self.culture.bg_mean
                        + self.culture.bg_std * gaussian(&mut self.rng)
                        + sensory[j];
                }
                if sub == 0 {
                    for j in 0..n {
                        drive[j] += feedback[j];
                    }
                }
                let _ = simulate(&self.culture.network, &mut self.state, &drive, 1, self.dt);
                let spikes = &self.state.neuron.spikes;
                self.plasticity.step(
                    spikes,
                    &mut self.culture.network.w_exc,
                    self.reward.level,
                    self.dt,
                );
                self.reward.decay(self.dt);
                for j in 0..n {
                    window_counts[j] += spikes[j];
                    total += spikes[j];
                }
            }
            log.population_rate_hz.push(total / n as f32 / game_step_s);
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

/// Neuron indices in CSR row `e` (electrode → detected/reachable neurons).
fn csr_row(map: &CsrMatrix, e: usize) -> Vec<usize> {
    map.indices[map.indptr[e]..map.indptr[e + 1]].to_vec()
}

/// Union of the neuron indices reached by a set of electrodes.
fn union_rows(map: &CsrMatrix, electrodes: &[usize]) -> Vec<usize> {
    let mut seen = vec![false; map.n_cols];
    let mut out = Vec::new();
    for &e in electrodes {
        for &i in &map.indices[map.indptr[e]..map.indptr[e + 1]] {
            if !seen[i] {
                seen[i] = true;
                out.push(i);
            }
        }
    }
    out
}

/// An electrode grid that tiles a `[w, h]` substrate with `rows × cols`
/// electrodes at cell centres; detection/activation radius = 0.6 × spacing.
fn tiled_mea(substrate: [f32; 2], rows: usize, cols: usize) -> MeaConfig {
    let [w, h] = substrate;
    let sx = w / cols as f32;
    let sy = h / rows as f32;
    let mut positions: Vec<Position> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            positions.push([(c as f32 + 0.5) * sx, (r as f32 + 0.5) * sy]);
        }
    }
    let radius = 0.6 * sx.min(sy);
    MeaConfig {
        name: "tiled".to_string(),
        positions,
        detection_radius: radius,
        activation_radius: radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> Config {
        Config::from_yaml_str(
            "culture:\n  n_neurons: 300\n  substrate_um: [1400, 1400]\n  p_max: 0.25\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n",
        )
        .unwrap()
    }

    #[test]
    fn loop_runs_and_produces_events() {
        let cfg = small_config();
        let mut loop_ = ClosedLoop::new(&cfg, &LoopConfig::default(), 42);
        // The ball needs ~1/ball_speed steps to cross, so run enough to score.
        let log = loop_.run(200);
        assert!(
            log.hits + log.misses > 0,
            "expected some hit/miss events, got none"
        );
        assert_eq!(log.population_rate_hz.len(), 200);
    }

    #[test]
    fn is_reproducible() {
        let cfg = small_config();
        let mut a = ClosedLoop::new(&cfg, &LoopConfig::default(), 7);
        let mut b = ClosedLoop::new(&cfg, &LoopConfig::default(), 7);
        let la = a.run(30);
        let lb = b.run(30);
        assert_eq!(la.hits, lb.hits);
        assert_eq!(la.misses, lb.misses);
        assert_eq!(la.rally_lengths, lb.rally_lengths);
    }

    #[test]
    fn electrode_regions_have_neurons() {
        let cfg = small_config();
        let loop_ = ClosedLoop::new(&cfg, &LoopConfig::default(), 1);
        assert!(!loop_.decoder.up_neurons.is_empty(), "up region empty");
        assert!(!loop_.decoder.down_neurons.is_empty(), "down region empty");
    }
}
