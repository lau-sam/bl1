//! The neural **substrate**: what turns a sensory bump into a feature vector.
//!
//! Every learnable game in this crate shares one recipe — encode a scalar the
//! game exposes (ball height, enemy bearing, …) as a Gaussian bump over a bank
//! of bands, run a spiking network for one neural window, and read a place code
//! (mean spike rate per band). Only *how the spikes are produced* differs, and
//! that is exactly what a [`Substrate`] abstracts:
//!
//! - [`FeedForwardBank`] — an independent bank of Izhikevich neurons per band.
//!   Clean, deterministic, sharp; the representation that tracks best.
//! - [`CultureReservoir`] — a full recurrent [`bl1_sim::Culture`] used as a fixed
//!   reservoir. Honest recurrent dynamics (distance-wired, STP, background
//!   noise) smear the place code, so it tracks less sharply but it *is* the
//!   living-culture model computing.
//!
//! Both return a **sum-1 normalised** per-band feature vector, so the readout's
//! learning signal `Δw ∝ x` stays well-scaled no matter how sparsely the network
//! fires. The recurrent reservoir consumes background-noise draws from the shared
//! RNG inside its window; the feed-forward bank consumes none — preserving each
//! substrate's exact draw order is what keeps learning curves reproducible.

use bl1_core::{IzhParams, NeuronState, SimState, build_population, izhikevich_step, simulate};
use bl1_sim::{Config, Culture};
use rand::Rng;
use rand_pcg::Pcg64;

/// Standard normal via Box-Muller (shared by every substrate + the policy).
pub(crate) fn gaussian<R: Rng>(rng: &mut R) -> f32 {
    let u1 = rng.random::<f32>().max(1e-7);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Which neural substrate produced the features (for save/load + the UI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateKind {
    /// Independent Izhikevich bank per band (feed-forward).
    FeedForward,
    /// Full recurrent [`bl1_sim::Culture`] used as a fixed reservoir.
    Reservoir,
}

impl SubstrateKind {
    /// Short human label for a live UI.
    pub fn label(self) -> &'static str {
        match self {
            SubstrateKind::FeedForward => "feed-forward bank",
            SubstrateKind::Reservoir => "recurrent culture",
        }
    }
}

/// A neural representation: map a per-band sensory bump through one spiking
/// window and read a sum-1 normalised place code back out.
pub trait Substrate {
    /// Run one neural window on `bump` (one current per band) and return the
    /// sum-1 normalised per-band feature vector (`len == n_bands`). May draw from
    /// `rng` (recurrent background noise); the draw order is part of the contract.
    fn encode(&mut self, bump: &[f32], rng: &mut Pcg64) -> &[f32];
    /// Mean population firing rate (Hz) from the most recent [`Self::encode`].
    fn population_rate_hz(&self) -> f32;
    /// Number of bands (= place-code resolution = readout dimensionality).
    fn n_bands(&self) -> usize;
    /// Which substrate this is.
    fn kind(&self) -> SubstrateKind;
}

// ---------------------------------------------------------------------------
// Feed-forward bank
// ---------------------------------------------------------------------------

/// A bank of `n_bands × per_band` Izhikevich neurons: every band gets its own
/// group, driven only by that band's sensory current. Deterministic given the
/// input — it draws nothing from the RNG.
pub struct FeedForwardBank {
    n_bands: usize,
    per_band: usize,
    window_steps: usize,
    dt: f32,
    params: IzhParams,
    state: NeuronState,
    current: Vec<f32>,
    features: Vec<f32>,
    rate_hz: f32,
}

impl FeedForwardBank {
    pub fn new(n_bands: usize, per_band: usize, window_steps: usize, dt: f32) -> Self {
        let params = build_population(n_bands * per_band, 1.0).params;
        let state = NeuronState::resting(&params);
        Self {
            n_bands,
            per_band,
            window_steps,
            dt,
            params,
            state,
            current: vec![0.0; n_bands * per_band],
            features: vec![0.0; n_bands],
            rate_hz: 0.0,
        }
    }
}

impl Substrate for FeedForwardBank {
    // Hot loop indexes parallel per-band arrays.
    #[allow(clippy::needless_range_loop)]
    fn encode(&mut self, bump: &[f32], _rng: &mut Pcg64) -> &[f32] {
        let (ni, pb) = (self.n_bands, self.per_band);
        // Apply each band's bump current to all of that band's neurons.
        for b in 0..ni {
            let cur = bump[b];
            for k in 0..pb {
                self.current[b * pb + k] = cur;
            }
        }
        // Run the window; feature = summed spikes per band.
        self.features.iter_mut().for_each(|v| *v = 0.0);
        let mut total = 0.0f32;
        for _ in 0..self.window_steps {
            izhikevich_step(&mut self.state, &self.params, &self.current, self.dt);
            for b in 0..ni {
                for k in 0..pb {
                    let sp = self.state.spikes[b * pb + k];
                    self.features[b] += sp;
                    total += sp;
                }
            }
        }
        // Sum-1 normalisation (per-band population size cancels, so skip it).
        let sum: f32 = self.features.iter().sum();
        if sum > 1e-6 {
            for v in self.features.iter_mut() {
                *v /= sum;
            }
        }
        let ns = (ni * pb) as f32;
        self.rate_hz = total / ns / (self.window_steps as f32 * self.dt / 1000.0);
        &self.features
    }

    fn population_rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn n_bands(&self) -> usize {
        self.n_bands
    }
    fn kind(&self) -> SubstrateKind {
        SubstrateKind::FeedForward
    }
}

// ---------------------------------------------------------------------------
// Recurrent-culture reservoir
// ---------------------------------------------------------------------------

/// Culture config for the reservoir, deterministic in `n_neurons`. A fixed
/// geometry keeps a saved brain reproducible from `(seed, n_neurons)` alone.
fn reservoir_config(n_neurons: usize) -> Config {
    let yaml = format!(
        "culture:\n  n_neurons: {n_neurons}\n  substrate_um: [1000, 1000]\n  p_max: 0.2\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n"
    );
    Config::from_yaml_str(&yaml).expect("reservoir config is valid")
}

/// The real recurrent culture used as a fixed reservoir: the bump is injected as
/// a current into a spatial band, the recurrent dynamics transform it, and we
/// read a place code. Recurrent weights + STP are never touched here — learning
/// lives entirely in the readout the [`crate::Learner`] owns.
pub struct CultureReservoir {
    n_bands: usize,
    window_steps: usize,
    culture: Culture,
    state: SimState,
    dt: f32,
    /// Band index for each neuron (by vertical position).
    band_of: Vec<usize>,
    /// Neuron count per band, for rate normalisation.
    band_size: Vec<f32>,
    features: Vec<f32>,
    drive: Vec<f32>,
    sensory: Vec<f32>,
    rate_hz: f32,
    n_neurons: usize,
}

impl CultureReservoir {
    pub fn new(n_neurons: usize, n_bands: usize, window_steps: usize, seed: u64) -> Self {
        let config = reservoir_config(n_neurons);
        let culture = Culture::build(&config, seed);
        let state = culture.make_sim_state();
        let n = culture.n_neurons();
        let dt = culture.dt_ms.max(0.01);

        // Bin neurons into vertical bands by their Y position on the substrate.
        let height = config.culture.substrate_um[1].max(1.0);
        let mut band_of = vec![0usize; n];
        let mut band_size = vec![0.0f32; n_bands];
        for (i, pos) in culture.positions.iter().enumerate() {
            let frac = (pos[1] / height).clamp(0.0, 1.0);
            let band = ((frac * n_bands as f32) as usize).min(n_bands - 1);
            band_of[i] = band;
            band_size[band] += 1.0;
        }
        // Guard against empty bands (small cultures) — avoid divide-by-zero.
        for s in band_size.iter_mut() {
            if *s < 1.0 {
                *s = 1.0;
            }
        }

        Self {
            n_bands,
            window_steps,
            culture,
            state,
            dt,
            band_of,
            band_size,
            features: vec![0.0; n_bands],
            drive: vec![0.0; n],
            sensory: vec![0.0; n],
            rate_hz: 0.0,
            n_neurons,
        }
    }

    /// Reservoir size (recurrent neuron count).
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }
}

impl Substrate for CultureReservoir {
    #[allow(clippy::needless_range_loop)]
    fn encode(&mut self, bump: &[f32], rng: &mut Pcg64) -> &[f32] {
        let ni = self.n_bands;
        let n = self.culture.n_neurons();
        // Fan the per-band bump out to every neuron in that band.
        for i in 0..n {
            self.sensory[i] = bump[self.band_of[i]];
        }
        // Run the recurrent culture; feature = summed spikes per band. Background
        // noise draws from the shared RNG (this order must be preserved).
        self.features.iter_mut().for_each(|v| *v = 0.0);
        let mut total = 0.0f32;
        for _ in 0..self.window_steps {
            for j in 0..n {
                self.drive[j] = self.culture.bg_mean
                    + self.culture.bg_std * gaussian(rng)
                    + self.sensory[j];
            }
            let _ = simulate(&self.culture.network, &mut self.state, &self.drive, 1, self.dt);
            let spikes = &self.state.neuron.spikes;
            for j in 0..n {
                let sp = spikes[j];
                self.features[self.band_of[j]] += sp;
                total += sp;
            }
        }
        // Per-band mean rate (divide by band population), then sum-1 normalise.
        for b in 0..ni {
            self.features[b] /= self.band_size[b];
        }
        let sum: f32 = self.features.iter().sum();
        if sum > 1e-6 {
            for v in self.features.iter_mut() {
                *v /= sum;
            }
        }
        self.rate_hz = total / n as f32 / (self.window_steps as f32 * self.dt / 1000.0);
        &self.features
    }

    fn population_rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn n_bands(&self) -> usize {
        self.n_bands
    }
    fn kind(&self) -> SubstrateKind {
        SubstrateKind::Reservoir
    }
}
