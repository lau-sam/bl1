//! `bl1-core` — the forward spiking-network simulator ported from the
//! JAX-based BL-1 model.
//!
//! It provides Izhikevich (2003) and Adaptive Exponential (Brette & Gerstner
//! 2005) neurons, conductance-based AMPA/NMDA/GABA_A/GABA_B synapses, short-term
//! (Tsodyks-Markram) and spike-timing-dependent plasticity, and a time-stepping
//! integrator whose per-step ordering matches the reference implementation.
//!
//! Integration is forward (semi-implicit) Euler at a default step of
//! `dt = 0.5 ms`; conductances and plasticity traces decay by exact
//! exponentials `exp(-dt/tau)`.

pub mod integrator;
pub mod neuron;
pub mod plasticity;
pub mod populations;
pub mod sparse;
pub mod synapse;

pub use integrator::{Network, Raster, SimState, simulate};
pub use neuron::{AdExParams, AdExState, IzhParams, NeuronState, adex_step, izhikevich_step};
pub use plasticity::{StdpParams, StdpState, StpParams, StpState};
pub use populations::{Population, build_population};
pub use sparse::CsrMatrix;
pub use synapse::{SynapseState, nmda_mg_block, synaptic_current_into};

/// Default simulation timestep in milliseconds.
pub const DEFAULT_DT_MS: f32 = 0.5;
