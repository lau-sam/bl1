//! `bl1-games` — closed-loop DishBrain game experiments on a native Rust
//! cortical-culture simulation.
//!
//! A simulated culture learns to play a game through one shared recipe: encode
//! what it senses as a Gaussian bump over a bank of bands, run a spiking network
//! for a neural window, read a place code, and train a linear readout by
//! reward-modulated node perturbation. Three pieces are swappable and composed
//! by the generic [`Learner`]:
//!
//! - a [`Substrate`] — the neural representation ([`substrate`]): a feed-forward
//!   Izhikevich bank or the real recurrent [`bl1_sim::Culture`] as a reservoir;
//! - an [`Environment`] — the game ([`env`]): [`pong`] (track the ball) or
//!   [`doom`] (aim at the enemy and shoot);
//! - a [`PaddleControl`] actuator — direct snap or inertial smooth pursuit.
//!
//! Add a game by implementing [`Environment`]; add a substrate by implementing
//! [`Substrate`]. The learning machinery is written once, in [`learner`].

pub mod agent;
pub mod closed_loop;
pub mod decoding;
pub mod doom;
pub mod encoding;
pub mod env;
pub mod feedback;
pub mod learner;
pub mod plasticity;
pub mod pong;
pub mod remote_brain;
pub mod substrate;
pub mod trainer;

pub use agent::{AgentParams, RstdpAgent};
pub use closed_loop::{ClosedLoop, LoopConfig, RunLog};
pub use doom::{DoomArena, DoomParams, DoomState};
pub use env::{EnvView, Environment, GameKind};
pub use learner::{Brain, EnvSpec, LearnParams, Learner, PaddleControl, SubstrateSpec};
pub use remote_brain::{BrainParams, RemoteBrain, RemoteBrainState};
pub use plasticity::{Reward, ThreeFactorParams, ThreeFactorStdp};
pub use pong::{Action, Event, Pong, PongEnv, PongState};
pub use substrate::{CultureReservoir, FeedForwardBank, Substrate, SubstrateKind};
pub use trainer::{Trainer, load_trainer};
