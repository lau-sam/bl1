//! `bl1-games` — closed-loop DishBrain game experiments on a native Rust
//! cortical-culture simulation.
//!
//! A simulated culture plays Pong: ball position is encoded as sensory
//! stimulation, motor regions decode a paddle action, and free-energy-principle
//! feedback (predictable on a hit, unpredictable on a miss; Kagan 2022) drives
//! online STDP so the culture can reorganise. Learning is measured as rally
//! length and hit rate over time.

pub mod closed_loop;
pub mod decoding;
pub mod encoding;
pub mod feedback;
pub mod pong;

pub use closed_loop::{ClosedLoop, LoopConfig, RunLog};
pub use pong::{Action, Event, Pong, PongState};
