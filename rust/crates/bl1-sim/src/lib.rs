//! `bl1-sim` — culture construction and configuration.
//!
//! Places neurons on a substrate, wires them with distance-dependent
//! connectivity, and assembles a [`bl1_core::Network`] plus the state and drive
//! needed to run it. Configuration mirrors the project's `configs/*.yaml`.

pub mod config;
pub mod connectivity;
pub mod culture;
pub mod placement;

pub use config::Config;
pub use connectivity::build_connectivity;
pub use culture::Culture;
pub use placement::{Position, distance, place_neurons};
