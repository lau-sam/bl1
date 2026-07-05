//! The **environment**: the game the culture plays.
//!
//! An [`Environment`] hides everything game-specific behind three moves the
//! generic [`crate::Learner`] needs: expose the scalar to sense
//! ([`Environment::sensory_position`]), consume the decoded actuator command and
//! advance one frame returning a dense reward + a discrete outcome
//! ([`Environment::step`]), and hand a render-agnostic snapshot to the UI
//! ([`Environment::view`]). Add a new game by implementing this trait — the
//! learning machinery is reused untouched.

use rand_pcg::Pcg64;

use crate::doom::DoomState;
use crate::pong::{Event, PongState};

/// Which game an environment is (for save/load + choosing the UI renderer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameKind {
    Pong,
    Doom,
}

impl GameKind {
    pub fn label(self) -> &'static str {
        match self {
            GameKind::Pong => "pong",
            GameKind::Doom => "doom",
        }
    }
}

/// A render-agnostic snapshot of the current game state, so the UI can draw the
/// right scene without the learner knowing anything about pixels.
pub enum EnvView<'a> {
    Pong(&'a PongState),
    Doom(&'a DoomState),
}

/// A game the culture can learn to play through a 1-D actuator.
pub trait Environment {
    /// The scalar the culture senses this frame, normalised to `[0, 1]`
    /// (ball height, enemy bearing, …). The learner encodes it as a bump.
    fn sensory_position(&self) -> f32;

    /// The actuator's current position in `[0, 1]` (paddle Y, view bearing, …),
    /// so the learner can apply inertial smoothing relative to it.
    fn actuator_position(&self) -> f32;

    /// Command the actuator to `pos` (already smoothed, in `[0, 1]`), compute the
    /// dense tracking reward against the *current* target, advance one frame, and
    /// return `(reward, event)`. Any RNG draws (re-spawns) come last.
    fn step(&mut self, pos: f32, rng: &mut Pcg64) -> (f32, Event);

    /// A snapshot for the UI renderer.
    fn view(&self) -> EnvView<'_>;

    /// Which game this is.
    fn kind(&self) -> GameKind;

    /// Re-initialise to a fresh episode (consumes RNG for the first spawn).
    fn reset(&mut self, rng: &mut Pcg64);
}
