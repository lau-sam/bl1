//! A common interface over the learnable Pong agents so a live UI can drive
//! either substrate — the feed-forward bank ([`crate::pursuit::PursuitAgent`])
//! or the recurrent culture ([`crate::reservoir::ReservoirAgent`]) — through one
//! handle. Both learn a linear readout by reward-modulated node perturbation;
//! they differ only in what produces the sensory feature vector.

use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::pong::{Event, PongState};
use crate::pursuit::{Brain, PaddleControl, PursuitAgent};
use crate::reservoir::ReservoirAgent;

/// The observable, steppable surface a live trainer view needs. Inherent
/// methods on each agent satisfy these (method resolution prefers them, so the
/// forwarding bodies below don't recurse).
pub trait Trainer {
    fn step(&mut self) -> Event;
    fn game(&self) -> &PongState;
    fn features(&self) -> &[f32];
    fn step_idx(&self) -> usize;
    fn hits(&self) -> u32;
    fn misses(&self) -> u32;
    fn last_target(&self) -> f32;
    fn sigma(&self) -> f32;
    fn control(&self) -> PaddleControl;
    fn hit_rate(&self) -> f32;
    fn hit_rate_curve(&self, block: usize) -> Vec<f32>;
    fn recent_outcomes(&self, n: usize) -> Vec<bool>;
    fn recent_hit_rate(&self, n: usize) -> f32;
    fn save(&self, path: &Path) -> Result<()>;
    /// Short label for the substrate driving the learning (for the UI).
    fn substrate(&self) -> &'static str;
}

macro_rules! impl_trainer {
    ($ty:ty, $substrate:expr) => {
        impl Trainer for $ty {
            fn step(&mut self) -> Event {
                <$ty>::step(self)
            }
            fn game(&self) -> &PongState {
                <$ty>::game(self)
            }
            fn features(&self) -> &[f32] {
                <$ty>::features(self)
            }
            fn step_idx(&self) -> usize {
                <$ty>::step_idx(self)
            }
            fn hits(&self) -> u32 {
                <$ty>::hits(self)
            }
            fn misses(&self) -> u32 {
                <$ty>::misses(self)
            }
            fn last_target(&self) -> f32 {
                <$ty>::last_target(self)
            }
            fn sigma(&self) -> f32 {
                <$ty>::sigma(self)
            }
            fn control(&self) -> PaddleControl {
                <$ty>::control(self)
            }
            fn hit_rate(&self) -> f32 {
                <$ty>::hit_rate(self)
            }
            fn hit_rate_curve(&self, block: usize) -> Vec<f32> {
                <$ty>::hit_rate_curve(self, block)
            }
            fn recent_outcomes(&self, n: usize) -> Vec<bool> {
                <$ty>::recent_outcomes(self, n)
            }
            fn recent_hit_rate(&self, n: usize) -> f32 {
                <$ty>::recent_hit_rate(self, n)
            }
            fn save(&self, path: &Path) -> Result<()> {
                <$ty>::save(self, path)
            }
            fn substrate(&self) -> &'static str {
                $substrate
            }
        }
    };
}

impl_trainer!(PursuitAgent, "feed-forward bank");
impl_trainer!(ReservoirAgent, "recurrent culture");

/// Load a shared brain file and rebuild the matching agent, dispatching on the
/// persisted `mode` tag (feed-forward pursuit vs. recurrent-culture reservoir).
pub fn load_trainer(path: &Path) -> Result<Box<dyn Trainer>> {
    let brain: Brain = serde_yaml::from_str(&fs::read_to_string(path)?)?;
    if brain.mode.contains("reservoir") {
        Ok(Box::new(ReservoirAgent::from_brain(&brain)))
    } else {
        Ok(Box::new(PursuitAgent::from_brain(&brain)))
    }
}
