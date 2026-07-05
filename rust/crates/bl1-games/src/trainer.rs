//! The object-safe surface a live UI needs to drive and inspect a [`Learner`],
//! plus [`load_trainer`] to rebuild the right learner from a shared brain file.
//! There is one learner type now — the substrate and game are swappable *inside*
//! it — so this is a thin observable façade rather than a polymorphism seam.

use std::path::Path;

use anyhow::Result;

use crate::env::{EnvView, GameKind};
use crate::learner::{Learner, PaddleControl};
use crate::pong::Event;

/// The observable, steppable surface a live trainer view needs.
pub trait Trainer {
    fn step(&mut self) -> Event;
    fn view(&self) -> EnvView<'_>;
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
    /// Which game the learner is playing (chooses the UI renderer).
    fn game_kind(&self) -> GameKind;
}

impl Trainer for Learner {
    fn step(&mut self) -> Event {
        Learner::step(self)
    }
    fn view(&self) -> EnvView<'_> {
        Learner::view(self)
    }
    fn features(&self) -> &[f32] {
        Learner::features(self)
    }
    fn step_idx(&self) -> usize {
        Learner::step_idx(self)
    }
    fn hits(&self) -> u32 {
        Learner::hits(self)
    }
    fn misses(&self) -> u32 {
        Learner::misses(self)
    }
    fn last_target(&self) -> f32 {
        Learner::last_target(self)
    }
    fn sigma(&self) -> f32 {
        Learner::sigma(self)
    }
    fn control(&self) -> PaddleControl {
        Learner::control(self)
    }
    fn hit_rate(&self) -> f32 {
        Learner::hit_rate(self)
    }
    fn hit_rate_curve(&self, block: usize) -> Vec<f32> {
        Learner::hit_rate_curve(self, block)
    }
    fn recent_outcomes(&self, n: usize) -> Vec<bool> {
        Learner::recent_outcomes(self, n)
    }
    fn recent_hit_rate(&self, n: usize) -> f32 {
        Learner::recent_hit_rate(self, n)
    }
    fn save(&self, path: &Path) -> Result<()> {
        Learner::save(self, path)
    }
    fn substrate(&self) -> &'static str {
        Learner::substrate_label(self)
    }
    fn game_kind(&self) -> GameKind {
        Learner::game_kind(self)
    }
}

/// Load a shared brain file and rebuild the matching learner (game + substrate +
/// control are all recovered from the persisted `mode` tag).
pub fn load_trainer(path: &Path) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(Learner::load(path)?))
}
