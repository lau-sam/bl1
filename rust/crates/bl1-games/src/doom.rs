//! A minimal DOOM-style aim-and-shoot arena for closed-loop culture experiments.
//!
//! The culture stands in a first-person arena; an enemy appears at some bearing
//! and closes in. The culture senses the enemy's bearing as a Gaussian bump over
//! its "retina" bands, and its readout drives the **view bearing** (where the
//! crosshair points). When the enemy reaches firing range the shot resolves:
//! aligned within a tolerance ⇒ **kill** (predictable, rewarded), otherwise the
//! enemy strikes ⇒ **miss** (Kagan-2022-style unpredictable feedback).
//!
//! Structurally this mirrors [`crate::pong::Pong`] so the exact same learning
//! machinery applies: the enemy launches centred and drifts (a moving target,
//! like the ball's vertical bounce), a dense tracking reward
//! (`1 − 2·|aim − enemy|`) shapes the aim each frame, and one encounter yields
//! one hit/miss outcome. Kill = hit, miss = miss, kill-streak = rally length.

use rand::Rng;
use rand_pcg::Pcg64;

use crate::env::{EnvView, Environment, GameKind};
use crate::pong::Event;

/// Arena dynamics and difficulty.
#[derive(Debug, Clone, Copy)]
pub struct DoomParams {
    /// Aim tolerance for a kill: `|aim − enemy| ≤ hit_tol` (≈ paddle half-height).
    pub hit_tol: f32,
    /// Enemy bearing drift per frame — a moving target (≈ ball speed).
    pub enemy_speed: f32,
    /// Frames from spawn to the shot resolving (the enemy closing distance).
    pub encounter_frames: u32,
}

impl Default for DoomParams {
    fn default() -> Self {
        Self {
            hit_tol: 0.1,
            enemy_speed: 0.03,
            encounter_frames: 33,
        }
    }
}

/// Complete arena state (all bearings span `[0, 1]`; crosshair is screen-centre).
#[derive(Debug, Clone, Copy)]
pub struct DoomState {
    /// View bearing the culture controls (crosshair points here).
    pub heading: f32,
    /// Enemy bearing.
    pub enemy_x: f32,
    /// Enemy bearing drift per frame.
    pub enemy_vx: f32,
    /// Frames until this encounter resolves (→ enemy distance / on-screen size).
    pub countdown: u32,
    /// Encounter length, for rendering distance as `countdown / encounter_frames`.
    pub encounter_frames: u32,
    pub kills: u32,
    pub misses: u32,
    /// Current consecutive-kill streak.
    pub streak: u32,
    /// Muzzle-flash frames remaining (render only).
    pub flash: u8,
    /// The last resolved outcome (colours the flash).
    pub last: Event,
}

/// The aim-and-shoot arena as an [`Environment`].
pub struct DoomArena {
    p: DoomParams,
    state: DoomState,
}

impl DoomArena {
    /// A fresh arena; the seed drives the first enemy's drift.
    pub fn new(p: DoomParams, rng: &mut Pcg64) -> Self {
        let mut arena = Self {
            p,
            state: DoomState {
                heading: 0.5,
                enemy_x: 0.5,
                enemy_vx: 0.0,
                countdown: p.encounter_frames,
                encounter_frames: p.encounter_frames,
                kills: 0,
                misses: 0,
                streak: 0,
                flash: 0,
                last: Event::None,
            },
        };
        arena.spawn(rng);
        arena
    }

    /// Launch a new enemy: centred, drifting at a random angle (like the ball).
    fn spawn<R: Rng>(&mut self, rng: &mut R) {
        let angle = rng.random_range(-std::f32::consts::FRAC_PI_4..std::f32::consts::FRAC_PI_4);
        self.state.enemy_x = 0.5;
        self.state.enemy_vx = self.p.enemy_speed * angle.sin();
        self.state.countdown = self.p.encounter_frames;
    }
}

impl Environment for DoomArena {
    fn sensory_position(&self) -> f32 {
        self.state.enemy_x
    }

    fn actuator_position(&self) -> f32 {
        self.state.heading
    }

    fn step(&mut self, pos: f32, rng: &mut Pcg64) -> (f32, Event) {
        // Dense aim reward on the commanded bearing vs. the enemy's current
        // bearing (before it drifts), then point the view there.
        let reward = 1.0 - 2.0 * (pos - self.state.enemy_x).abs();
        self.state.heading = pos;
        if self.state.flash > 0 {
            self.state.flash -= 1;
        }

        let event = if self.state.countdown <= 1 {
            // Enemy in range: resolve the shot on the current bearing.
            let hit = (pos - self.state.enemy_x).abs() <= self.p.hit_tol;
            self.state.flash = 3;
            let event = if hit {
                self.state.kills += 1;
                self.state.streak += 1;
                self.state.last = Event::Hit;
                Event::Hit
            } else {
                self.state.misses += 1;
                self.state.streak = 0;
                self.state.last = Event::Miss;
                Event::Miss
            };
            self.spawn(rng);
            event
        } else {
            // Enemy closes in and drifts (bounces off the arena edges).
            self.state.countdown -= 1;
            self.state.enemy_x += self.state.enemy_vx;
            if self.state.enemy_x <= 0.0 || self.state.enemy_x >= 1.0 {
                self.state.enemy_vx = -self.state.enemy_vx;
            }
            self.state.enemy_x = self.state.enemy_x.clamp(0.0, 1.0);
            Event::None
        };
        (reward, event)
    }

    fn view(&self) -> EnvView<'_> {
        EnvView::Doom(&self.state)
    }

    fn kind(&self) -> GameKind {
        GameKind::Doom
    }

    fn reset(&mut self, rng: &mut Pcg64) {
        self.state.kills = 0;
        self.state.misses = 0;
        self.state.streak = 0;
        self.state.flash = 0;
        self.state.last = Event::None;
        self.state.heading = 0.5;
        self.spawn(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn enemy_spawns_centred_and_resolves() {
        let mut rng = Pcg64::seed_from_u64(1);
        let arena = DoomArena::new(DoomParams::default(), &mut rng);
        assert_eq!(arena.state.enemy_x, 0.5);
        assert_eq!(arena.state.countdown, 33);
    }

    #[test]
    fn aligned_shot_kills() {
        let mut rng = Pcg64::seed_from_u64(1);
        let p = DoomParams {
            encounter_frames: 1,
            enemy_speed: 0.0, // hold the enemy still for a deterministic check
            ..DoomParams::default()
        };
        let mut arena = DoomArena::new(p, &mut rng);
        // Enemy is centred at 0.5; aim dead-on -> kill on the resolving frame.
        let (_r, ev) = arena.step(0.5, &mut rng);
        assert_eq!(ev, Event::Hit);
        assert_eq!(arena.state.kills, 1);
        assert_eq!(arena.state.streak, 1);
    }

    #[test]
    fn misaligned_shot_misses_and_resets_streak() {
        let mut rng = Pcg64::seed_from_u64(1);
        let p = DoomParams {
            encounter_frames: 1,
            enemy_speed: 0.0,
            ..DoomParams::default()
        };
        let mut arena = DoomArena::new(p, &mut rng);
        arena.state.streak = 4;
        // Enemy centred at 0.5, aim far off -> miss.
        let (_r, ev) = arena.step(0.05, &mut rng);
        assert_eq!(ev, Event::Miss);
        assert_eq!(arena.state.misses, 1);
        assert_eq!(arena.state.streak, 0);
    }

    #[test]
    fn dense_reward_peaks_on_target() {
        let mut rng = Pcg64::seed_from_u64(1);
        let mut arena = DoomArena::new(DoomParams::default(), &mut rng);
        let (r_on, _) = arena.step(arena.state.enemy_x, &mut rng);
        assert!(r_on > 0.99, "aiming on the enemy should reward ~1, got {r_on}");
    }
}
