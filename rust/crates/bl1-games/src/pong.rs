//! Minimal Pong environment for closed-loop culture experiments.
//!
//! A direct port of the Python `bl1.games.pong` state machine (no JAX). The
//! ball moves from the left toward a paddle on the right edge; the culture's
//! decoded motor output moves the paddle. Hits and misses are returned as
//! events so the controller can deliver distinct feedback (Kagan 2022).

use rand::Rng;
use rand_pcg::Pcg64;

use crate::env::{EnvView, Environment, GameKind};

/// Paddle action decoded from the culture's motor region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Stay,
    Up,
    Down,
}

/// Event emitted when the ball reaches the paddle plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    None,
    Hit,
    Miss,
}

/// Complete game state (field spans `[0, 1]` in x and y).
#[derive(Debug, Clone, Copy)]
pub struct PongState {
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_vx: f32,
    pub ball_vy: f32,
    pub paddle_y: f32,
    pub score_hits: u32,
    pub score_misses: u32,
    pub rally_length: u32,
}

/// Pong dynamics and parameters.
#[derive(Debug, Clone, Copy)]
pub struct Pong {
    pub paddle_height: f32,
    pub paddle_speed: f32,
    pub ball_speed: f32,
}

impl Default for Pong {
    fn default() -> Self {
        Self {
            paddle_height: 0.2,
            paddle_speed: 0.03,
            ball_speed: 0.01,
        }
    }
}

impl Pong {
    /// Fresh state: ball at the left edge with a random rightward angle.
    pub fn reset<R: Rng>(&self, rng: &mut R) -> PongState {
        let mut state = PongState {
            ball_x: 0.0,
            ball_y: 0.5,
            ball_vx: self.ball_speed,
            ball_vy: 0.0,
            paddle_y: 0.5,
            score_hits: 0,
            score_misses: 0,
            rally_length: 0,
        };
        self.launch_ball(&mut state, rng);
        state
    }

    /// Advance one game tick. Returns the new state and any event.
    pub fn step<R: Rng>(
        &self,
        state: &PongState,
        action: Action,
        rng: &mut R,
    ) -> (PongState, Event) {
        let mut s = *state;

        // Paddle movement.
        let paddle_dy = match action {
            Action::Up => self.paddle_speed,
            Action::Down => -self.paddle_speed,
            Action::Stay => 0.0,
        };
        s.paddle_y = (s.paddle_y + paddle_dy).clamp(0.0, 1.0);

        // Ball movement.
        s.ball_x += s.ball_vx;
        s.ball_y += s.ball_vy;

        // Bounce off top/bottom walls.
        if s.ball_y <= 0.0 || s.ball_y >= 1.0 {
            s.ball_vy = -s.ball_vy;
        }
        s.ball_y = s.ball_y.clamp(0.0, 1.0);

        // Safety bounce off the left wall (ball should head right).
        if s.ball_x <= 0.0 {
            s.ball_vx = s.ball_vx.abs();
            s.ball_x = 0.0;
        }

        // Did the ball reach the paddle plane?
        let event = if s.ball_x >= 1.0 {
            let half = self.paddle_height / 2.0;
            let hit = s.ball_y >= s.paddle_y - half && s.ball_y <= s.paddle_y + half;
            if hit {
                s.score_hits += 1;
                s.rally_length += 1;
                self.launch_ball(&mut s, rng);
                Event::Hit
            } else {
                s.score_misses += 1;
                s.rally_length = 0;
                self.launch_ball(&mut s, rng);
                Event::Miss
            }
        } else {
            Event::None
        };

        (s, event)
    }

    /// Re-launch the ball from the left edge at a random angle in `[-π/4, π/4]`.
    fn launch_ball<R: Rng>(&self, state: &mut PongState, rng: &mut R) {
        let angle = rng.random_range(-std::f32::consts::FRAC_PI_4..std::f32::consts::FRAC_PI_4);
        state.ball_x = 0.0;
        state.ball_y = 0.5;
        state.ball_vx = self.ball_speed * angle.cos();
        state.ball_vy = self.ball_speed * angle.sin();
    }
}

/// Pong as an [`Environment`]: the culture's actuator *is* the paddle. The ball
/// height is what it senses; the reward is dense tracking of the ball by the
/// paddle, and each ball reaching the plane is one hit/miss outcome.
pub struct PongEnv {
    pong: Pong,
    state: PongState,
}

impl PongEnv {
    /// A fresh Pong environment at the given ball speed (seed drives the launch).
    pub fn new(ball_speed: f32, rng: &mut Pcg64) -> Self {
        let pong = Pong {
            ball_speed,
            ..Pong::default()
        };
        let state = pong.reset(rng);
        Self { pong, state }
    }
}

impl Environment for PongEnv {
    fn sensory_position(&self) -> f32 {
        self.state.ball_y
    }

    fn actuator_position(&self) -> f32 {
        self.state.paddle_y
    }

    fn step(&mut self, pos: f32, rng: &mut Pcg64) -> (f32, Event) {
        // Dense tracking reward on the actual paddle position vs. the current ball
        // height (computed before the ball advances), then step with the paddle
        // driven directly to `pos` (the game itself never moves the paddle).
        let reward = 1.0 - 2.0 * (pos - self.state.ball_y).abs();
        self.state.paddle_y = pos;
        let (next, event) = self.pong.step(&self.state, Action::Stay, rng);
        self.state = next;
        (reward, event)
    }

    fn view(&self) -> EnvView<'_> {
        EnvView::Pong(&self.state)
    }

    fn kind(&self) -> GameKind {
        GameKind::Pong
    }

    fn reset(&mut self, rng: &mut Pcg64) {
        self.state = self.pong.reset(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn ball_starts_left_heading_right() {
        let pong = Pong::default();
        let mut rng = Pcg64::seed_from_u64(1);
        let s = pong.reset(&mut rng);
        assert_eq!(s.ball_x, 0.0);
        assert!(s.ball_vx > 0.0, "ball must head right");
    }

    #[test]
    fn centered_paddle_hits_centered_ball() {
        // Ball one step from the plane, dead centre; paddle centred -> hit.
        let pong = Pong::default();
        let mut rng = Pcg64::seed_from_u64(1);
        let state = PongState {
            ball_x: 0.999,
            ball_y: 0.5,
            ball_vx: 0.01,
            ball_vy: 0.0,
            paddle_y: 0.5,
            score_hits: 0,
            score_misses: 0,
            rally_length: 3,
        };
        let (s, ev) = pong.step(&state, Action::Stay, &mut rng);
        assert_eq!(ev, Event::Hit);
        assert_eq!(s.score_hits, 1);
        assert_eq!(s.rally_length, 4);
    }

    #[test]
    fn far_paddle_misses_and_resets_rally() {
        let pong = Pong::default();
        let mut rng = Pcg64::seed_from_u64(1);
        let state = PongState {
            ball_x: 0.999,
            ball_y: 0.5,
            ball_vx: 0.01,
            ball_vy: 0.0,
            paddle_y: 0.0, // paddle at the bottom, ball centred -> miss
            score_hits: 0,
            score_misses: 0,
            rally_length: 5,
        };
        let (s, ev) = pong.step(&state, Action::Stay, &mut rng);
        assert_eq!(ev, Event::Miss);
        assert_eq!(s.score_misses, 1);
        assert_eq!(s.rally_length, 0);
    }
}
