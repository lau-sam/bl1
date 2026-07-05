//! `bl1-brain` — the culture as a controller for an external game over stdio.
//!
//! A tiny, dependency-free line protocol so any client (e.g. the ViZDoom bridge
//! in `scripts/vizdoom_bridge.py`) can hand the culture observations and get back
//! actions:
//!
//! ```text
//!   client → brain (one line):  <reward> <obs_0> <obs_1> ... <obs_{n-1}>
//!   brain → client (one line):  <action_0> <action_1> ... <action_{m-1}>
//! ```
//!
//! `reward` scores the *previous* action (0 on the first line). Observations are
//! floats (a coarse retina / feature vector, ideally in `[0, 1]`); actions come
//! back in `[0, 1]`, one per head, for the client to map onto game buttons. The
//! brain learns online by reward-modulated node perturbation on whichever
//! substrate is selected — the same culture that learns Pong and the DOOM arena.

use std::io::{self, BufRead, Write};

use anyhow::Result;
use bl1_games::{BrainParams, CultureReservoir, FeedForwardBank, RemoteBrain, Substrate};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bl1-brain",
    about = "The cortical culture as a stdio controller for an external game (e.g. ViZDoom)"
)]
struct Cli {
    /// Observation length = number of substrate input bands.
    #[arg(long, default_value_t = 32)]
    inputs: usize,

    /// Number of action heads (independent readouts, e.g. turn / move / shoot).
    #[arg(long, default_value_t = 3)]
    actions: usize,

    /// Use the recurrent-culture reservoir substrate instead of the feed-forward
    /// bank.
    #[arg(long)]
    reservoir: bool,

    /// Reservoir size (neurons) when --reservoir is set.
    #[arg(long, default_value_t = 800)]
    neurons: usize,

    /// Feed-forward neurons per input band.
    #[arg(long, default_value_t = 32)]
    per_band: usize,

    /// Random seed (reproducible culture + policy).
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Readout learning rate.
    #[arg(long, default_value_t = 0.05)]
    learning_rate: f32,

    /// Steps over which policy exploration decays from its start to its floor.
    #[arg(long, default_value_t = 20000)]
    explore_decay: usize,

    /// Exploration floor (kept > 0 so the culture keeps refining, not freezing).
    #[arg(long, default_value_t = 0.08)]
    explore_min: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let substrate: Box<dyn Substrate> = if cli.reservoir {
        Box::new(CultureReservoir::new(cli.neurons, cli.inputs, 40, cli.seed))
    } else {
        Box::new(FeedForwardBank::new(cli.inputs, cli.per_band, 40, 0.5))
    };
    let params = BrainParams {
        n_input: cli.inputs,
        n_heads: cli.actions,
        learning_rate: cli.learning_rate,
        explore_decay_steps: cli.explore_decay,
        explore_min: cli.explore_min,
        ..BrainParams::default()
    };
    let mut brain = RemoteBrain::new(params, substrate, cli.seed);

    // Announce readiness on stderr (stdout carries only action lines).
    eprintln!(
        "bl1-brain ready: {} inputs, {} action heads, substrate {}",
        cli.inputs,
        cli.actions,
        if cli.reservoir {
            "recurrent culture"
        } else {
            "feed-forward bank"
        }
    );

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut obs = vec![0.0f32; cli.inputs];

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            break; // graceful shutdown
        }
        let mut it = line.split_whitespace();
        let reward: f32 = match it.next() {
            Some(tok) => tok
                .parse()
                .map_err(|_| anyhow::anyhow!("bad reward token: {tok:?}"))?,
            None => break,
        };
        // Missing observation values are treated as silence — a length mismatch
        // is the client's bug but shouldn't crash the brain mid-game.
        for o in obs.iter_mut() {
            *o = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.0);
        }

        let actions = brain.act(&obs, reward);
        let mut s = String::with_capacity(actions.len() * 8);
        for (i, a) in actions.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{a:.5}"));
        }
        writeln!(out, "{s}")?;
        out.flush()?;
    }
    Ok(())
}
