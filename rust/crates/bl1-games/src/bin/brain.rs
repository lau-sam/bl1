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

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Result;
use bl1_games::{
    BrainParams, CultureReservoir, FeedForwardBank, RemoteBrain, RemoteBrainState, Substrate,
};
use clap::Parser;

/// Set by the SIGTERM/SIGINT handler so the main loop can save the readout and
/// exit cleanly. The TUI stops a real-DOOM session by SIGTERM-ing the whole
/// process group, so the brain must persist on signal — not only on the bridge's
/// empty-line / EOF shutdown, which the group signal would otherwise pre-empt.
static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_stop_signal(_sig: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

/// Catch SIGTERM/SIGINT so a group-kill doesn't terminate the brain before it
/// saves. The handler only does an atomic store, which is async-signal-safe.
#[cfg(unix)]
fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGTERM, on_stop_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_stop_signal as *const () as libc::sighandler_t);
    }
}

#[cfg(not(unix))]
fn install_signal_handlers() {}

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

    /// Load a saved readout from this file at startup (if it exists and matches).
    #[arg(long, value_name = "PATH")]
    load: Option<PathBuf>,

    /// Save the readout to this file on shutdown (empty line / EOF / SIGTERM).
    #[arg(long, value_name = "PATH")]
    save: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    install_signal_handlers();

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
    let substrate_tag = if cli.reservoir { "reservoir" } else { "feedforward" };

    // Resume from a saved readout if one exists and matches this culture.
    if let Some(path) = &cli.load
        && let Ok(text) = fs::read_to_string(path)
        && let Ok(st) = serde_yaml::from_str::<RemoteBrainState>(&text)
    {
        let compatible =
            st.substrate == substrate_tag && st.n_input == cli.inputs && st.n_heads == cli.actions;
        if compatible && brain.set_readout(st.w, st.b, st.baseline, st.step_idx) {
            eprintln!(
                "bl1-brain: resumed readout from {} (step {})",
                path.display(),
                brain.step_idx()
            );
        } else {
            eprintln!(
                "bl1-brain: {} is incompatible with this culture — starting fresh",
                path.display()
            );
        }
    }

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

    // Read stdin on a worker thread so the main loop can poll the STOP flag: a
    // blocking `stdin.lines()` would swallow SIGTERM (the read auto-retries on
    // EINTR) and never observe the signal until the client sends more data.
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break; // main loop gone
                    }
                }
                Err(_) => break, // read error / EOF
            }
        }
        // Dropping `tx` here disconnects `rx`, waking the main loop for a final save.
    });

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut obs = vec![0.0f32; cli.inputs];

    loop {
        if STOP.load(Ordering::Relaxed) {
            break; // SIGTERM/SIGINT: fall through to the final save.
        }
        let line = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(l) => l,
            Err(mpsc::RecvTimeoutError::Timeout) => continue, // idle: re-check STOP
            Err(mpsc::RecvTimeoutError::Disconnected) => break, // stdin closed / EOF
        };
        let line = line.trim();
        if line.is_empty() {
            break; // graceful shutdown
        }
        let mut it = line.split_whitespace();
        let reward: f32 = match it.next().and_then(|tok| tok.parse().ok()) {
            Some(r) => r,
            None => break, // malformed line: stop and save what we have
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

        // Persist periodically so a hard kill (e.g. the TUI stopping the session)
        // loses at most a little progress, not the whole run.
        if cli.save.is_some() && brain.step_idx().is_multiple_of(1000) {
            write_state(&cli, &brain, substrate_tag, false);
        }
    }

    // Final save on any shutdown path: empty line, EOF, or SIGTERM/SIGINT.
    write_state(&cli, &brain, substrate_tag, true);
    Ok(())
}

/// Save the brain's readout to `--save` (no-op if unset), so the next session can
/// resume this culture. `announce` logs the save on stderr.
fn write_state(cli: &Cli, brain: &RemoteBrain, substrate_tag: &str, announce: bool) {
    let Some(path) = &cli.save else {
        return;
    };
    let (w, b, baseline, step_idx) = brain.readout();
    let state = RemoteBrainState {
        version: 1,
        substrate: substrate_tag.to_string(),
        n_input: cli.inputs,
        n_heads: cli.actions,
        per_band: if cli.reservoir { 0 } else { cli.per_band },
        n_neurons: if cli.reservoir { cli.neurons } else { 0 },
        seed: cli.seed,
        w,
        b,
        baseline,
        step_idx,
    };
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        let _ = fs::create_dir_all(dir);
    }
    if let Ok(yaml) = serde_yaml::to_string(&state)
        && fs::write(path, yaml).is_ok()
        && announce
    {
        eprintln!("bl1-brain: saved readout to {} (step {step_idx})", path.display());
    }
}
