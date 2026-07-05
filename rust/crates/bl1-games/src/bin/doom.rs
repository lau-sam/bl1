//! `bl1-doom` — run the closed-loop DOOM aim-and-shoot arena headless and report
//! the learning signal (kill rate over time), with optional CSV export.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use bl1_games::{EnvSpec, Event, Learner, PaddleControl, RunLog, SubstrateSpec};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bl1-doom",
    about = "Closed-loop DOOM aim-and-shoot on a Rust cortical culture"
)]
struct Cli {
    /// Number of game steps to play.
    #[arg(long, default_value_t = 6000)]
    steps: usize,

    /// Random seed (reproducible).
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Number of neurons in the reservoir (with --reservoir).
    #[arg(long, default_value_t = 400)]
    neurons: usize,

    /// Block size (events) for the kill-rate learning curve.
    #[arg(long, default_value_t = 20)]
    block: usize,

    /// Use the recurrent-culture reservoir substrate instead of the feed-forward
    /// bank.
    #[arg(long)]
    reservoir: bool,

    /// Aim through an inertial smooth-pursuit actuator (it lags and overshoots)
    /// instead of snapping the view onto the target.
    #[arg(long)]
    smooth: bool,

    /// Write a per-event CSV to this path.
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let control = if cli.smooth {
        PaddleControl::SmoothPursuit
    } else {
        PaddleControl::Direct
    };
    let substrate = if cli.reservoir {
        SubstrateSpec::Reservoir {
            n_neurons: cli.neurons,
        }
    } else {
        SubstrateSpec::FeedForward { per_band: 32 }
    };
    let sub_label = if cli.reservoir {
        format!("reservoir ({} neurons)", cli.neurons)
    } else {
        "feed-forward bank".to_string()
    };

    println!(
        "Running DOOM aim-and-shoot [{}, {}]: {} game steps, seed {} ...",
        sub_label,
        control.label(),
        cli.steps,
        cli.seed
    );
    let mut agent = Learner::build(EnvSpec::Doom, substrate, control, cli.seed);
    let log: RunLog = agent.run(cli.steps);

    let total = log.hits + log.misses;
    let mean_streak = if log.rally_lengths.is_empty() {
        0.0
    } else {
        log.rally_lengths.iter().sum::<u32>() as f32 / log.rally_lengths.len() as f32
    };
    println!("\nResults");
    println!("  kills / misses:  {} / {}", log.hits, log.misses);
    println!("  kill rate:       {:.1}%", log.hit_rate() * 100.0);
    println!("  encounters:      {}", total);
    println!("  mean streak:     {mean_streak:.2}");
    println!(
        "  improvement:     {:+.0} pts (2nd half − 1st half kill rate)",
        log.improvement() * 100.0
    );
    if total > 0 {
        let curve = log.hit_rate_curve(cli.block);
        println!(
            "\nKill-rate learning curve (blocks of {} events):",
            cli.block
        );
        print_sparkline(&curve);
        if let (Some(&first), Some(&last)) = (curve.first(), curve.last()) {
            println!(
                "  first block {:.0}%  →  last block {:.0}%  (Δ {:+.0} pts)",
                first * 100.0,
                last * 100.0,
                (last - first) * 100.0
            );
        }
    }

    if let Some(path) = &cli.csv {
        write_csv(path, &log)?;
        println!("\nWrote per-event CSV to {}", path.display());
    }

    Ok(())
}

/// Render a 0–1 series as a unicode block sparkline.
fn print_sparkline(series: &[f32]) {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let line: String = series
        .iter()
        .map(|&v| {
            let idx = (v.clamp(0.0, 1.0) * (BARS.len() - 1) as f32).round() as usize;
            BARS[idx]
        })
        .collect();
    println!("  {line}");
}

fn write_csv(path: &PathBuf, log: &RunLog) -> Result<()> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir)?;
    }
    let mut file = fs::File::create(path)?;
    writeln!(file, "game_step,event")?;
    for (step, event) in &log.events {
        let e = match event {
            Event::Hit => "kill",
            Event::Miss => "miss",
            Event::None => "none",
        };
        writeln!(file, "{step},{e}")?;
    }
    Ok(())
}
