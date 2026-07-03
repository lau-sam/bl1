//! `bl1-pong` — run the closed-loop Pong experiment headless and report the
//! learning signal (rally length + hit rate over time), with optional CSV
//! export for plotting.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use bl1_games::{AgentParams, ClosedLoop, Event, LoopConfig, RstdpAgent, RunLog};
use bl1_sim::Config;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bl1-pong",
    about = "Closed-loop Pong on a Rust cortical culture"
)]
struct Cli {
    /// Number of neurons in the culture.
    #[arg(long, default_value_t = 400)]
    neurons: usize,

    /// Number of game steps to play.
    #[arg(long, default_value_t = 2000)]
    steps: usize,

    /// Random seed (reproducible).
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Neural window per game step (ms).
    #[arg(long, default_value_t = 50.0)]
    game_step_ms: f32,

    /// Block size (events) for the hit-rate learning curve.
    #[arg(long, default_value_t = 20)]
    block: usize,

    /// Use the R-STDP feed-forward learning agent (Wunderlich-style) instead of
    /// the recurrent-culture reflex loop.
    #[arg(long)]
    rstdp: bool,

    /// Write a per-event CSV to this path.
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log: RunLog = if cli.rstdp {
        println!(
            "Running R-STDP Pong agent: {} game steps, seed {} ...",
            cli.steps, cli.seed
        );
        let mut agent = RstdpAgent::new(AgentParams::default(), cli.seed);
        agent.run(cli.steps)
    } else {
        let yaml = format!(
            "culture:\n  n_neurons: {}\n  substrate_um: [1400, 1400]\n  p_max: 0.25\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n",
            cli.neurons
        );
        let config = Config::from_yaml_str(&yaml)?;
        let cfg = LoopConfig {
            game_step_ms: cli.game_step_ms,
            ..LoopConfig::default()
        };
        println!(
            "Running closed-loop Pong: {} neurons, {} game steps, seed {} ...",
            cli.neurons, cli.steps, cli.seed
        );
        let mut game = ClosedLoop::new(&config, &cfg, cli.seed);
        game.run(cli.steps)
    };

    let total = log.hits + log.misses;
    let mean_rally = if log.rally_lengths.is_empty() {
        0.0
    } else {
        log.rally_lengths.iter().sum::<u32>() as f32 / log.rally_lengths.len() as f32
    };
    println!("\nResults");
    println!("  hits / misses:   {} / {}", log.hits, log.misses);
    println!("  hit rate:        {:.1}%", log.hit_rate() * 100.0);
    println!("  rallies played:  {}", log.rally_lengths.len());
    println!("  mean rally:      {mean_rally:.2}");
    println!(
        "  improvement:     {:+.0} pts (2nd half − 1st half hit rate)",
        log.improvement() * 100.0
    );
    if total > 0 {
        let curve = log.hit_rate_curve(cli.block);
        println!(
            "\nHit-rate learning curve (blocks of {} events):",
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

fn write_csv(path: &PathBuf, log: &bl1_games::RunLog) -> Result<()> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir)?;
    }
    let mut file = fs::File::create(path)?;
    writeln!(file, "game_step,event")?;
    for (step, event) in &log.events {
        let e = match event {
            Event::Hit => "hit",
            Event::Miss => "miss",
            Event::None => "none",
        };
        writeln!(file, "{step},{e}")?;
    }
    Ok(())
}
