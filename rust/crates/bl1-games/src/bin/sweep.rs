//! `bl1-pong-sweep` — grid-search closed-loop Pong parameters, scoring each
//! config by its learning improvement **averaged over several seeds**.
//!
//! Single runs are noisy (few events), so eyeballing one curve is unreliable.
//! This averages the second-half-minus-first-half hit-rate improvement across
//! seeds and ranks configs by that robust score.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use bl1_games::{ClosedLoop, LoopConfig, Reward, ThreeFactorParams};
use bl1_sim::Config;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "bl1-pong-sweep", about = "Multi-seed parameter sweep for closed-loop Pong")]
struct Cli {
    #[arg(long, default_value_t = 300)]
    neurons: usize,
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    #[arg(long, default_value_t = 3)]
    seeds: u64,
    #[arg(long, value_name = "PATH", default_value = "results/pong_sweep.csv")]
    csv: PathBuf,
}

/// One grid point's averaged score.
struct Score {
    learning_rate: f32,
    tau_elig: f32,
    sensory_amplitude: f32,
    mean_improvement: f32,
    std_improvement: f32,
    mean_hit_rate: f32,
}

fn evaluate(
    config: &Config,
    loopcfg: &LoopConfig,
    seeds: u64,
    steps: usize,
) -> (f32, f32, f32) {
    let mut improvements = Vec::new();
    let mut hit_rates = Vec::new();
    for seed in 0..seeds {
        let mut game = ClosedLoop::new(config, loopcfg, seed + 1);
        let log = game.run(steps);
        improvements.push(log.improvement());
        hit_rates.push(log.hit_rate());
    }
    let mean_impr = mean(&improvements);
    let std_impr = std(&improvements, mean_impr);
    (mean_impr, std_impr, mean(&hit_rates))
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

fn std(xs: &[f32], m: f32) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32).sqrt()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let yaml = format!(
        "culture:\n  n_neurons: {}\n  substrate_um: [1400, 1400]\n  p_max: 0.25\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n",
        cli.neurons
    );
    let config = Config::from_yaml_str(&yaml)?;

    let learning_rates = [0.004f32, 0.008, 0.016];
    let tau_eligs = [500.0f32, 1000.0, 2000.0];
    let amplitudes = [8.0f32, 12.0];
    let total = learning_rates.len() * tau_eligs.len() * amplitudes.len();

    println!(
        "Sweep: {} configs × {} seeds, {} neurons, {} steps\n",
        total, cli.seeds, cli.neurons, cli.steps
    );

    let mut scores = Vec::new();
    let mut done = 0;
    for &lr in &learning_rates {
        for &tau in &tau_eligs {
            for &amp in &amplitudes {
                let loopcfg = LoopConfig {
                    sensory_amplitude: amp,
                    plasticity: ThreeFactorParams {
                        learning_rate: lr,
                        tau_elig: tau,
                        ..ThreeFactorParams::default()
                    },
                    reward: Reward::default(),
                    ..LoopConfig::default()
                };
                let (mi, si, hr) = evaluate(&config, &loopcfg, cli.seeds, cli.steps);
                scores.push(Score {
                    learning_rate: lr,
                    tau_elig: tau,
                    sensory_amplitude: amp,
                    mean_improvement: mi,
                    std_improvement: si,
                    mean_hit_rate: hr,
                });
                done += 1;
                println!(
                    "  [{done}/{total}] lr={lr} tau_elig={tau} amp={amp}  →  Δ {:+.1}±{:.1} pts, hit {:.0}%",
                    mi * 100.0,
                    si * 100.0,
                    hr * 100.0
                );
            }
        }
    }

    // Rank by mean improvement, then by mean hit rate.
    scores.sort_by(|a, b| {
        b.mean_improvement
            .partial_cmp(&a.mean_improvement)
            .unwrap()
            .then(b.mean_hit_rate.partial_cmp(&a.mean_hit_rate).unwrap())
    });

    println!("\nTop configs (by mean learning improvement):");
    for s in scores.iter().take(5) {
        println!(
            "  Δ {:+.1}±{:.1} pts | hit {:.0}% | lr={} tau_elig={} amp={}",
            s.mean_improvement * 100.0,
            s.std_improvement * 100.0,
            s.mean_hit_rate * 100.0,
            s.learning_rate,
            s.tau_elig,
            s.sensory_amplitude
        );
    }

    write_csv(&cli.csv, &scores)?;
    println!("\nWrote full grid to {}", cli.csv.display());
    Ok(())
}

fn write_csv(path: &PathBuf, scores: &[Score]) -> Result<()> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir)?;
    }
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "learning_rate,tau_elig,sensory_amplitude,mean_improvement,std_improvement,mean_hit_rate"
    )?;
    for s in scores {
        writeln!(
            f,
            "{},{},{},{:.4},{:.4},{:.4}",
            s.learning_rate,
            s.tau_elig,
            s.sensory_amplitude,
            s.mean_improvement,
            s.std_improvement,
            s.mean_hit_rate
        )?;
    }
    Ok(())
}
