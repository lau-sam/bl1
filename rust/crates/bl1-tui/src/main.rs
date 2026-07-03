//! `bl1` — a lazygit-style terminal UI to configure, run, and inspect BL-1
//! culture simulations.

mod app;
mod ui;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use app::App;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};

#[derive(Parser, Debug)]
#[command(name = "bl1", about = "Terminal UI for BL-1 culture simulations")]
struct Cli {
    /// Directory to scan for `*.yaml` configuration files.
    #[arg(long, value_name = "DIR")]
    configs: Option<PathBuf>,

    /// Run a single preview of the first config and print statistics, then
    /// exit (no interactive UI). Useful for smoke tests.
    #[arg(long)]
    headless: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = cli.configs.clone().or_else(default_configs_dir);
    let mut app = App::new(dir.as_deref());

    if cli.headless {
        app.run_selected();
        print_headless(&app);
        return Ok(());
    }

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;
        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            handle_key(app, key.code);
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('j') | KeyCode::Down => app.select_next(),
        KeyCode::Char('k') | KeyCode::Up => app.select_prev(),
        KeyCode::Char('+') | KeyCode::Char('=') => app.increase_neurons(),
        KeyCode::Char('-') => app.decrease_neurons(),
        KeyCode::Char(']') => app.increase_duration(),
        KeyCode::Char('[') => app.decrease_duration(),
        KeyCode::Char('s') => app.reseed(),
        KeyCode::Enter | KeyCode::Char('r') => app.run_selected(),
        _ => {}
    }
}

/// Look for a `configs/` directory next to the current working directory or its
/// parent, so running from the repo root or `rust/` both work.
fn default_configs_dir() -> Option<PathBuf> {
    for candidate in ["configs", "../configs"] {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return Some(path);
        }
    }
    None
}

fn print_headless(app: &App) {
    println!("{}", app.status);
    if let Some(r) = &app.result {
        println!(
            "  neurons:            {} ({} excitatory)",
            r.n_neurons, r.n_exc
        );
        println!(
            "  window:             {:.0} ms @ dt {:.2} ms",
            r.duration_ms, r.dt_ms
        );
        println!("  mean firing rate:   {:.3} Hz", r.mean_fr_hz);
        println!(
            "  bursts:             {} ({:.2}/min)",
            r.n_bursts, r.burst_rate_per_min
        );
        println!(
            "  IBI mean / CV:      {:.0} ms / {:.2}",
            r.ibi_mean_ms, r.ibi_cv
        );
        println!("  recruitment:        {:.0}%", r.recruitment * 100.0);
        println!("  branching ratio:    {:.3}", r.branching_ratio);
        println!("  avalanche size exp: {:.3}", r.avalanche_size_exp);
    }
}
