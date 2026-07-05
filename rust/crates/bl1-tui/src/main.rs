//! `bl1` — a lazygit/k9s-style terminal cockpit to configure, run, and inspect
//! BL-1 culture simulations. Mouse- and keyboard-driven.

mod app;
mod ui;

use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use app::{App, Tab};
use clap::Parser;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton,
    MouseEvent, MouseEventKind,
};
use crossterm::execute;

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
    let _ = execute!(stdout(), EnableMouseCapture);
    let result = run(&mut terminal, &mut app);
    let _ = execute!(stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<()> {
    loop {
        app.poll_run();
        app.train_tick();
        terminal.draw(|frame| ui::draw(frame, app))?;
        // Redraw quickly while a sim runs or training is live (smooth animation).
        let timeout = if app.is_running() || app.training {
            30
        } else {
            200
        };
        if event::poll(Duration::from_millis(timeout))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => handle_key(app, key.code),
                Event::Mouse(m) => handle_mouse(app, m),
                _ => {}
            }
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    // Any key dismisses the help overlay.
    if app.show_help {
        app.show_help = false;
        return;
    }
    // Train tab consumes a few keys for the live training controls.
    if app.active_tab == Tab::Train {
        match code {
            KeyCode::Char(' ') => {
                app.toggle_training();
                return;
            }
            KeyCode::Char('r') => {
                app.reset_trainer();
                return;
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                app.train_faster();
                return;
            }
            KeyCode::Char('-') => {
                app.train_slower();
                return;
            }
            KeyCode::Char('w') => {
                app.save_brain();
                return;
            }
            KeyCode::Char('o') => {
                app.load_brain();
                return;
            }
            KeyCode::Char('m') => {
                app.toggle_control();
                return;
            }
            KeyCode::Char('b') => {
                app.toggle_substrate();
                return;
            }
            KeyCode::Char('g') => {
                app.toggle_game();
                return;
            }
            _ => {}
        }
    }
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('?') => app.toggle_help(),
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Char('1') => app.set_tab(Tab::Dashboard),
        KeyCode::Char('2') => app.set_tab(Tab::Simulate),
        KeyCode::Char('3') => app.set_tab(Tab::Train),
        KeyCode::Char('4') => app.set_tab(Tab::Science),
        KeyCode::Char('5') => app.set_tab(Tab::Results),
        KeyCode::Char('j') | KeyCode::Down => app.browse_next(),
        KeyCode::Char('k') | KeyCode::Up => app.browse_prev(),
        KeyCode::Char('+') | KeyCode::Char('=') => app.increase_neurons(),
        KeyCode::Char('-') => app.decrease_neurons(),
        KeyCode::Char(']') => app.increase_duration(),
        KeyCode::Char('[') => app.decrease_duration(),
        KeyCode::Char('s') => app.reseed(),
        KeyCode::Char('e') => app.export_results(),
        KeyCode::Enter | KeyCode::Char('r') => {
            if app.active_tab == Tab::Dashboard {
                app.set_tab(Tab::Simulate);
            } else {
                app.start_run();
            }
        }
        _ => {}
    }
}

fn handle_mouse(app: &mut App, m: MouseEvent) {
    if app.show_help {
        app.show_help = false;
        return;
    }
    match m.kind {
        MouseEventKind::Down(MouseButton::Left) => app.handle_click(m.column, m.row),
        MouseEventKind::ScrollDown => app.handle_scroll(false, m.column, m.row),
        MouseEventKind::ScrollUp => app.handle_scroll(true, m.column, m.row),
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

#[cfg(test)]
mod render_tests {
    use super::app::{App, Tab};
    use super::ui;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    /// Render the live Train view for a game after a little learning, to smoke
    /// out any panic in the per-game canvas / stats / sensory renderers.
    fn render_train(doom: bool) {
        let mut app = App::new(None);
        app.set_tab(Tab::Train);
        if doom {
            app.toggle_game();
        }
        app.toggle_training(); // builds the trainer and starts it
        for _ in 0..50 {
            app.train_tick();
        }
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|f| ui::draw(f, &mut app)).unwrap();
    }

    #[test]
    fn renders_pong_train_view() {
        render_train(false);
    }

    #[test]
    fn renders_doom_train_view() {
        render_train(true);
    }
}
