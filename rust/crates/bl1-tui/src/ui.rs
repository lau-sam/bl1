//! Rendering: a lazygit/k9s-style, mouse-friendly cockpit.
//!
//! Every frame refreshes `app.regions` with the on-screen rectangles of the
//! tab bar, panels, and buttons so the event loop can hit-test mouse clicks.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Points};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem, ListState,
    Paragraph, Sparkline, Wrap,
};

use bl1_games::{DoomState, EnvView, PongState};

use crate::app::{
    App, DOOM_SCENARIOS, DoomSession, Focus, GameChoice, MenuField, RunResult, Tab, TrainScreen,
    substrate_label,
};

const CYAN: Color = Color::Cyan;
const BG_BAR: Color = Color::Rgb(30, 30, 40);

/// Draw the whole UI for the current frame.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // tab bar
            Constraint::Min(0),    // body
            Constraint::Length(1), // status line
            Constraint::Length(1), // keybar
        ])
        .split(frame.area());

    draw_tabs(frame, app, root[0]);

    match app.active_tab {
        Tab::Dashboard => draw_dashboard(frame, app, root[1]),
        Tab::Simulate => draw_simulate(frame, app, root[1]),
        Tab::Train => draw_train(frame, app, root[1]),
        Tab::Science => draw_science(frame, app, root[1]),
        Tab::Results => draw_results(frame, app, root[1]),
    }

    draw_status(frame, app, root[2]);
    draw_keybar(frame, app, root[3]);

    if app.show_help {
        draw_help(frame, app);
    }
}

/// Braille spinner frame for the given elapsed time.
fn spinner(elapsed_ms: u128) -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[((elapsed_ms / 100) as usize) % FRAMES.len()]
}

/// The always-visible status line: latest message, prefixed with a spinner
/// while a background simulation is running.
fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = Vec::new();
    if app.is_running() {
        spans.push(Span::styled(
            format!(" {} ", spinner(app.run_elapsed_ms())),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        app.status.clone(),
        Style::default().fg(Color::Gray),
    ));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

fn draw_tabs(frame: &mut Frame, app: &mut App, area: Rect) {
    let mut spans = vec![Span::styled(
        " bl1 ",
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
    )];
    let mut x = area.x + 5;
    let mut rects = Vec::with_capacity(Tab::ALL.len());

    for tab in Tab::ALL {
        spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
        x += 1;
        let label = format!(" {} ", tab.title());
        let w = label.chars().count() as u16;
        let style = if tab == app.active_tab {
            Style::default()
                .fg(Color::Black)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        rects.push(Rect {
            x,
            y: area.y,
            width: w,
            height: 1,
        });
        spans.push(Span::styled(label, style));
        x += w;
    }
    app.regions.tabs = rects;

    let p = Paragraph::new(Line::from(spans)).style(Style::default().bg(BG_BAR));
    frame.render_widget(p, area);
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

fn draw_dashboard(frame: &mut Frame, app: &App, area: Rect) {
    let dim = Style::default().fg(Color::DarkGray);
    let head = Style::default().fg(CYAN).add_modifier(Modifier::BOLD);
    let val = Style::default().add_modifier(Modifier::BOLD);

    let last = match &app.result {
        Some(r) => format!(
            "{} neurons, {:.2} Hz, {} bursts",
            r.n_neurons, r.mean_fr_hz, r.n_bursts
        ),
        None => "— none yet —".to_string(),
    };

    let mut lines = vec![
        Line::from(Span::styled(
            "BL-1 — in-silico cortical culture simulator",
            head,
        )),
        Line::from(Span::styled(
            "Native spiking-network forward simulation on your machine.",
            dim,
        )),
        Line::from(""),
        Line::from(Span::styled("Getting started", head)),
        Line::from("  1. Open the Simulate tab (press 2, Tab, or click it)."),
        Line::from("  2. Pick a config in the left list (click or j / k)."),
        Line::from("  3. Tune neurons / duration with the [-] [+] buttons."),
        Line::from("  4. Press Enter (or click Run) — watch the raster + stats."),
        Line::from("  5. Review past runs in the Results tab."),
        Line::from(vec![
            Span::raw("  6. Open "),
            Span::styled(
                "Train",
                Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" (press 3) and hit Space — watch the culture learn Pong live."),
        ]),
        Line::from(""),
        Line::from(Span::styled("System status", head)),
        kv("configs loaded", format!("{}", app.configs.len()), val),
        kv(
            "  from disk / presets",
            format!(
                "{} / {}",
                app.configs_from_dir,
                app.configs.len() - app.configs_from_dir
            ),
            val,
        ),
        kv("runs this session", format!("{}", app.history.len()), val),
        kv("last run", last, val),
        Line::from(""),
        Line::from(Span::styled(
            "Press ? for the full list of actions in any view.",
            dim,
        )),
    ];
    // A tiny bit of breathing room at the top.
    lines.insert(0, Line::from(""));

    let block = Block::default().borders(Borders::ALL).title(" Dashboard ");
    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

fn kv(label: &str, value: String, val_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {label:<22}"), Style::default().fg(Color::Gray)),
        Span::styled(value, val_style),
    ])
}

// ---------------------------------------------------------------------------
// Simulate view
// ---------------------------------------------------------------------------

fn draw_simulate(frame: &mut Frame, app: &mut App, area: Rect) {
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(0)])
        .split(area);

    draw_sidebar(frame, app, body[0]);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Min(0),
            Constraint::Length(10),
        ])
        .split(body[1]);

    draw_params(frame, app, right[0]);
    draw_raster(frame, app, right[1]);
    draw_stats(frame, app, right[2]);
}

fn panel(title: &str, focused: bool) -> Block<'_> {
    let border = if focused {
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border)
}

fn draw_sidebar(frame: &mut Frame, app: &mut App, area: Rect) {
    app.regions.configs = area;
    let items: Vec<ListItem> = app
        .configs
        .iter()
        .map(|c| ListItem::new(c.name.clone()))
        .collect();
    let list = List::new(items)
        .block(panel(" Configs ", app.focus == Focus::Configs))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    let mut state = ListState::default();
    state.select(Some(app.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_params(frame: &mut Frame, app: &mut App, area: Rect) {
    app.regions.params = area;
    let block = panel(" Parameters ", app.focus == Focus::Params);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let x0 = inner.x;
    let y0 = inner.y;
    let btn = |line: u16, col: u16, w: u16| Rect {
        x: x0 + col,
        y: y0 + line,
        width: w,
        height: 1,
    };
    app.regions.btn_neuron_dec = btn(0, 10, 3);
    app.regions.btn_neuron_inc = btn(0, 22, 3);
    app.regions.btn_dur_dec = btn(1, 10, 3);
    app.regions.btn_dur_inc = btn(1, 22, 3);
    app.regions.btn_reseed = btn(2, 22, 8);
    let run_label = "[ Run  (Enter) ]";
    app.regions.btn_run = btn(4, 0, run_label.chars().count() as u16);

    let lines = vec![
        param_line("neurons", &format!("{}", app.neuron_cap), true),
        param_line("duration", &format!("{:.0} ms", app.preview_ms), true),
        Line::from(vec![
            gray_label("seed"),
            Span::raw("    "),
            value_span(&format!("{:<8}", app.seed)),
            button_span("[reseed]"),
        ]),
        Line::from(""),
        Line::from(button_span(run_label)),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// A `label  [-] value [+]` control row, aligned to the button hit-rects.
fn param_line(label: &str, value: &str, _adjustable: bool) -> Line<'static> {
    Line::from(vec![
        gray_label(label),
        button_span("[-]"),
        Span::raw(" "),
        value_span(&format!("{value:<8}")),
        button_span("[+]"),
    ])
}

fn gray_label(s: &str) -> Span<'static> {
    Span::styled(format!("{s:<10}"), Style::default().fg(Color::Gray))
}

fn value_span(s: &str) -> Span<'static> {
    Span::styled(
        s.to_string(),
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    )
}

fn button_span(s: &str) -> Span<'static> {
    Span::styled(
        s.to_string(),
        Style::default()
            .fg(Color::Black)
            .bg(CYAN)
            .add_modifier(Modifier::BOLD),
    )
}

fn draw_raster(frame: &mut Frame, app: &mut App, area: Rect) {
    app.regions.raster = area;
    let block = panel(
        " Raster (green = excitatory, red = inhibitory) ",
        app.focus == Focus::Raster,
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.is_running() {
        let secs = app.run_elapsed_ms() as f64 / 1000.0;
        let banner = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    format!("  {}  ", spinner(app.run_elapsed_ms())),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("Simulating…  {secs:.1}s"),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(banner), inner);
        return;
    }

    let (lines, total_rows) = match &app.result {
        Some(res) => {
            let rows = res.raster.n_neurons.clamp(1, 120);
            (
                render_raster_lines(res, inner.width as usize, rows),
                rows as u16,
            )
        }
        None => (
            vec![Line::from(Span::styled(
                "No run yet — press Enter or click Run.",
                Style::default().fg(Color::DarkGray),
            ))],
            1,
        ),
    };

    // Clamp scroll so it can never run past the content.
    let max_scroll = total_rows.saturating_sub(inner.height);
    if app.raster_scroll > max_scroll {
        app.raster_scroll = max_scroll;
    }
    frame.render_widget(Paragraph::new(lines).scroll((app.raster_scroll, 0)), inner);
}

/// Turn a raster into colored block-character rows sized to `width × rows`.
fn render_raster_lines(res: &RunResult, width: usize, rows: usize) -> Vec<Line<'static>> {
    let raster = &res.raster;
    let (n_steps, n_neurons) = (raster.n_steps, raster.n_neurons);
    if width == 0 || rows == 0 || n_steps == 0 || n_neurons == 0 {
        return vec![Line::from("")];
    }
    let cols = width.min(n_steps).max(1);

    let mut sums = vec![0f32; rows * cols];
    let mut counts = vec![0u32; rows * cols];
    // Subsample: a 5000-neuron × 8000-step raster is 40M cells — iterating all
    // of them every redraw makes the UI lag. Stride so the work is bounded to a
    // few times the display resolution (the panel is a downsampled preview).
    let t_stride = (n_steps / (cols * 4).max(1)).max(1);
    let n_stride = (n_neurons / (rows * 4).max(1)).max(1);
    let mut t = 0;
    while t < n_steps {
        let c = (t * cols / n_steps).min(cols - 1);
        let row_t = raster.row(t);
        let mut j = 0;
        while j < n_neurons {
            let r = (j * rows / n_neurons).min(rows - 1);
            sums[r * cols + c] += row_t[j];
            counts[r * cols + c] += 1;
            j += n_stride;
        }
        t += t_stride;
    }

    const SHADES: [char; 6] = [' ', '·', '░', '▒', '▓', '█'];
    let mut lines = Vec::with_capacity(rows);
    for r in 0..rows {
        let band_start = r * n_neurons / rows;
        let color = if band_start < res.n_exc_rows {
            Color::Green
        } else {
            Color::Red
        };
        let mut s = String::with_capacity(cols);
        for c in 0..cols {
            let idx = r * cols + c;
            let density = if counts[idx] > 0 {
                sums[idx] / counts[idx] as f32
            } else {
                0.0
            };
            let level = ((density * 40.0).sqrt() * (SHADES.len() - 1) as f32)
                .round()
                .clamp(0.0, (SHADES.len() - 1) as f32) as usize;
            s.push(SHADES[level]);
        }
        lines.push(Line::from(Span::styled(s, Style::default().fg(color))));
    }
    lines
}

fn draw_stats(frame: &mut Frame, app: &mut App, area: Rect) {
    app.regions.stats = area;
    let block = panel(" Statistics ", app.focus == Focus::Stats);
    let lines: Vec<Line> = match &app.result {
        Some(r) => vec![
            stat("neurons", format!("{} ({} exc)", r.n_neurons, r.n_exc)),
            stat(
                "window",
                format!("{:.0} ms @ dt {:.2} ms", r.duration_ms, r.dt_ms),
            ),
            stat("mean firing rate", format!("{:.2} Hz", r.mean_fr_hz)),
            stat(
                "bursts",
                format!("{} ({:.1}/min)", r.n_bursts, r.burst_rate_per_min),
            ),
            stat("IBI mean / CV", fmt_ibi(r.ibi_mean_ms, r.ibi_cv)),
            stat("recruitment", fmt_frac(r.recruitment)),
            stat("branching ratio σ", fmt_f64(r.branching_ratio)),
            stat("avalanche size exp", fmt_f64(r.avalanche_size_exp)),
        ],
        None => vec![Line::from(Span::styled(
            "Run a preview to see statistics.",
            Style::default().fg(Color::DarkGray),
        ))],
    };
    let p = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    frame.render_widget(p, area);
}

fn stat(label: &str, value: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<20}"), Style::default().fg(Color::Gray)),
        Span::styled(value, Style::default().add_modifier(Modifier::BOLD)),
    ])
}

// ---------------------------------------------------------------------------
// Results view
// ---------------------------------------------------------------------------

fn draw_results(frame: &mut Frame, app: &mut App, area: Rect) {
    if app.history.is_empty() {
        app.regions.results = area;
        let block = panel(" Results — this session's runs ", true);
        let p = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No runs yet.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  Go to Simulate (press 2), pick a config, and press Run.",
                Style::default().fg(Color::DarkGray),
            )),
        ])
        .block(block);
        frame.render_widget(p, area);
        return;
    }

    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(7)])
        .split(area);
    app.regions.results = split[0];
    draw_results_detail(frame, app, split[1]);

    let block = panel(" Results — this session's runs ", true);
    let header = format!(
        "  {:<22} {:>7} {:>9} {:>7} {:>7}",
        "config", "neurons", "rate Hz", "bursts", "σ"
    );
    let items: Vec<ListItem> = std::iter::once(ListItem::new(Line::from(Span::styled(
        header,
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
    ))))
    .chain(app.history.iter().map(|h| {
        ListItem::new(format!(
            "  {:<22} {:>7} {:>9.2} {:>7} {:>7.3}",
            truncate(&h.config_name, 22),
            h.n_neurons,
            h.mean_fr_hz,
            h.n_bursts,
            h.branching_ratio,
        ))
    }))
    .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    let mut state = ListState::default();
    // +1 for the header row, which is never selectable.
    state.select(Some(app.results_selected + 1));
    frame.render_stateful_widget(list, app.regions.results, &mut state);
}

/// Full parameters + metrics of the currently-selected run.
fn draw_results_detail(frame: &mut Frame, app: &App, area: Rect) {
    let block = panel(" Selected run ", false);
    let lines = match app.history.get(app.results_selected) {
        Some(h) => vec![
            Line::from(vec![
                stat_inline("config", h.config_name.clone()),
                Span::raw("   "),
                stat_inline("seed", format!("{}", h.seed)),
            ]),
            Line::from(vec![
                stat_inline("neuron cap", format!("{}", h.neuron_cap)),
                Span::raw("   "),
                stat_inline("window", format!("{:.0} ms", h.preview_ms)),
            ]),
            Line::from(vec![
                stat_inline("firing rate", format!("{:.2} Hz", h.mean_fr_hz)),
                Span::raw("   "),
                stat_inline(
                    "bursts",
                    format!("{} ({:.1}/min)", h.n_bursts, h.burst_rate_per_min),
                ),
                Span::raw("   "),
                stat_inline("σ", format!("{:.3}", h.branching_ratio)),
            ]),
        ],
        None => vec![Line::from(Span::styled(
            "  Select a run above.",
            Style::default().fg(Color::DarkGray),
        ))],
    };
    let p = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    frame.render_widget(p, area);
}

fn stat_inline(label: &str, value: String) -> Span<'static> {
    Span::styled(
        format!("{label}: {value}"),
        Style::default().add_modifier(Modifier::BOLD),
    )
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

// ---------------------------------------------------------------------------
// Keybar + help overlay
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Train view (live learning)
// ---------------------------------------------------------------------------

fn draw_train(frame: &mut Frame, app: &App, area: Rect) {
    // The Train tab is a menu → play state machine.
    if app.train_screen == TrainScreen::Menu {
        draw_train_menu(frame, app, area);
        return;
    }

    // Real DOOM plays in its own window; the cockpit shows the session monitor.
    if app.game_choice == GameChoice::DoomReal {
        draw_doom_playing(frame, app, area);
        return;
    }

    // A TUI game (Pong / Doom arena) plays live inside the cockpit.
    let Some(trainer) = app.trainer.as_deref() else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  starting…",
                Style::default().fg(Color::DarkGray),
            )))
            .block(Block::default().borders(Borders::ALL).title(" Train ")),
            area,
        );
        return;
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(3), // per-event hit/miss timeline
            Constraint::Length(9),
        ])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(outer[0]);
    draw_game_canvas(frame, trainer, top[0], app.training);
    draw_learning_chart(frame, trainer, top[1]);

    draw_outcomes(frame, trainer, outer[1]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(36),
            Constraint::Percentage(30),
            Constraint::Percentage(34),
        ])
        .split(outer[2]);
    draw_train_gauges(frame, trainer, bottom[0]);
    draw_sensory(frame, trainer, bottom[1]);
    draw_train_stats(frame, trainer, app, bottom[2]);
}

/// The Train menu: pick what the culture plays before entering the game.
fn draw_train_menu(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Train — choose a mode ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let control_val = if app.game_choice.is_tui() {
        app.train_control.label()
    } else {
        "— (real Doom aims via ViZDoom)"
    };
    let scenario_val = if app.game_choice == GameChoice::DoomReal {
        DOOM_SCENARIOS[app.doom_scenario]
    } else {
        "— (real Doom only)"
    };
    let substrate_val = substrate_label(app.train_substrate);
    let seed_val = app.train_seed.to_string();

    // Each row carries a short inline hint (shown greyed after the value) so every
    // option is self-describing at a glance.
    let rows: [(MenuField, &str, String, bool, &str); 5] = [
        (MenuField::Game, "Game", app.game_choice.label().to_string(), true, ""),
        (
            MenuField::Substrate,
            "Substrate",
            substrate_val.to_string(),
            true,
            "sharp bank vs. the real recurrent culture",
        ),
        (
            MenuField::Control,
            "Control",
            control_val.to_string(),
            app.game_choice.is_tui(),
            "snap vs. inertial aim",
        ),
        (
            MenuField::Scenario,
            "Scenario",
            scenario_val.to_string(),
            app.game_choice == GameChoice::DoomReal,
            "ViZDoom map",
        ),
        (
            MenuField::Seed,
            "Seed",
            seed_val,
            true,
            "reproducible run: game + culture wiring + noise",
        ),
    ];

    let mut lines = vec![
        Line::from(Span::styled(
            "  What should the culture play?",
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "  ↑/↓ field · ←/→ change · Enter start",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    for (i, (_f, label, value, active, hint)) in rows.iter().enumerate() {
        let selected = i == app.menu_field;
        let marker = if selected { "› " } else { "  " };
        let label_style = if selected {
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        // Selected value: bold in the terminal's default fg (never fg(White) —
        // it vanishes on light backgrounds).
        let value_style = if !active {
            Style::default().fg(Color::DarkGray)
        } else if selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let mut spans = vec![
            Span::styled(format!("  {marker}"), label_style),
            Span::styled(format!("{label:<11}"), label_style),
            Span::styled(value.clone(), value_style),
        ];
        // Inline hint, only where the field is active (inactive rows already show
        // a self-explanatory "— (…)" placeholder as their value).
        if *active && !hint.is_empty() {
            spans.push(Span::styled(
                format!("  — {hint}"),
                Style::default().fg(Color::DarkGray),
            ));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(""));
    let blurb = match app.game_choice {
        GameChoice::PongTui => "Pong: the culture tracks the ball with a paddle, rendered here.",
        GameChoice::DoomTui => "Doom arena: aim-and-shoot toy rendered here in the terminal.",
        GameChoice::DoomReal => {
            "Real Doom: opens a ViZDoom window; the culture aims and shoots. Needs `pip install vizdoom numpy`."
        }
    };
    lines.push(Line::from(Span::styled(
        format!("  {blurb}"),
        Style::default().fg(Color::DarkGray),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}

/// The Playing view for real DOOM: the game is in its own window, so the cockpit
/// shows the session monitor + how to control it.
fn draw_doom_playing(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);
    match &app.doom {
        Some(session) => draw_doom_monitor(frame, session, rows[0]),
        None => frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  real DOOM session ended.",
                Style::default().fg(Color::DarkGray),
            )))
            .block(Block::default().borders(Borders::ALL).title(" real DOOM ")),
            rows[0],
        ),
    }
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  Doom is in its own window · auto-saves to brains/doom_real_*.yaml · r relaunch · Esc menu",
            Style::default().fg(Color::DarkGray),
        )))
        .block(Block::default().borders(Borders::ALL)),
        rows[1],
    );
}

/// The real-DOOM session monitor: the live learning signal of the *separate*
/// Doom brain (kills per episode), tailed from the bridge log.
fn draw_doom_monitor(frame: &mut Frame, s: &DoomSession, area: Rect) {
    let state = if s.finished { "finished" } else { "running" };
    let scolor = if s.finished { Color::DarkGray } else { Color::Green };
    let title = format!(
        " real DOOM ▸ {} · {} · pid {} [{}] ",
        s.scenario,
        substrate_label(s.substrate),
        s.pid,
        state
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(scolor))
        .title(Span::styled(
            title,
            Style::default().fg(scolor).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // stat lines
            Constraint::Min(0),    // kills sparkline
            Constraint::Length(1), // note
        ])
        .split(inner);

    let total_kills: u32 = s.kills.iter().sum();
    let best_kills = s.kills.iter().copied().max().unwrap_or(0);
    // Show the per-episode ceiling (starting ammo) once the bridge reports it.
    let best_str = if s.ammo_cap > 0 {
        format!("best {best_kills}/{} ammo", s.ammo_cap)
    } else {
        format!("best {best_kills}")
    };
    let accuracy = if s.shots > 0 {
        100.0 * total_kills as f64 / s.shots as f64
    } else {
        0.0
    };
    let stats = if s.kills.is_empty() {
        vec![Line::from(Span::styled(
            "waiting for the first episode…  (Doom opens in its own window)",
            Style::default().fg(Color::Gray),
        ))]
    } else {
        vec![
            Line::from(Span::styled(
                format!(
                    "episode {}   {} kills total · mean {:.2}/ep · {} · last {} · accuracy {:.0}%",
                    s.kills.len(),
                    total_kills,
                    s.mean_kills(),
                    best_str,
                    s.kills.last().copied().unwrap_or(0),
                    accuracy,
                ),
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                format!(
                    "this session: {} frames · {} shots   ·   lifetime: {} frames (across sessions)",
                    s.frames, s.shots, s.lifetime_frames,
                ),
                Style::default().fg(Color::DarkGray),
            )),
        ]
    };
    frame.render_widget(Paragraph::new(stats), rows[0]);

    if s.kills.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "the culture is learning to aim and shoot…",
                Style::default().fg(Color::DarkGray),
            ))),
            rows[1],
        );
    } else {
        // Show the most recent episodes that fit: a Sparkline renders leading
        // data, so without this the curve freezes once episodes exceed the width.
        let w = rows[1].width as usize;
        let start = s.kills.len().saturating_sub(w);
        let data: Vec<u64> = s.kills[start..].iter().map(|&k| k as u64).collect();
        frame.render_widget(
            Sparkline::default()
                .data(&data)
                .max((best_kills as u64).max(1)) // pin the scale to `best` (≥1, avoids /0)
                .style(Style::default().fg(Color::Red)),
            rows[1],
        );
    }
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "separate process — Esc back to the menu to change scenario / substrate / seed",
            Style::default().fg(Color::DarkGray),
        ))),
        rows[2],
    );
}

/// A per-event hit/miss timeline: one coloured block per ball, oldest→newest,
/// so the early red (misses) visibly turns green (hits) as the culture learns.
fn draw_outcomes(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, area: Rect) {
    let block = panel(
        " Outcomes — every ball, oldest → newest (green hit · red miss) ",
        false,
    );
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let w = inner.width.max(1) as usize;
    let outs = trainer.recent_outcomes(w);
    if outs.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no balls yet…",
                Style::default().fg(Color::DarkGray),
            ))),
            inner,
        );
        return;
    }
    let spans: Vec<Span> = outs
        .iter()
        .map(|&hit| {
            Span::styled(
                "█",
                Style::default().fg(if hit { Color::Green } else { Color::Red }),
            )
        })
        .collect();
    frame.render_widget(Paragraph::new(Line::from(spans)), inner);
}

/// Dispatch to the right scene renderer for whichever game the culture plays.
fn draw_game_canvas(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, area: Rect, playing: bool) {
    match trainer.view() {
        EnvView::Pong(s) => draw_pong_canvas(frame, s, area, playing),
        EnvView::Doom(s) => draw_doom_canvas(frame, s, area, playing),
    }
}

fn draw_pong_canvas(frame: &mut Frame, g: &PongState, area: Rect, playing: bool) {
    let (bx, by) = (g.ball_x as f64, g.ball_y as f64);
    let py = g.paddle_y as f64;

    // Classic Pong: a dashed centre net, a solid square ball, a solid paddle.
    // Everything is drawn as dense point grids so shapes render *filled* (not
    // outlines) at cell resolution — the retro block look.
    let mut net: Vec<(f64, f64)> = Vec::new();
    let mut k = 0.0;
    while k < 1.0 {
        net.push((0.5, k));
        k += 0.05; // dashed: one dot every 0.05
    }

    // Ball: filled square (~square on screen given ~2:1 cells → x narrower).
    // Dense sampling so every covered cell fills solid (no gaps).
    let (bw, bh) = (0.022, 0.04);
    let mut ball: Vec<(f64, f64)> = Vec::new();
    let mut ax = bx - bw;
    while ax <= bx + bw {
        let mut ay = by - bh;
        while ay <= by + bh {
            ball.push((ax.clamp(0.0, 1.0), ay.clamp(0.0, 1.0)));
            ay += 0.008;
        }
        ax += 0.004;
    }
    // Paddle: one solid vertical bar hugging the right edge. Fine steps so the
    // covered cells form a single contiguous block (no split into two bars).
    let mut paddle: Vec<(f64, f64)> = Vec::new();
    let mut px = 0.95;
    while px <= 1.0 {
        let mut pyy = py - 0.11;
        while pyy <= py + 0.11 {
            paddle.push((px, pyy.clamp(0.0, 1.0)));
            pyy += 0.008;
        }
        px += 0.004;
    }

    let title = if playing {
        " Pong — live "
    } else {
        " Pong — paused "
    };
    let canvas = Canvas::default()
        .block(panel(title, playing))
        .marker(Marker::Block)
        // Black court so the white ball / cyan paddle pop on any terminal theme.
        .background_color(Color::Black)
        .x_bounds([0.0, 1.0])
        .y_bounds([0.0, 1.0])
        .paint(move |ctx| {
            ctx.draw(&Points {
                coords: &net,
                color: Color::DarkGray,
            });
            ctx.draw(&Points {
                coords: &paddle,
                color: CYAN,
            });
            ctx.draw(&Points {
                coords: &ball,
                color: Color::White,
            });
        });
    frame.render_widget(canvas, area);
}

fn draw_doom_canvas(frame: &mut Frame, s: &DoomState, area: Rect, playing: bool) {
    // Split off a HUD strip below the scene: a legend + a live status line, so a
    // viewer can tell what the red block is and what just happened.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(4)])
        .split(area);

    // Enemy screen column from its bearing relative to the view; proximity grows
    // as the countdown runs down (the enemy closing in), so the sprite swells.
    let rel = (s.enemy_x - s.heading) as f64;
    let ex = (0.5 + rel * 0.5).clamp(0.0, 1.0);
    let prox = 1.0 - s.countdown as f64 / s.encounter_frames.max(1) as f64;
    let hw = 0.05 + 0.13 * prox;
    let hh = 0.10 + 0.26 * prox;
    let aligned = (s.enemy_x - s.heading).abs() <= 0.1;
    let hit_flash = s.flash > 0 && s.last == bl1_games::Event::Hit;

    // Enemy sprite: a dense body block with two dark eyes and a ground shadow, so
    // it reads as a creature rather than a stray square.
    let mut body: Vec<(f64, f64)> = Vec::new();
    let mut x = ex - hw;
    while x <= ex + hw {
        let mut y = 0.5 - hh;
        while y <= 0.5 + hh {
            body.push((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
            y += 0.01;
        }
        x += 0.005;
    }
    let eyes = vec![
        (ex - hw * 0.4, 0.5 + hh * 0.45),
        (ex + hw * 0.4, 0.5 + hh * 0.45),
    ];
    let mut shadow: Vec<(f64, f64)> = Vec::new();
    let mut sx = ex - hw;
    while sx <= ex + hw {
        shadow.push((sx.clamp(0.0, 1.0), (0.5 - hh - 0.02).max(0.0)));
        sx += 0.005;
    }

    // On a kill, a bright star-burst explodes over the enemy; the muzzle flashes
    // at the bottom-centre gun on every shot.
    let mut burst: Vec<(f64, f64)> = Vec::new();
    if hit_flash {
        for k in 0..40 {
            let a = k as f64 / 40.0 * std::f64::consts::TAU;
            for r in [0.04, 0.08, 0.12] {
                burst.push(((ex + r * a.cos()).clamp(0.0, 1.0), (0.5 + r * a.sin()).clamp(0.0, 1.0)));
            }
        }
    }
    let mut muzzle: Vec<(f64, f64)> = Vec::new();
    if s.flash > 0 {
        let mut bx = 0.45;
        while bx <= 0.55 {
            let mut by = 0.0;
            while by <= 0.12 {
                muzzle.push((bx, by));
                by += 0.012;
            }
            bx += 0.006;
        }
    }

    // Static corridor drawn in one-point perspective (vanishing box in the
    // centre), plus a couple of receding floor lines, so it reads as a room.
    let gray = Color::Rgb(90, 90, 110);
    let floor = Color::Rgb(60, 60, 75);
    let (vl, vr, vt, vb) = (0.36, 0.64, 0.62, 0.38);
    let walls = [
        (0.0, 1.0, vl, vt),
        (0.0, 0.0, vl, vb),
        (1.0, 1.0, vr, vt),
        (1.0, 0.0, vr, vb),
        (vl, vt, vr, vt),
        (vl, vb, vr, vb),
        (vl, vb, vl, vt),
        (vr, vb, vr, vt),
    ];
    let floor_lines = [(0.0, 0.13, 1.0, 0.13), (0.13, 0.26, 0.87, 0.26)];

    let title = if playing {
        " DOOM — live "
    } else {
        " DOOM — paused "
    };
    let canvas = Canvas::default()
        .block(panel(title, playing))
        .marker(Marker::Block)
        .background_color(Color::Black)
        .x_bounds([0.0, 1.0])
        .y_bounds([0.0, 1.0])
        .paint(move |ctx| {
            for &(x1, y1, x2, y2) in &floor_lines {
                ctx.draw(&CanvasLine { x1, y1, x2, y2, color: floor });
            }
            for &(x1, y1, x2, y2) in &walls {
                ctx.draw(&CanvasLine { x1, y1, x2, y2, color: gray });
            }
            ctx.draw(&Points {
                coords: &shadow,
                color: Color::Rgb(30, 30, 30),
            });
            ctx.draw(&Points {
                coords: &body,
                color: if hit_flash { Color::Yellow } else { Color::Red },
            });
            ctx.draw(&Points {
                coords: &eyes,
                color: Color::Rgb(20, 20, 20),
            });
            if !burst.is_empty() {
                ctx.draw(&Points {
                    coords: &burst,
                    color: Color::Yellow,
                });
            }
            if !muzzle.is_empty() {
                ctx.draw(&Points {
                    coords: &muzzle,
                    color: Color::Rgb(255, 220, 120),
                });
            }
            // Crosshair, fixed at screen centre — green when on target.
            let cross = if aligned { Color::Green } else { CYAN };
            ctx.draw(&CanvasLine { x1: 0.44, y1: 0.5, x2: 0.56, y2: 0.5, color: cross });
            ctx.draw(&CanvasLine { x1: 0.5, y1: 0.42, x2: 0.5, y2: 0.58, color: cross });
        });
    frame.render_widget(canvas, rows[0]);

    draw_doom_hud(frame, s, aligned, prox, rows[1]);
}

/// The legend + live status strip under the DOOM scene: what the symbols mean,
/// and what the culture is doing right now (tracking / locked / kill / miss).
fn draw_doom_hud(frame: &mut Frame, s: &DoomState, aligned: bool, prox: f64, area: Rect) {
    let legend = Line::from(vec![
        Span::styled("🔴 enemy", Style::default().fg(Color::Red)),
        Span::styled("   ✚ your aim", Style::default().fg(CYAN)),
        Span::styled(" (green = locked on)", Style::default().fg(Color::DarkGray)),
        Span::styled("   💥 shot fired", Style::default().fg(Color::Yellow)),
    ]);

    // Live verdict: a fresh kill/miss flash wins, else lock-on vs. tracking.
    let (status, scolor) = if s.flash > 0 && s.last == bl1_games::Event::Hit {
        ("💀 KILL!", Color::Green)
    } else if s.flash > 0 && s.last == bl1_games::Event::Miss {
        ("✗ MISS", Color::Red)
    } else if aligned {
        ("🔒 LOCKED ON", Color::Green)
    } else {
        ("… tracking", Color::Yellow)
    };
    // A distance meter that fills as the enemy closes in.
    let filled = (prox * 8.0).round() as usize;
    let bar: String = (0..8).map(|i| if i < filled { '▓' } else { '░' }).collect();
    let status_line = Line::from(vec![
        Span::styled(
            format!("{status}  "),
            Style::default().fg(scolor).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("kills {} · misses {}  enemy [{}]", s.kills, s.misses, bar),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = panel(" What you're seeing ", false);
    frame.render_widget(
        Paragraph::new(vec![legend, status_line]).block(block),
        area,
    );
}

fn draw_learning_chart(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, area: Rect) {
    let curve = trainer.hit_rate_curve(20);
    let (hits, misses) = (trainer.hits(), trainer.misses());
    let title = format!(" Learning curve — {hits} hits / {misses} misses ");
    let block = panel(&title, false);
    if curve.len() < 2 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  collecting events…",
                Style::default().fg(Color::DarkGray),
            )))
            .block(block),
            area,
        );
        return;
    }
    let pts: Vec<(f64, f64)> = curve
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64 * 100.0))
        .collect();
    let x_max = (pts.len() - 1).max(1) as f64;
    let total_events = hits + misses;

    // Faint horizontal gridlines every 20% so heights are easy to read off.
    let grid: Vec<Vec<(f64, f64)>> = [20.0, 40.0, 60.0, 80.0]
        .iter()
        .map(|&gy| vec![(0.0, gy), (x_max, gy)])
        .collect();
    let mut datasets: Vec<Dataset> = grid
        .iter()
        .map(|g| {
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(g)
        })
        .collect();
    datasets.push(
        Dataset::default()
            .name("hit %")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&pts),
    );

    let x_labels = vec![
        "0".to_string(),
        format!("{}", total_events / 2),
        format!("{total_events} events"),
    ];
    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(
            Axis::default()
                .title("events played →")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("hit %")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, 100.0])
                .labels(vec!["0", "20", "40", "60", "80", "100"]),
        );
    frame.render_widget(chart, area);
}

fn draw_train_gauges(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, area: Rect) {
    let block = panel(" Skill ", false);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // plain-language verdict
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
        ])
        .split(inner);

    let hit = trainer.hit_rate().clamp(0.0, 1.0);
    let recent = trainer.recent_hit_rate(200).clamp(0.0, 1.0);
    let explore = (trainer.sigma() / 0.3).clamp(0.0, 1.0);
    let events = trainer.hits() + trainer.misses();

    // A one-line verdict a non-scientist can read at a glance.
    let (verdict, vcolor) = if events < 40 {
        ("🧠 warming up — just starting to play…", Color::Gray)
    } else if recent >= 0.7 {
        ("🧠 playing well — scoring most encounters!", Color::Green)
    } else if recent >= 0.45 {
        ("🧠 getting the hang of it…", Color::Yellow)
    } else {
        ("🧠 still learning — missing a lot", Color::Red)
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            verdict,
            Style::default().fg(vcolor).add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(hit as f64)
            .label(format!("overall hit {:.0}%", hit * 100.0)),
        rows[1],
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(CYAN))
            .ratio(recent as f64)
            .label(format!("recent hit {:.0}%", recent * 100.0)),
        rows[2],
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Magenta))
            .ratio(explore as f64)
            .label(format!("exploration {:.2}", trainer.sigma())),
        rows[3],
    );
}

fn draw_sensory(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, area: Rect) {
    let (title, legend) = match trainer.view() {
        EnvView::Pong(_) => (
            " Sensory input — ball-Y place code ",
            "◄ ball low    the peak = ball height the culture senses    ball high ►",
        ),
        EnvView::Doom(_) => (
            " Sensory input — enemy-bearing place code ",
            "◄ enemy left    the peak = enemy bearing the culture senses    enemy right ►",
        ),
    };
    let block = panel(title, false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(inner);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            legend,
            Style::default().fg(Color::DarkGray),
        ))),
        rows[0],
    );
    // The place code has one value per band (16), far fewer than the panel is
    // wide. Resample it (linear interpolation) to one bar per column so the bump
    // spreads across the whole width instead of hugging the left edge.
    let bands = trainer.features();
    let width = rows[1].width.max(1) as usize;
    let feats: Vec<u64> = (0..width)
        .map(|x| {
            let pos = if width > 1 {
                x as f32 / (width - 1) as f32 * (bands.len() - 1) as f32
            } else {
                0.0
            };
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(bands.len() - 1);
            let frac = pos - lo as f32;
            let v = bands[lo] * (1.0 - frac) + bands[hi] * frac;
            (v * 1000.0) as u64
        })
        .collect();
    let spark = Sparkline::default()
        .data(&feats)
        .style(Style::default().fg(Color::Magenta));
    frame.render_widget(spark, rows[1]);
}

fn draw_train_stats(frame: &mut Frame, trainer: &dyn bl1_games::Trainer, app: &App, area: Rect) {
    // Game-specific sense/actuator readout.
    let (outcome_label, sense_label, sense_val, act_label, act_val) = match trainer.view() {
        EnvView::Pong(g) => ("hits / misses", "ball y", g.ball_y, "paddle → target", g.paddle_y),
        EnvView::Doom(s) => (
            "kills / misses",
            "enemy x",
            s.enemy_x,
            "aim → target",
            s.heading,
        ),
    };
    let lines = vec![
        stat("step", format!("{}", trainer.step_idx())),
        stat(
            outcome_label,
            format!("{} / {}", trainer.hits(), trainer.misses()),
        ),
        stat("speed", format!("{} steps/frame", app.train_speed)),
        stat("substrate", trainer.substrate().to_string()),
        stat("control", trainer.control().label().to_string()),
        stat("seed", format!("{}", app.train_seed)),
        stat(sense_label, format!("{sense_val:.2}")),
        stat(
            act_label,
            format!("{:.2} → {:.2}", act_val, trainer.last_target()),
        ),
    ];
    frame.render_widget(Paragraph::new(lines).block(panel(" State ", false)), area);
}

// ---------------------------------------------------------------------------
// Science view — the biology metrics, in plain language
// ---------------------------------------------------------------------------

fn draw_science(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Science — what the numbers mean ");
    let Some(r) = &app.result else {
        let p = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No simulation yet.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  Run one from the Simulate tab (press 2) to populate the biology metrics.",
                Style::default().fg(Color::DarkGray),
            )),
        ])
        .block(block);
        frame.render_widget(p, area);
        return;
    };

    let head = Style::default().fg(CYAN).add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "  Culture: {} neurons ({} excitatory) · {:.0} ms window",
                r.n_neurons, r.n_exc, r.duration_ms
            ),
            head,
        )),
        Line::from(""),
    ];

    // Firing rate.
    let fr = r.mean_fr_hz;
    let (fr_tag, fr_c) = if !(0.2..=20.0).contains(&fr) {
        ("outside the usual range", Color::Yellow)
    } else {
        ("physiological ✓", Color::Green)
    };
    lines.extend(sci_metric(
        "Mean firing rate",
        format!("{fr:.2} Hz"),
        fr_tag,
        fr_c,
        "How often a neuron fires. Dissociated cortical cultures rest around 1–10 Hz.",
    ));

    // Burst rate (Wagenaar).
    let br = r.burst_rate_per_min;
    lines.extend(sci_metric(
        "Network burst rate",
        format!("{br:.1} / min"),
        "vs Wagenaar 2006 ≈ 8/min",
        Color::Green,
        "Whole-culture bursts — the signature rhythm of cultured cortex (Wagenaar 2006).",
    ));

    // Branching ratio σ — criticality.
    let sigma = r.branching_ratio;
    let (s_tag, s_c) = if sigma.is_nan() {
        ("n/a", Color::Gray)
    } else if (0.9..=1.1).contains(&sigma) {
        ("near-critical ✓ — like healthy cortex", Color::Green)
    } else if sigma < 0.9 {
        ("subcritical — activity dies out", Color::Yellow)
    } else {
        ("supercritical — activity runs away", Color::Red)
    };
    lines.extend(sci_metric(
        "Branching ratio σ",
        fmt_f64(sigma),
        s_tag,
        s_c,
        "Does one spike trigger ~one more? σ≈1 is criticality — the regime real \
         cortex self-organises to (Beggs & Plenz 2003), best for computation.",
    ));

    // Avalanche exponent.
    let av = r.avalanche_size_exp;
    lines.extend(sci_metric(
        "Avalanche size exponent",
        fmt_f64(av),
        "criticality ≈ −1.5",
        Color::Green,
        "Cascade sizes follow a power law; an exponent near −1.5 is the fingerprint \
         of a critical network (Beggs & Plenz 2003).",
    ));

    // Burst regularity.
    lines.extend(sci_metric(
        "Inter-burst interval",
        fmt_ibi(r.ibi_mean_ms, r.ibi_cv),
        "CV<1 = fairly regular",
        Color::Green,
        "Average gap between bursts and its variability (coefficient of variation).",
    ));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  In short: these are the same metrics neuroscientists use to check that a",
        dim,
    )));
    lines.push(Line::from(Span::styled(
        "  cultured network behaves like living cortex — computed live from the sim.",
        dim,
    )));

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

/// One biology metric as two lines: a headline (label + value + verdict) and a
/// plain-language explanation beneath it.
fn sci_metric(
    label: &str,
    value: String,
    tag: &str,
    tag_color: Color,
    note: &str,
) -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::styled(format!("  {label:<24}"), Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{value:<12}"),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                tag.to_string(),
                Style::default().fg(tag_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!("      {note}"),
            Style::default().fg(Color::DarkGray),
        )),
    ]
}

fn draw_keybar(frame: &mut Frame, app: &App, area: Rect) {
    let groups: Vec<(&str, &str)> = match app.active_tab {
        Tab::Dashboard => vec![
            ("2", "simulate"),
            ("Tab", "view"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Tab::Simulate => vec![
            ("Enter", "run"),
            ("j/k", "select"),
            ("+/-", "neurons"),
            ("[ ]", "duration"),
            ("s", "reseed"),
            ("Tab", "view"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Tab::Train => match app.train_screen {
            TrainScreen::Menu => vec![
                ("↑/↓", "field"),
                ("←/→", "change"),
                ("Enter", "start"),
                ("Tab", "view"),
                ("?", "help"),
                ("q", "quit"),
            ],
            TrainScreen::Playing if app.game_choice.is_tui() => vec![
                ("Space", "play/pause"),
                ("+/-", "speed"),
                ("r", "fresh culture"),
                ("w/o", "save/load brain"),
                ("Esc", "menu"),
                ("?", "help"),
                ("q", "quit"),
            ],
            TrainScreen::Playing => vec![
                ("r", "relaunch"),
                ("Esc", "menu"),
                ("Tab", "view"),
                ("?", "help"),
                ("q", "quit"),
            ],
        },
        Tab::Science => vec![
            ("2", "simulate"),
            ("Tab", "view"),
            ("?", "help"),
            ("q", "quit"),
        ],
        Tab::Results => vec![
            ("j/k", "browse"),
            ("e", "export csv"),
            ("Tab", "view"),
            ("?", "help"),
            ("q", "quit"),
        ],
    };
    let mut line: Vec<Span> = Vec::new();
    for (i, (k, desc)) in groups.into_iter().enumerate() {
        if i > 0 {
            line.push(Span::raw("  "));
        }
        line.extend(key(k, desc));
    }
    let p = Paragraph::new(Line::from(line))
        .alignment(Alignment::Left)
        .style(Style::default().bg(BG_BAR));
    frame.render_widget(p, area);
}

fn key(k: &str, desc: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!(" {k} "),
            Style::default()
                .fg(Color::Black)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {desc}"), Style::default().fg(Color::Gray)),
    ]
}

fn draw_help(frame: &mut Frame, app: &App) {
    let area = centered_rect(64, 70, frame.area());
    frame.render_widget(Clear, area);

    let title = format!(" Help — {} ", app.active_tab.title());
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(CYAN).add_modifier(Modifier::BOLD));

    let mut lines = vec![
        help_head("Global"),
        help_row("Tab / Shift+Tab", "next / previous view"),
        help_row("1 / 2 / 3", "jump to Dashboard / Simulate / Results"),
        help_row("mouse click", "switch tab, focus panel, press a button"),
        help_row("?", "toggle this help"),
        help_row("q / Esc", "quit"),
        Line::from(""),
    ];
    match app.active_tab {
        Tab::Dashboard => {
            lines.push(help_head("Dashboard"));
            lines.push(help_row("Enter / 2", "go to the Simulate view"));
        }
        Tab::Simulate => {
            lines.push(help_head("Simulate"));
            lines.push(help_row(
                "j / k / ↑ ↓",
                "select a config (or scroll a focused raster)",
            ));
            lines.push(help_row("click a config", "select it directly"));
            lines.push(help_row(
                "+ / -  or [-] [+]",
                "double / halve the neuron cap",
            ));
            lines.push(help_row(
                "[ / ]  or [-] [+]",
                "halve / double the preview window",
            ));
            lines.push(help_row("s / [reseed]", "advance the random seed"));
            lines.push(help_row("Enter / r / Run", "run a preview simulation"));
            lines.push(help_row("wheel over raster", "scroll neuron rows"));
        }
        Tab::Train => {
            lines.push(help_head("Train — menu (choose a mode, then Enter)"));
            lines.push(help_row("↑ / ↓", "move between fields"));
            lines.push(help_row(
                "← / →",
                "change: Game (Pong / Doom arena / real Doom), Substrate (feed-forward ↔ recurrent culture), Control, Scenario, Seed",
            ));
            lines.push(help_row("Enter", "enter the selected game"));
            lines.push(Line::from(""));
            lines.push(help_head("Train — while playing"));
            lines.push(help_row("Esc", "return to the mode menu (stops the session)"));
            lines.push(help_row(
                "Space / + / -",
                "TUI games: play-pause, faster, slower",
            ));
            lines.push(help_row("r", "TUI: fresh culture (new seed) · real Doom: relaunch"));
            lines.push(help_row(
                "w / o",
                "TUI: save / load the trained brain (brains/<game>_brain.yaml — share it!)",
            ));
            lines.push(help_row(
                "real Doom",
                "opens a ViZDoom window (separate process); the monitor shows its live kills/episode. Needs `pip install vizdoom numpy`",
            ));
            lines.push(Line::from(""));
            lines.push(help_head("Reading the panels"));
            lines.push(help_row(
                "Pong",
                "white ball crosses left→right; cyan paddle should track it",
            ));
            lines.push(help_row(
                "DOOM",
                "red enemy swells as it nears; crosshair turns green when on target",
            ));
            lines.push(help_row(
                "Learning curve",
                "hit % over events played — climbs as it learns",
            ));
            lines.push(help_row(
                "Outcomes",
                "every ball as a dot: green = hit, red = miss (time →)",
            ));
            lines.push(help_row(
                "Skill",
                "overall & recent hit rate; exploration shrinks over time",
            ));
            lines.push(help_row(
                "Sensory input",
                "16 Y-bands; the lit bar = ball height the culture senses",
            ));
            lines.push(help_row(
                "State",
                "step count, hits/misses, and ball-y vs decoded paddle target",
            ));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  The culture learns Pong by reward-modulated Hebbian pursuit: it",
                Style::default().fg(Color::DarkGray),
            )));
            lines.push(Line::from(Span::styled(
                "  reads the ball's height from the sensory bump and moves the paddle",
                Style::default().fg(Color::DarkGray),
            )));
            lines.push(Line::from(Span::styled(
                "  there; hits reward the synapses that produced the move.",
                Style::default().fg(Color::DarkGray),
            )));
        }
        Tab::Science => {
            lines.push(help_head("Science"));
            lines.push(Line::from(Span::styled(
                "  Plain-language reading of the last simulation's biology metrics",
                Style::default().fg(Color::DarkGray),
            )));
            lines.push(Line::from(Span::styled(
                "  (firing rate, bursts, criticality). Run a sim (2) to populate it.",
                Style::default().fg(Color::DarkGray),
            )));
        }
        Tab::Results => {
            lines.push(help_head("Results"));
            lines.push(help_row("j / k / click", "browse past runs"));
            lines.push(help_row("wheel", "scroll the list"));
            lines.push(help_row("e", "export all runs to results/session_runs.csv"));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Press ? or Esc to close.",
        Style::default().fg(Color::DarkGray),
    )));

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(p, area);
}

fn help_head(s: &str) -> Line<'static> {
    Line::from(Span::styled(
        format!("  {s}"),
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
    ))
}

fn help_row(k: &str, desc: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("   {k:<20}"), Style::default().fg(Color::Yellow)),
        Span::styled(desc.to_string(), Style::default().fg(Color::Gray)),
    ])
}

/// A rectangle centered in `area`, sized as a percentage of it.
fn centered_rect(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_y) / 2),
            Constraint::Percentage(pct_y),
            Constraint::Percentage((100 - pct_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_x) / 2),
            Constraint::Percentage(pct_x),
            Constraint::Percentage((100 - pct_x) / 2),
        ])
        .split(v[1])[1]
}

fn fmt_f64(x: f64) -> String {
    if x.is_nan() {
        "n/a".to_string()
    } else {
        format!("{x:.3}")
    }
}

fn fmt_frac(x: f32) -> String {
    if x.is_nan() {
        "n/a".to_string()
    } else {
        format!("{:.0}%", x * 100.0)
    }
}

fn fmt_ibi(mean: f32, cv: f32) -> String {
    if mean.is_nan() {
        "n/a".to_string()
    } else {
        format!("{mean:.0} ms / {cv:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    /// Render `app` into a `w × h` virtual terminal; panics on any layout bug.
    fn render(app: &mut App, w: u16, h: u16) {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
    }

    #[test]
    fn renders_every_view_without_panicking() {
        let mut app = App::new(None);
        for tab in Tab::ALL {
            app.set_tab(tab);
            render(&mut app, 100, 40);
        }
    }

    #[test]
    fn renders_after_a_run_with_history_and_help() {
        let mut app = App::new(None);
        app.neuron_cap = 100;
        app.preview_ms = 500.0;
        app.run_selected(); // populates result + history, switches to Simulate
        render(&mut app, 100, 40);
        app.set_tab(Tab::Results);
        render(&mut app, 100, 40);
        app.show_help = true;
        render(&mut app, 100, 40);
    }

    #[test]
    fn renders_in_a_tiny_terminal() {
        let mut app = App::new(None);
        app.show_help = true;
        for tab in Tab::ALL {
            app.set_tab(tab);
            render(&mut app, 12, 6);
            render(&mut app, 1, 1);
        }
    }

    #[test]
    fn renders_train_view_with_an_active_trainer() {
        let mut app = App::new(None);
        app.set_tab(Tab::Train);
        app.toggle_training(); // creates the trainer and starts it
        if let Some(t) = app.trainer.as_mut() {
            for _ in 0..60 {
                t.step();
            }
        }
        render(&mut app, 120, 40);
        render(&mut app, 40, 16);
        render(&mut app, 1, 1);
    }
}
