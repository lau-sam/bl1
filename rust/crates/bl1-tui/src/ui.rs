//! Rendering: a lazygit/k9s-style, mouse-friendly cockpit.
//!
//! Every frame refreshes `app.regions` with the on-screen rectangles of the
//! tab bar, panels, and buttons so the event loop can hit-test mouse clicks.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Points};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem, ListState,
    Paragraph, Sparkline, Wrap,
};

use crate::app::{App, Focus, RunResult, Tab};

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
    let Some(trainer) = app.trainer.as_ref() else {
        let p = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Live training — a cultured network learns to play Pong.",
                Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press Space to start; watch the paddle track the ball and the",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "  hit-rate curve climb as reward-modulated plasticity kicks in.",
                Style::default().fg(Color::DarkGray),
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title(" Train "));
        frame.render_widget(p, area);
        return;
    };

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(9)])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(outer[0]);
    draw_pong_canvas(frame, trainer, top[0], app.training);
    draw_learning_chart(frame, trainer, top[1]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(36),
            Constraint::Percentage(30),
            Constraint::Percentage(34),
        ])
        .split(outer[1]);
    draw_train_gauges(frame, trainer, bottom[0]);
    draw_sensory(frame, trainer, bottom[1]);
    draw_train_stats(frame, trainer, app, bottom[2]);
}

fn draw_pong_canvas(
    frame: &mut Frame,
    trainer: &bl1_games::PursuitAgent,
    area: Rect,
    playing: bool,
) {
    let g = trainer.game();
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
    let (bw, bh) = (0.02, 0.035);
    let mut ball: Vec<(f64, f64)> = Vec::new();
    for a in 0..=6 {
        for b in 0..=6 {
            ball.push((
                (bx - bw + 2.0 * bw * a as f64 / 6.0).clamp(0.0, 1.0),
                (by - bh + 2.0 * bh * b as f64 / 6.0).clamp(0.0, 1.0),
            ));
        }
    }
    // Paddle: filled vertical bar hugging the right edge.
    let mut paddle: Vec<(f64, f64)> = Vec::new();
    for a in 0..=5 {
        for b in 0..=28 {
            paddle.push((
                0.955 + 0.045 * a as f64 / 5.0,
                (py - 0.11 + 0.22 * b as f64 / 28.0).clamp(0.0, 1.0),
            ));
        }
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

fn draw_learning_chart(frame: &mut Frame, trainer: &bl1_games::PursuitAgent, area: Rect) {
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
    let x_max = (pts.len() - 1) as f64;
    let total_events = hits + misses;
    let datasets = vec![
        Dataset::default()
            .name("hit %")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&pts),
    ];
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
                .labels(vec!["0", "50", "100"]),
        );
    frame.render_widget(chart, area);
}

fn draw_train_gauges(frame: &mut Frame, trainer: &bl1_games::PursuitAgent, area: Rect) {
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
        ("🧠 playing well — returning most balls!", Color::Green)
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

fn draw_sensory(frame: &mut Frame, trainer: &bl1_games::PursuitAgent, area: Rect) {
    let block = panel(" Sensory input — ball-Y place code ", false);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(inner);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "16 Y-bands · the lit bar = ball height the culture senses",
            Style::default().fg(Color::DarkGray),
        ))),
        rows[0],
    );
    let feats: Vec<u64> = trainer
        .features()
        .iter()
        .map(|&v| (v * 1000.0) as u64)
        .collect();
    let spark = Sparkline::default()
        .data(&feats)
        .style(Style::default().fg(Color::Magenta));
    frame.render_widget(spark, rows[1]);
}

fn draw_train_stats(frame: &mut Frame, trainer: &bl1_games::PursuitAgent, app: &App, area: Rect) {
    let g = trainer.game();
    let lines = vec![
        stat("step", format!("{}", trainer.step_idx())),
        stat(
            "hits / misses",
            format!("{} / {}", trainer.hits(), trainer.misses()),
        ),
        stat("speed", format!("{} steps/frame", app.train_speed)),
        stat("seed", format!("{}", app.train_seed)),
        stat("ball y", format!("{:.2}", g.ball_y)),
        stat(
            "paddle → target",
            format!("{:.2} → {:.2}", g.paddle_y, trainer.last_target()),
        ),
    ];
    frame.render_widget(Paragraph::new(lines).block(panel(" State ", false)), area);
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
        Tab::Train => vec![
            ("Space", "play/pause"),
            ("r", "reset"),
            ("+/-", "speed"),
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
            lines.push(help_head("Train"));
            lines.push(help_row("Space", "start / pause live training"));
            lines.push(help_row("r", "reset to a fresh culture (new seed)"));
            lines.push(help_row("+ / -", "faster / slower (steps per frame)"));
            lines.push(Line::from(""));
            lines.push(help_head("Reading the panels"));
            lines.push(help_row("Pong", "yellow ball crosses left→right; cyan paddle should track it"));
            lines.push(help_row("Learning curve", "hit % over events played — climbs as it learns"));
            lines.push(help_row("Skill", "overall & recent hit rate; exploration shrinks over time"));
            lines.push(help_row("Sensory input", "16 Y-bands; the lit bar = ball height the culture senses"));
            lines.push(help_row("State", "step count, hits/misses, and ball-y vs decoded paddle target"));
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
