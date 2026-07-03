//! Rendering: a lazygit/k9s-style, mouse-friendly cockpit.
//!
//! Every frame refreshes `app.regions` with the on-screen rectangles of the
//! tab bar, panels, and buttons so the event loop can hit-test mouse clicks.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

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
            Constraint::Length(1), // keybar
        ])
        .split(frame.area());

    draw_tabs(frame, app, root[0]);

    match app.active_tab {
        Tab::Dashboard => draw_dashboard(frame, app, root[1]),
        Tab::Simulate => draw_simulate(frame, app, root[1]),
        Tab::Results => draw_results(frame, app, root[1]),
    }

    draw_keybar(frame, app, root[2]);

    if app.show_help {
        draw_help(frame, app);
    }
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
    let val = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);

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
    for t in 0..n_steps {
        let c = (t * cols / n_steps).min(cols - 1);
        for (j, &s) in raster.row(t).iter().enumerate() {
            let r = (j * rows / n_neurons).min(rows - 1);
            sums[r * cols + c] += s;
            counts[r * cols + c] += 1;
        }
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
        Span::styled(
            value,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
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
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
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
        Tab::Results => vec![
            ("j/k", "browse"),
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
        Tab::Results => {
            lines.push(help_head("Results"));
            lines.push(help_row("j / k / click", "browse past runs"));
            lines.push(help_row("wheel", "scroll the list"));
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
}
