//! Rendering: a lazygit-style multi-panel layout.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{App, RunResult};

/// Draw the whole UI for the current frame.
pub fn draw(frame: &mut Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(frame.area());

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(0)])
        .split(root[0]);

    draw_sidebar(frame, app, body[0]);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Min(0),
            Constraint::Length(10),
        ])
        .split(body[1]);

    draw_params(frame, app, right[0]);
    draw_raster(frame, app, right[1]);
    draw_stats(frame, app, right[2]);
    draw_keybar(frame, root[1]);
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .configs
        .iter()
        .map(|c| ListItem::new(c.name.clone()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Configs "))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    let mut state = ListState::default();
    state.select(Some(app.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_params(frame: &mut Frame, app: &App, area: Rect) {
    let text = vec![
        Line::from(vec![
            Span::styled("neuron cap  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.neuron_cap),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   (+/-)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("preview     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.0} ms", app.preview_ms),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   ([ / ])", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("seed        ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.seed),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   (s)", Style::default().fg(Color::DarkGray)),
        ]),
    ];
    let p =
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title(" Parameters "));
    frame.render_widget(p, area);
}

fn draw_raster(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Raster (green = excitatory, red = inhibitory) ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = match &app.result {
        Some(res) => render_raster_lines(res, inner.width as usize, inner.height as usize),
        None => vec![Line::from(Span::styled(
            "No run yet — press Enter or r.",
            Style::default().fg(Color::DarkGray),
        ))],
    };
    frame.render_widget(Paragraph::new(lines), inner);
}

/// Turn a raster into colored block-character rows sized to `width × height`.
fn render_raster_lines(res: &RunResult, width: usize, height: usize) -> Vec<Line<'static>> {
    let raster = &res.raster;
    let (n_steps, n_neurons) = (raster.n_steps, raster.n_neurons);
    if width == 0 || height == 0 || n_steps == 0 || n_neurons == 0 {
        return vec![Line::from("")];
    }
    let rows = height.min(n_neurons);
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
            // Map density (mean spikes per cell) to a shade; scale up since
            // per-cell spike fractions are small.
            let level = ((density * 40.0).sqrt() * (SHADES.len() - 1) as f32)
                .round()
                .clamp(0.0, (SHADES.len() - 1) as f32) as usize;
            s.push(SHADES[level]);
        }
        lines.push(Line::from(Span::styled(s, Style::default().fg(color))));
    }
    lines
}

fn draw_stats(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Statistics ");
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

fn draw_keybar(frame: &mut Frame, area: Rect) {
    let spans = vec![
        key("j/k", "select"),
        key("+/-", "neurons"),
        key("[ ]", "duration"),
        key("s", "reseed"),
        key("Enter/r", "run"),
        key("q", "quit"),
    ];
    let mut line: Vec<Span> = Vec::new();
    for (i, group) in spans.into_iter().enumerate() {
        if i > 0 {
            line.push(Span::styled("  ", Style::default()));
        }
        line.extend(group);
    }
    let p = Paragraph::new(Line::from(line))
        .alignment(Alignment::Left)
        .style(Style::default().bg(Color::Rgb(30, 30, 40)));
    frame.render_widget(p, area);
}

fn key(k: &'static str, desc: &'static str) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!(" {k} "),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {desc}"), Style::default().fg(Color::Gray)),
    ]
}
