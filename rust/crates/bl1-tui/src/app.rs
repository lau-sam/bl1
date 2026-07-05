//! Application state and simulation driving for the TUI.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Instant;

use bl1_analysis::{
    avalanche_distributions, branching_ratio, burst_statistics, detect_bursts,
    estimate_power_law_exponent,
};
use bl1_core::simulate;
use bl1_sim::{Config, Culture};
use ratatui::layout::{Position, Rect};

/// Which neural substrate the Train view learns on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Substrate {
    /// A feed-forward bank of Izhikevich neurons (fast, sharp place code).
    Feedforward,
    /// The full recurrent `bl1-sim` culture as a fixed reservoir (the real brain).
    Reservoir,
}

/// Top-level views, switchable by clicking the tab bar, `Tab`, or `1`/`2`/`3`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Simulate,
    Train,
    Science,
    Results,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Dashboard,
        Tab::Simulate,
        Tab::Train,
        Tab::Science,
        Tab::Results,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Simulate => "Simulate",
            Tab::Train => "Train",
            Tab::Science => "Science",
            Tab::Results => "Results",
        }
    }
}

/// Focusable panels inside the Simulate view (drives keyboard scroll target
/// and the highlighted border).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Configs,
    Params,
    Raster,
    Stats,
}

/// Clickable/hit-testable screen rectangles, refreshed every frame by the
/// renderer so the event loop can map mouse coordinates back to actions.
#[derive(Default, Clone)]
pub struct Regions {
    pub tabs: Vec<Rect>, // parallel to `Tab::ALL`
    pub configs: Rect,
    pub params: Rect,
    pub raster: Rect,
    pub stats: Rect,
    pub btn_neuron_dec: Rect,
    pub btn_neuron_inc: Rect,
    pub btn_dur_dec: Rect,
    pub btn_dur_inc: Rect,
    pub btn_reseed: Rect,
    pub btn_run: Rect,
    pub results: Rect,
}

/// A selectable configuration: a display name and its raw YAML.
pub struct ConfigEntry {
    pub name: String,
    pub yaml: String,
}

/// A compact record of a finished run, listed in the Results view.
pub struct HistoryEntry {
    pub config_name: String,
    pub neuron_cap: usize,
    pub preview_ms: f32,
    pub seed: u64,
    pub n_neurons: usize,
    pub mean_fr_hz: f64,
    pub n_bursts: usize,
    pub burst_rate_per_min: f32,
    pub branching_ratio: f64,
}

/// Results of a completed preview run.
pub struct RunResult {
    pub n_neurons: usize,
    pub n_exc: usize,
    pub dt_ms: f32,
    pub duration_ms: f32,
    pub n_bursts: usize,
    pub mean_fr_hz: f64,
    pub burst_rate_per_min: f32,
    pub ibi_mean_ms: f32,
    pub ibi_cv: f32,
    pub recruitment: f32,
    pub branching_ratio: f64,
    pub avalanche_size_exp: f64,
    /// Raster kept for display (already small: capped neurons × preview steps).
    pub raster: bl1_core::Raster,
    pub n_exc_rows: usize,
}

/// A simulation running on a background thread, plus the parameter snapshot
/// needed to record it when the result arrives.
struct PendingRun {
    rx: Receiver<RunResult>,
    config_name: String,
    neuron_cap: usize,
    preview_ms: f32,
    seed: u64,
    started: Instant,
}

/// Whole-application state.
pub struct App {
    pub configs: Vec<ConfigEntry>,
    pub selected: usize,
    pub neuron_cap: usize,
    pub preview_ms: f32,
    pub seed: u64,
    pub result: Option<RunResult>,
    pub status: String,
    pub should_quit: bool,
    // --- navigation / cockpit state ---
    pub active_tab: Tab,
    pub focus: Focus,
    pub show_help: bool,
    pub raster_scroll: u16,
    pub history: Vec<HistoryEntry>,
    pub results_selected: usize,
    pub regions: Regions,
    pub configs_from_dir: usize,
    pending: Option<PendingRun>,
    // --- live training (Train tab) ---
    pub trainer: Option<Box<dyn bl1_games::Trainer>>,
    pub training: bool,
    pub train_speed: usize,
    pub train_seed: u64,
    /// How the paddle follows the culture's decoded target (direct teleport vs.
    /// inertial smooth pursuit). Baked into the agent at build time, so changing
    /// it rebuilds a fresh trainer.
    pub train_control: bl1_games::PaddleControl,
    /// Which substrate the trainer learns on (feed-forward vs. recurrent culture).
    pub train_substrate: Substrate,
    /// Which game the culture is learning (Pong vs. DOOM aim-and-shoot).
    pub train_game: bl1_games::EnvSpec,
}

impl App {
    /// Build the app, discovering `*.yaml` files under `config_dir` (if any)
    /// and always offering two built-in presets.
    pub fn new(config_dir: Option<&Path>) -> Self {
        let mut configs = builtin_presets();
        let presets = configs.len();
        if let Some(dir) = config_dir
            && let Ok(entries) = fs::read_dir(dir)
        {
            let mut files: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            files.sort();
            for path in files {
                if path.extension().and_then(|s| s.to_str()) == Some("yaml")
                    && let Ok(yaml) = fs::read_to_string(&path)
                    // Only list files that parse as a simulation config. This
                    // skips Python training-only configs (e.g. wagenaar_burst.yaml
                    // has just a `training:` section, no `culture:`).
                    && Config::from_yaml_str(&yaml).is_ok()
                {
                    let name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("config")
                        .to_string();
                    configs.push(ConfigEntry { name, yaml });
                }
            }
        }
        let configs_from_dir = configs.len() - presets;
        Self {
            configs,
            selected: 0,
            neuron_cap: 400,
            preview_ms: 2000.0,
            seed: 1,
            result: None,
            status: "Select a config and press Enter / r to run a preview.".to_string(),
            should_quit: false,
            active_tab: Tab::Dashboard,
            focus: Focus::Configs,
            show_help: false,
            raster_scroll: 0,
            history: Vec::new(),
            results_selected: 0,
            regions: Regions::default(),
            configs_from_dir,
            pending: None,
            trainer: None,
            training: false,
            train_speed: 20,
            train_seed: 1,
            train_control: bl1_games::PaddleControl::Direct,
            train_substrate: Substrate::Feedforward,
            train_game: bl1_games::EnvSpec::Pong,
        }
    }

    /// Build a trainer on the current seed with the current game + substrate +
    /// control. All three are orthogonal choices behind one generic `Learner`.
    fn build_trainer(&self) -> Box<dyn bl1_games::Trainer> {
        let substrate = match self.train_substrate {
            Substrate::Feedforward => bl1_games::SubstrateSpec::FeedForward { per_band: 32 },
            Substrate::Reservoir => bl1_games::SubstrateSpec::Reservoir { n_neurons: 400 },
        };
        Box::new(bl1_games::Learner::build(
            self.train_game,
            substrate,
            self.train_control,
            self.train_seed,
        ))
    }

    // --- selection ---------------------------------------------------------

    pub fn select_next(&mut self) {
        if !self.configs.is_empty() {
            self.selected = (self.selected + 1) % self.configs.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.configs.is_empty() {
            self.selected = (self.selected + self.configs.len() - 1) % self.configs.len();
        }
    }

    pub fn increase_neurons(&mut self) {
        self.neuron_cap = (self.neuron_cap * 2).min(5000);
    }

    pub fn decrease_neurons(&mut self) {
        self.neuron_cap = (self.neuron_cap / 2).max(50);
    }

    pub fn increase_duration(&mut self) {
        self.preview_ms = (self.preview_ms * 2.0).min(20_000.0);
    }

    pub fn decrease_duration(&mut self) {
        self.preview_ms = (self.preview_ms / 2.0).max(500.0);
    }

    pub fn reseed(&mut self) {
        self.seed = self.seed.wrapping_add(1);
    }

    // --- navigation --------------------------------------------------------

    pub fn set_tab(&mut self, tab: Tab) {
        self.active_tab = tab;
    }

    pub fn next_tab(&mut self) {
        let i = Tab::ALL
            .iter()
            .position(|t| *t == self.active_tab)
            .unwrap_or(0);
        self.active_tab = Tab::ALL[(i + 1) % Tab::ALL.len()];
    }

    pub fn prev_tab(&mut self) {
        let i = Tab::ALL
            .iter()
            .position(|t| *t == self.active_tab)
            .unwrap_or(0);
        self.active_tab = Tab::ALL[(i + Tab::ALL.len() - 1) % Tab::ALL.len()];
    }

    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    // --- live training (Train tab) ---

    fn ensure_trainer(&mut self) {
        if self.trainer.is_none() {
            self.trainer = Some(self.build_trainer());
        }
    }

    /// Start/pause the live training loop.
    pub fn toggle_training(&mut self) {
        self.ensure_trainer();
        self.training = !self.training;
        let game = match self.train_game {
            bl1_games::EnvSpec::Pong => "Pong",
            bl1_games::EnvSpec::Doom => "DOOM",
        };
        self.status = if self.training {
            format!("Training… the culture is learning to play {game}.")
        } else {
            "Training paused.".to_string()
        };
    }

    /// Fresh, untrained culture on a new seed.
    pub fn reset_trainer(&mut self) {
        self.train_seed = self.train_seed.wrapping_add(1);
        self.trainer = Some(self.build_trainer());
        self.training = false;
        self.status = format!("Trainer reset (seed {}).", self.train_seed);
    }

    /// Switch the learning substrate (feed-forward bank ↔ recurrent culture).
    /// Rebuilds a fresh trainer on the same seed.
    pub fn toggle_substrate(&mut self) {
        self.train_substrate = match self.train_substrate {
            Substrate::Feedforward => Substrate::Reservoir,
            Substrate::Reservoir => Substrate::Feedforward,
        };
        self.trainer = Some(self.build_trainer());
        self.training = false;
        self.status = match self.train_substrate {
            Substrate::Feedforward => {
                "Substrate: feed-forward bank — fast, sharp place code. Retraining from scratch.".to_string()
            }
            Substrate::Reservoir => {
                "Substrate: recurrent culture — the real bl1-sim brain as a fixed reservoir (heavier). Retraining from scratch.".to_string()
            }
        };
    }

    /// Switch the game the culture is learning (Pong ↔ DOOM aim-and-shoot).
    /// Rebuilds a fresh trainer on the same seed.
    pub fn toggle_game(&mut self) {
        self.train_game = match self.train_game {
            bl1_games::EnvSpec::Pong => bl1_games::EnvSpec::Doom,
            bl1_games::EnvSpec::Doom => bl1_games::EnvSpec::Pong,
        };
        self.trainer = Some(self.build_trainer());
        self.training = false;
        self.status = match self.train_game {
            bl1_games::EnvSpec::Pong => {
                "Game: Pong — track the ball with the paddle. Retraining from scratch.".to_string()
            }
            bl1_games::EnvSpec::Doom => {
                "Game: DOOM — aim at the enemy and shoot. Retraining from scratch.".to_string()
            }
        };
    }

    /// Switch between direct and smooth-pursuit paddle control. The mode is
    /// baked into the agent, so this rebuilds a fresh trainer on the same seed.
    pub fn toggle_control(&mut self) {
        self.train_control = match self.train_control {
            bl1_games::PaddleControl::Direct => bl1_games::PaddleControl::SmoothPursuit,
            bl1_games::PaddleControl::SmoothPursuit => bl1_games::PaddleControl::Direct,
        };
        self.trainer = Some(self.build_trainer());
        self.training = false;
        self.status = match self.train_control {
            bl1_games::PaddleControl::Direct => {
                "Paddle control: direct — the culture's output is the paddle. Retraining from scratch.".to_string()
            }
            bl1_games::PaddleControl::SmoothPursuit => {
                "Paddle control: smooth-pursuit — inertial paddle, the culture must lead the ball. Retraining from scratch.".to_string()
            }
        };
    }

    /// Path of the shareable brain file, per game (copy it to hand off a trained
    /// culture).
    fn brain_path(game: bl1_games::EnvSpec) -> &'static Path {
        match game {
            bl1_games::EnvSpec::Pong => Path::new("brains/pong_brain.yaml"),
            bl1_games::EnvSpec::Doom => Path::new("brains/doom_brain.yaml"),
        }
    }

    /// Save the current trained brain to a shareable YAML file.
    pub fn save_brain(&mut self) {
        let Some(t) = self.trainer.as_ref() else {
            self.status = "Nothing to save — start training first (Space).".to_string();
            return;
        };
        let path = Self::brain_path(self.train_game);
        self.status = match t.save(path) {
            Ok(()) => format!(
                "Brain saved to {} — share this file to hand off your culture.",
                path.display()
            ),
            Err(e) => format!("Save failed: {e}"),
        };
    }

    /// Load a shared brain file and continue training from it. The substrate and
    /// paddle-control mode are restored from the file.
    pub fn load_brain(&mut self) {
        let path = Self::brain_path(self.train_game);
        match bl1_games::load_trainer(path) {
            Ok(agent) => {
                self.train_control = agent.control();
                self.train_substrate = if agent.substrate().contains("culture") {
                    Substrate::Reservoir
                } else {
                    Substrate::Feedforward
                };
                self.train_game = match agent.game_kind() {
                    bl1_games::GameKind::Pong => bl1_games::EnvSpec::Pong,
                    bl1_games::GameKind::Doom => bl1_games::EnvSpec::Doom,
                };
                self.trainer = Some(agent);
                self.training = false;
                self.status = format!(
                    "Loaded brain from {} — press Space to continue.",
                    path.display()
                );
            }
            Err(e) => self.status = format!("Load failed ({}): {e}", path.display()),
        }
    }

    /// Locate the ViZDoom bridge script + repo root, trying paths relative to
    /// both the repo root and the `rust/` dir the TUI is usually launched from.
    fn find_bridge() -> Option<(PathBuf, PathBuf)> {
        for cand in ["scripts/vizdoom_bridge.py", "../scripts/vizdoom_bridge.py"] {
            let p = Path::new(cand);
            if p.exists() {
                let script = p.canonicalize().ok()?;
                let repo = script.parent()?.parent()?.to_path_buf();
                return Some((script, repo));
            }
        }
        None
    }

    /// Pick a Python interpreter: the repo venv if present, else system python3.
    fn find_python(repo: &Path) -> String {
        for cand in [repo.join(".venv/bin/python"), repo.join(".venv/bin/python3")] {
            if cand.exists() {
                return cand.to_string_lossy().into_owned();
            }
        }
        "python3".to_string()
    }

    /// Launch the **real DOOM** (ViZDoom) bridge as a separate process, driven by
    /// the culture. Doom opens its own window; the TUI is just the launcher, so
    /// this pre-flights the prerequisites and reports precisely what's missing.
    /// The current substrate (feed-forward vs. reservoir) carries over.
    pub fn launch_real_doom(&mut self) {
        let Some((script, repo)) = Self::find_bridge() else {
            self.status =
                "Can't find scripts/vizdoom_bridge.py — launch the TUI from the repo.".to_string();
            return;
        };
        // The bridge spawns the brain binary, so it must be built.
        let brain_built = ["release", "debug"]
            .iter()
            .any(|p| repo.join(format!("rust/target/{p}/bl1-brain")).exists());
        if !brain_built {
            self.status =
                "Build the brain first: cd rust && cargo build --release -p bl1-games".to_string();
            return;
        }
        let python = Self::find_python(&repo);
        // Pre-flight: is ViZDoom importable in that interpreter?
        let vizdoom_ok = Command::new(&python)
            .args(["-c", "import vizdoom"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !vizdoom_ok {
            self.status = format!("ViZDoom not installed — run:  {python} -m pip install vizdoom numpy");
            return;
        }
        // Doom + the TUI can't share the terminal, so tee the bridge's output to
        // a log file the user can tail if the Doom window doesn't appear.
        let log_path = repo.join(".vizdoom_bridge.log");
        let mut cmd = Command::new(&python);
        cmd.arg(&script)
            .args(["--scenario", "defend_the_center", "--episodes", "200"])
            .stdin(Stdio::null());
        if self.train_substrate == Substrate::Reservoir {
            cmd.args(["--reservoir", "--neurons", "800"]);
        }
        if let Ok(log) = fs::File::create(&log_path)
            && let Ok(log2) = log.try_clone()
        {
            cmd.stdout(Stdio::from(log)).stderr(Stdio::from(log2));
        }
        match cmd.spawn() {
            Ok(child) => {
                self.status = format!(
                    "Launched real DOOM (ViZDoom) — watch the Doom window (pid {}). Log: {}",
                    child.id(),
                    log_path.display()
                );
            }
            Err(e) => self.status = format!("Failed to launch the DOOM bridge: {e}"),
        }
    }

    pub fn train_faster(&mut self) {
        self.train_speed = (self.train_speed * 2).min(1000);
    }

    pub fn train_slower(&mut self) {
        self.train_speed = (self.train_speed / 2).max(1);
    }

    /// Advance the trainer by `train_speed` game steps if playing. Call once per
    /// event-loop tick.
    pub fn train_tick(&mut self) {
        if self.training
            && let Some(t) = self.trainer.as_mut()
        {
            for _ in 0..self.train_speed {
                t.step();
            }
        }
    }

    /// `j` / down-arrow: context-dependent — scroll the focused raster, browse
    /// the results list, or move to the next config.
    pub fn browse_next(&mut self) {
        match self.active_tab {
            Tab::Simulate if self.focus == Focus::Raster => {
                self.raster_scroll = self.raster_scroll.saturating_add(1);
            }
            Tab::Simulate => self.select_next(),
            Tab::Results => {
                if !self.history.is_empty() {
                    self.results_selected = (self.results_selected + 1).min(self.history.len() - 1);
                }
            }
            Tab::Dashboard | Tab::Train | Tab::Science => {}
        }
    }

    /// `k` / up-arrow: mirror of [`browse_next`].
    pub fn browse_prev(&mut self) {
        match self.active_tab {
            Tab::Simulate if self.focus == Focus::Raster => {
                self.raster_scroll = self.raster_scroll.saturating_sub(1);
            }
            Tab::Simulate => self.select_prev(),
            Tab::Results => self.results_selected = self.results_selected.saturating_sub(1),
            Tab::Dashboard | Tab::Train | Tab::Science => {}
        }
    }

    // --- mouse -------------------------------------------------------------

    /// Route a left-click at `(x, y)` to the region it landed in.
    pub fn handle_click(&mut self, x: u16, y: u16) {
        let pos = Position { x, y };

        // Tab bar is live in every view.
        for (i, r) in self.regions.tabs.iter().enumerate() {
            if r.contains(pos) {
                self.active_tab = Tab::ALL[i];
                return;
            }
        }

        match self.active_tab {
            Tab::Simulate => self.click_simulate(pos, y),
            Tab::Results => {
                if self.regions.results.contains(pos) {
                    let top = self.regions.results.y + 1; // border
                    if y >= top {
                        let idx = (y - top) as usize;
                        if idx < self.history.len() {
                            self.results_selected = idx;
                        }
                    }
                }
            }
            Tab::Dashboard | Tab::Train | Tab::Science => {}
        }
    }

    fn click_simulate(&mut self, pos: Position, y: u16) {
        let r = &self.regions;
        if r.btn_run.contains(pos) {
            self.start_run();
        } else if r.btn_neuron_inc.contains(pos) {
            self.increase_neurons();
        } else if r.btn_neuron_dec.contains(pos) {
            self.decrease_neurons();
        } else if r.btn_dur_inc.contains(pos) {
            self.increase_duration();
        } else if r.btn_dur_dec.contains(pos) {
            self.decrease_duration();
        } else if r.btn_reseed.contains(pos) {
            self.reseed();
        } else if r.configs.contains(pos) {
            self.focus = Focus::Configs;
            let top = r.configs.y + 1; // border
            if y >= top {
                let idx = (y - top) as usize;
                if idx < self.configs.len() {
                    self.selected = idx;
                }
            }
        } else if r.raster.contains(pos) {
            self.focus = Focus::Raster;
        } else if r.params.contains(pos) {
            self.focus = Focus::Params;
        } else if r.stats.contains(pos) {
            self.focus = Focus::Stats;
        }
    }

    /// Route a scroll-wheel tick to whatever the cursor hovers.
    pub fn handle_scroll(&mut self, up: bool, x: u16, y: u16) {
        let pos = Position { x, y };
        if self.regions.raster.contains(pos) {
            self.focus = Focus::Raster;
            self.raster_scroll = if up {
                self.raster_scroll.saturating_sub(1)
            } else {
                self.raster_scroll.saturating_add(1)
            };
        } else if self.regions.configs.contains(pos) {
            if up {
                self.select_prev();
            } else {
                self.select_next();
            }
        } else if self.regions.results.contains(pos) {
            if up {
                self.browse_prev();
            } else {
                self.browse_next();
            }
        }
    }

    // --- simulation --------------------------------------------------------

    /// True while a background simulation is in flight.
    pub fn is_running(&self) -> bool {
        self.pending.is_some()
    }

    /// Milliseconds elapsed since the current run started (0 when idle).
    pub fn run_elapsed_ms(&self) -> u128 {
        self.pending
            .as_ref()
            .map(|p| p.started.elapsed().as_millis())
            .unwrap_or(0)
    }

    /// Parse the selected config, returning `(name, config)` or setting a status
    /// message and returning `None` on error.
    fn prepare_run(&mut self) -> Option<(String, Config)> {
        self.active_tab = Tab::Simulate;
        let Some(entry) = self.configs.get(self.selected) else {
            self.status = "No configuration selected.".to_string();
            return None;
        };
        let name = entry.name.clone();
        match Config::from_yaml_str(&entry.yaml) {
            Ok(c) => Some((name, c)),
            Err(e) => {
                self.status = format!("Config parse error: {e}");
                None
            }
        }
    }

    /// Kick off a preview on a background thread so the UI stays responsive.
    /// A no-op if a run is already in flight.
    pub fn start_run(&mut self) {
        if self.pending.is_some() {
            return;
        }
        let Some((config_name, config)) = self.prepare_run() else {
            return;
        };
        let (cap, ms, seed) = (self.neuron_cap, self.preview_ms, self.seed);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(compute_run(config, cap, ms, seed));
        });
        self.pending = Some(PendingRun {
            rx,
            config_name,
            neuron_cap: cap,
            preview_ms: ms,
            seed,
            started: Instant::now(),
        });
        self.status = format!("Simulating {cap} neurons for {ms:.0} ms…");
    }

    /// Poll the background run; call once per event-loop tick.
    pub fn poll_run(&mut self) {
        let Some(p) = &self.pending else { return };
        match p.rx.try_recv() {
            Ok(result) => {
                let (name, cap, ms, seed) =
                    (p.config_name.clone(), p.neuron_cap, p.preview_ms, p.seed);
                self.pending = None;
                self.record_result(result, name, cap, ms, seed);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.pending = None;
                self.status = "Simulation failed (worker thread stopped).".to_string();
            }
        }
    }

    /// Run a capped preview synchronously (used by `--headless` and tests).
    pub fn run_selected(&mut self) {
        let Some((name, config)) = self.prepare_run() else {
            return;
        };
        let (cap, ms, seed) = (self.neuron_cap, self.preview_ms, self.seed);
        let result = compute_run(config, cap, ms, seed);
        self.record_result(result, name, cap, ms, seed);
    }

    /// Store a finished run: update the raster/stats panels and the history.
    fn record_result(
        &mut self,
        result: RunResult,
        config_name: String,
        neuron_cap: usize,
        preview_ms: f32,
        seed: u64,
    ) {
        self.raster_scroll = 0;
        self.history.push(HistoryEntry {
            config_name,
            neuron_cap,
            preview_ms,
            seed,
            n_neurons: result.n_neurons,
            mean_fr_hz: result.mean_fr_hz,
            n_bursts: result.n_bursts,
            burst_rate_per_min: result.burst_rate_per_min,
            branching_ratio: result.branching_ratio,
        });
        self.results_selected = self.history.len() - 1;
        self.status = format!(
            "Ran {} neurons for {:.0} ms (seed {}). {} bursts detected.",
            result.n_neurons, result.duration_ms, seed, result.n_bursts
        );
        self.result = Some(result);
    }

    /// Write the session's run history to a CSV file and return its path.
    pub fn export_results(&mut self) {
        if self.history.is_empty() {
            self.status = "Nothing to export yet — run a preview first.".to_string();
            return;
        }
        match self.write_results_csv() {
            Ok(path) => self.status = format!("Exported {} runs to {path}", self.history.len()),
            Err(e) => self.status = format!("Export failed: {e}"),
        }
    }

    fn write_results_csv(&self) -> std::io::Result<String> {
        let dir = Path::new("results");
        fs::create_dir_all(dir)?;
        let path = dir.join("session_runs.csv");
        let mut file = fs::File::create(&path)?;
        writeln!(
            file,
            "config,neuron_cap,preview_ms,seed,n_neurons,mean_fr_hz,n_bursts,burst_rate_per_min,branching_ratio"
        )?;
        for h in &self.history {
            writeln!(
                file,
                "{},{},{:.0},{},{},{:.4},{},{:.4},{:.4}",
                h.config_name,
                h.neuron_cap,
                h.preview_ms,
                h.seed,
                h.n_neurons,
                h.mean_fr_hz,
                h.n_bursts,
                h.burst_rate_per_min,
                h.branching_ratio,
            )?;
        }
        Ok(path.display().to_string())
    }
}

/// Build a culture from `config`, run a capped preview, and analyze the raster.
/// Pure and self-contained so it can run on a background thread.
fn compute_run(mut config: Config, neuron_cap: usize, preview_ms: f32, seed: u64) -> RunResult {
    config.culture.n_neurons = config.culture.n_neurons.min(neuron_cap).max(1);
    let dt = config.simulation.dt_ms.max(0.01);
    let n_steps = ((preview_ms / dt).round() as usize).max(1);

    let culture = Culture::build(&config, seed);
    let drive = culture.background_current(n_steps, seed.wrapping_mul(2654435761));
    let mut state = culture.make_sim_state();
    let raster = simulate(&culture.network, &mut state, &drive, n_steps, dt);

    let thr = config.burst_detection.threshold_std;
    let min_dur = config.burst_detection.min_duration_ms;
    let bursts = detect_bursts(&raster, dt, thr, min_dur);
    let bstats = burst_statistics(&bursts, n_steps as f32 * dt);
    let sigma = branching_ratio(&raster, dt, 4.0);
    let (sizes, _dur) = avalanche_distributions(&raster, dt, 4.0);
    let size_exp = estimate_power_law_exponent(&sizes);
    let mean_fr_hz = raster.mean_firing_rate_hz(dt);

    RunResult {
        n_neurons: culture.n_neurons(),
        n_exc: culture.n_exc,
        dt_ms: dt,
        duration_ms: preview_ms,
        n_bursts: bstats.n_bursts,
        mean_fr_hz,
        burst_rate_per_min: bstats.burst_rate_per_min,
        ibi_mean_ms: bstats.ibi_mean_ms,
        ibi_cv: bstats.ibi_cv,
        recruitment: bstats.recruitment_mean,
        branching_ratio: sigma,
        avalanche_size_exp: size_exp,
        n_exc_rows: culture.n_exc,
        raster,
    }
}

/// Two always-available presets so the TUI is useful without any config files.
fn builtin_presets() -> Vec<ConfigEntry> {
    vec![
        ConfigEntry {
            name: "[preset] quick-200".to_string(),
            yaml: "culture:\n  n_neurons: 200\n  substrate_um: [800, 800]\n  p_max: 0.3\n  g_exc: 0.12\n  g_inh: 0.36\nsimulation:\n  dt_ms: 0.5\nstp:\n  enabled: true\n".to_string(),
        },
        ConfigEntry {
            name: "[preset] wagenaar-like".to_string(),
            yaml: "culture:\n  n_neurons: 1000\n  ei_ratio: 0.8\n  substrate_um: [3000, 3000]\n  lambda_um: 200.0\n  p_max: 0.21\n  g_exc: 0.12\n  g_inh: 0.36\nsimulation:\n  dt_ms: 0.5\nsynapses:\n  nmda_ratio: 0.37\nstp:\n  enabled: true\n  excitatory:\n    U: 0.30\n    tau_rec_ms: 800.0\n    tau_fac_ms: 0.001\n  inhibitory:\n    U: 0.04\n    tau_rec_ms: 100.0\n    tau_fac_ms: 1000.0\nburst_detection:\n  threshold_std: 1.5\n  min_duration_ms: 50.0\n".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_cycle_both_ways() {
        let mut app = App::new(None);
        assert!(app.active_tab == Tab::Dashboard);
        app.next_tab();
        assert!(app.active_tab == Tab::Simulate);
        app.prev_tab();
        app.prev_tab();
        assert!(app.active_tab == Tab::Results); // wrapped around
    }

    #[test]
    fn click_on_tab_rect_switches_view() {
        let mut app = App::new(None);
        app.regions.tabs = vec![
            Rect::new(0, 0, 5, 1),
            Rect::new(6, 0, 5, 1),
            Rect::new(12, 0, 5, 1),
        ];
        app.handle_click(7, 0); // inside the second tab (Simulate)
        assert!(app.active_tab == Tab::Simulate);
    }

    #[test]
    fn click_on_button_adjusts_parameter() {
        let mut app = App::new(None);
        app.set_tab(Tab::Simulate);
        app.regions.btn_neuron_inc = Rect::new(10, 10, 3, 1);
        let before = app.neuron_cap;
        app.handle_click(11, 10);
        assert!(app.neuron_cap > before);
    }

    #[test]
    fn results_browse_clamps_without_history() {
        let mut app = App::new(None);
        app.set_tab(Tab::Results);
        app.browse_next();
        assert_eq!(app.results_selected, 0);
        app.browse_prev();
        assert_eq!(app.results_selected, 0);
    }

    #[test]
    fn running_a_preset_records_history() {
        let mut app = App::new(None);
        app.neuron_cap = 100;
        app.preview_ms = 500.0;
        app.run_selected();
        assert_eq!(app.history.len(), 1);
        assert!(app.result.is_some());
        assert!(app.active_tab == Tab::Simulate);
        assert!(!app.is_running());
    }

    #[test]
    fn export_without_history_reports_and_writes_nothing() {
        let mut app = App::new(None);
        app.export_results();
        assert!(app.status.contains("Nothing to export"));
    }

    #[test]
    fn background_run_completes_and_records() {
        let mut app = App::new(None);
        app.neuron_cap = 60;
        app.preview_ms = 500.0;
        app.start_run();
        assert!(app.is_running());
        // A second start is a no-op while one is in flight.
        app.start_run();

        for _ in 0..2000 {
            app.poll_run();
            if !app.is_running() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(!app.is_running(), "run should finish");
        assert_eq!(app.history.len(), 1);
        assert!(app.result.is_some());
    }
}
