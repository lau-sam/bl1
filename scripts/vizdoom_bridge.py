#!/usr/bin/env python3
"""Wire the BL-1 cortical culture to *real* DOOM through ViZDoom.

The culture is the brain: the `bl1-brain` Rust binary runs the spiking-culture
substrate + a reward-modulated node-perturbation readout, and speaks a tiny
line protocol over stdio (``<reward> <obs...>`` in, ``<action...>`` out). This
script is the game side of the loop — it drives a real ViZDoom scenario, turns
each rendered frame into a coarse "retina" the culture can sense, sends it the
reward the last action earned, and maps the returned actions onto Doom buttons.

    ViZDoom frame ─► retina[N] ─┐
                                ├─► bl1-brain ─► actions[M] ─► Doom buttons
    kills/health  ─► reward ────┘   (spiking culture, learns online)

This is the honest DishBrain-DOOM loop: a *simulated* culture learning to aim
and shoot in the real Doom engine. It is not a full-campaign agent — use an
aim-and-shoot scenario (defend_the_center, basic) where the culture's 1-D-ish
sensory/motor map is a fit.

Requirements (install on the machine with a display / SDL):
    pip install vizdoom numpy
ViZDoom ships the scenarios and a freedoom WAD, so no commercial DOOM data is
needed. Build the brain first:  (cd rust && cargo build --release -p bl1-games)

Example:
    python scripts/vizdoom_bridge.py --scenario defend_the_center --episodes 50
"""

from __future__ import annotations

import argparse
import itertools
import os
import signal
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("This bridge needs numpy:  pip install numpy")


def find_brain_binary(explicit: str | None) -> Path:
    """Locate the compiled bl1-brain binary."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--brain-bin not found: {p}")
        return p
    repo = Path(__file__).resolve().parent.parent
    for profile in ("release", "debug"):
        cand = repo / "rust" / "target" / profile / "bl1-brain"
        if cand.exists():
            return cand
    sys.exit(
        "bl1-brain binary not found. Build it first:\n"
        "    cd rust && cargo build --release -p bl1-games"
    )


def enemy_retina(state, n_bins: int, screen_w: int):
    """Encode enemy bearings from ViZDoom's labels buffer into a place code.

    The labels buffer gives ground-truth on-screen objects, so we get a clean
    bearing for each enemy instead of guessing from pixels — the culture reads
    the same kind of Gaussian-bump place code it learns on in Pong and the arena.
    Returns ``(obs[n_bins], nearest_bearing)`` where ``nearest_bearing`` is the
    screen-x in ``[0, 1]`` of the largest (closest) enemy, or ``None`` if none is
    visible. Bumps are weighted by apparent size, so closer enemies dominate.
    """
    obs = np.zeros(n_bins, dtype=np.float32)
    labels = getattr(state, "labels", None) or []
    idx = np.arange(n_bins, dtype=np.float32)
    best = None  # (bearing, size)
    for lab in labels:
        name = getattr(lab, "object_name", "")
        if name in ("DoomPlayer", ""):  # skip the agent itself
            continue
        cx = (lab.x + lab.width / 2.0) / max(1, screen_w)
        size = float(lab.width * lab.height)
        center = min(max(cx, 0.0), 1.0) * (n_bins - 1)
        obs += size * np.exp(-((idx - center) ** 2) / (2.0 * 1.5 * 1.5))
        if best is None or size > best[1]:
            best = (cx, size)
    m = obs.max()
    if m > 1e-6:
        obs /= m
    return obs, (best[0] if best is not None else None)


def encode_retina_pixels(frame: np.ndarray, n_bins: int) -> np.ndarray:
    """Fallback bearing code from raw pixels (used only if labels are off).

    Per screen-column band, how far its brightness deviates from the frame's
    median — a crude saliency bump wherever something stands out.
    """
    if frame.ndim == 3:
        frame = frame.mean(axis=0) if frame.shape[0] <= 4 else frame.mean(axis=2)
    h, w = frame.shape
    band = frame[h // 3 : 2 * h // 3, :].astype(np.float32)
    col = band.mean(axis=0)
    salience = np.abs(col - np.median(col))
    edges = np.linspace(0, w, n_bins + 1).astype(int)
    obs = np.array([salience[edges[i] : edges[i + 1]].mean() for i in range(n_bins)])
    m = obs.max()
    return obs / m if m > 1e-6 else obs


def main() -> None:
    ap = argparse.ArgumentParser(description="Drive real DOOM (ViZDoom) with the BL-1 culture.")
    ap.add_argument("--scenario", default="defend_the_center",
                    help="ViZDoom scenario name (defend_the_center, basic, ...)")
    ap.add_argument("--config", default=None, help="Explicit path to a .cfg (overrides --scenario)")
    ap.add_argument("--inputs", type=int, default=32, help="Retina bins = brain inputs")
    ap.add_argument("--actions", type=int, default=3, help="Brain action heads (turn/none/shoot)")
    ap.add_argument("--reservoir", action="store_true", help="Use the recurrent-culture substrate")
    ap.add_argument("--neurons", type=int, default=800, help="Reservoir size (with --reservoir)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--frame-skip", type=int, default=4, help="Game tics per decision")
    ap.add_argument("--no-window", action="store_true", help="Run headless (no Doom window)")
    ap.add_argument("--brain-bin", default=None, help="Path to the bl1-brain binary")
    ap.add_argument("--brain-file", default=None,
                    help="Load/save the culture's readout here (resume across sessions)")
    args = ap.parse_args()

    try:
        import vizdoom as vzd
    except ImportError:
        sys.exit("This bridge needs ViZDoom:  pip install vizdoom")

    # --- set up the real Doom game -----------------------------------------
    game = vzd.DoomGame()
    if args.config:
        game.load_config(args.config)
    else:
        cfg = Path(vzd.scenarios_path) / f"{args.scenario}.cfg"
        if not cfg.exists():
            sys.exit(f"scenario config not found: {cfg}")
        game.load_config(str(cfg))
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_labels_buffer_enabled(True)  # ground-truth enemy bearings
    game.set_window_visible(not args.no_window)
    # We drive our own buttons: turn left, turn right, attack.
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables(
        [vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.AMMO2]
    )
    game.set_seed(args.seed)
    game.init()

    # --- launch the culture brain ------------------------------------------
    brain_bin = find_brain_binary(args.brain_bin)
    cmd = [str(brain_bin), "--inputs", str(args.inputs), "--actions", str(args.actions),
           "--seed", str(args.seed)]
    if args.reservoir:
        cmd += ["--reservoir", "--neurons", str(args.neurons)]
    if args.brain_file:
        cmd += ["--load", args.brain_file, "--save", args.brain_file]
    brain = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    assert brain.stdin and brain.stdout

    def decide(obs: np.ndarray, reward: float) -> list[float]:
        line = f"{reward:.5f} " + " ".join(f"{v:.5f}" for v in obs)
        brain.stdin.write(line + "\n")
        brain.stdin.flush()
        reply = brain.stdout.readline()
        return [float(t) for t in reply.split()]

    screen_w = game.get_screen_width()

    def bearing_now() -> float | None:
        """Nearest-enemy screen bearing of the current (post-action) frame."""
        if game.is_episode_finished():
            return None
        _, b = enemy_retina(game.get_state(), args.inputs, screen_w)
        return b

    total_kills = 0
    frames = 0  # decisions taken this session
    shots = 0   # real bullets fired this session (AMMO2 decrements, not dry clicks)
    start_ammo = 0  # this episode's starting AMMO2 = the per-episode kill ceiling
    # 0 (or less) episodes = run until stopped (Esc in the TUI / Ctrl-C / SIGTERM).
    ep_iter = itertools.count() if args.episodes <= 0 else range(args.episodes)
    total_str = "inf" if args.episodes <= 0 else str(args.episodes)
    try:
        for ep in ep_iter:
            game.new_episode()
            prev_reward = 0.0
            prev_kills = 0.0
            prev_health = 100.0
            start_ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
            prev_ammo = start_ammo
            while not game.is_episode_finished():
                state = game.get_state()
                obs, _ = enemy_retina(state, args.inputs, screen_w)
                actions = decide(obs, prev_reward)

                # Map action heads onto buttons: head 0 steers (low = left, high =
                # right, mid = hold), the last head fires.
                turn = actions[0] if actions else 0.5
                shoot = actions[-1] if len(actions) >= 2 else 0.0
                buttons = [1 if turn < 0.4 else 0, 1 if turn > 0.6 else 0, 1 if shoot > 0.5 else 0]
                game.make_action(buttons, args.frame_skip)
                frames += 1
                # Count bullets actually spent (AMMO2 dropped), so accuracy reflects
                # real shots, not trigger pulls with an empty clip.
                ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
                shots += int(max(0.0, prev_ammo - ammo))
                prev_ammo = ammo

                # Dense reward for the action just taken: a big kill bonus, a small
                # penalty for taking damage, and — crucially — a dense shaping term
                # that rewards bringing the nearest enemy toward the crosshair
                # (screen centre). Without this shaping, kills are far too sparse
                # for the node-perturbation readout to get any gradient.
                kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                health = game.get_game_variable(vzd.GameVariable.HEALTH)
                bearing = bearing_now()
                center = (1.0 - 2.0 * abs(bearing - 0.5)) if bearing is not None else -0.1
                # Couple the shoot head to aim: firing while centred pays, spraying
                # off-target costs — so the shoot readout gets a dense gradient
                # instead of waiting on rare kills.
                shot_bonus = 0.4 * (center - 0.3) if buttons[2] else 0.0
                prev_reward = (
                    3.0 * (kills - prev_kills)
                    - 0.02 * max(0.0, prev_health - health)
                    + 0.3 * center
                    + shot_bonus
                )
                prev_kills, prev_health = kills, health

            ep_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            total_kills += ep_kills
            # The TUI monitor parses this line (kills / shots / frames / ammo per token).
            print(f"episode {ep + 1:>3}/{total_str}: kills {ep_kills:.0f}  "
                  f"shots {shots}  frames {frames}  ammo {start_ammo:.0f}  "
                  f"(mean {total_kills / (ep + 1):.2f})")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Graceful shutdown: send the empty line so the brain saves its readout,
        # close stdin, give it a moment to write, then close the Doom engine.
        try:
            brain.stdin.write("\n")
            brain.stdin.flush()
            brain.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            brain.wait(timeout=10)
        except subprocess.TimeoutExpired:
            brain.terminate()
        game.close()
        print(f"\nStopped. {total_kills} kills · {shots} shots · {frames} frames this session.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Exit cleanly on SIGTERM/SIGINT so the `finally` above closes ViZDoom and the
    # brain (the TUI sends SIGTERM to the whole process group to stop a session).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
