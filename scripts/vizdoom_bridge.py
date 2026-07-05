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
import os
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


def encode_retina(frame: np.ndarray, n_bins: int) -> np.ndarray:
    """Turn a grayscale Doom frame into an ``n_bins`` horizontal bearing code.

    Enemies and muzzle contrast stand out against the arena; we take, per screen
    column band, how far its brightness deviates from the frame's median (a crude
    saliency), giving a bump wherever something interesting sits in that bearing.
    Coarse on purpose — the culture reads a place code, not pixels.
    """
    if frame.ndim == 3:  # (C, H, W) or (H, W, C) → luminance
        frame = frame.mean(axis=0) if frame.shape[0] <= 4 else frame.mean(axis=2)
    h, w = frame.shape
    # Focus on the central vertical band (where enemies appear on the horizon).
    band = frame[h // 3 : 2 * h // 3, :].astype(np.float32)
    col = band.mean(axis=0)  # per-column brightness, length w
    salience = np.abs(col - np.median(col))
    # Aggregate columns into n_bins.
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
    game.set_window_visible(not args.no_window)
    # We drive our own buttons: turn left, turn right, attack.
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT])
    game.set_seed(args.seed)
    game.init()

    # --- launch the culture brain ------------------------------------------
    brain_bin = find_brain_binary(args.brain_bin)
    cmd = [str(brain_bin), "--inputs", str(args.inputs), "--actions", str(args.actions),
           "--seed", str(args.seed)]
    if args.reservoir:
        cmd += ["--reservoir", "--neurons", str(args.neurons)]
    brain = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    assert brain.stdin and brain.stdout

    def decide(obs: np.ndarray, reward: float) -> list[float]:
        line = f"{reward:.5f} " + " ".join(f"{v:.5f}" for v in obs)
        brain.stdin.write(line + "\n")
        brain.stdin.flush()
        reply = brain.stdout.readline()
        return [float(t) for t in reply.split()]

    total_kills = 0
    try:
        for ep in range(args.episodes):
            game.new_episode()
            prev_reward = 0.0
            prev_kills = 0.0
            prev_health = 100.0
            while not game.is_episode_finished():
                state = game.get_state()
                obs = encode_retina(state.screen_buffer, args.inputs)
                actions = decide(obs, prev_reward)

                # Map action heads onto buttons: head 0 steers (low = left, high =
                # right, mid = hold), the last head fires.
                turn = actions[0] if actions else 0.5
                shoot = actions[-1] if len(actions) >= 2 else 0.0
                buttons = [1 if turn < 0.4 else 0, 1 if turn > 0.6 else 0, 1 if shoot > 0.5 else 0]
                game.make_action(buttons, args.frame_skip)

                # Reward the action just taken: reward kills, punish taking damage.
                kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                health = game.get_game_variable(vzd.GameVariable.HEALTH)
                prev_reward = 1.0 * (kills - prev_kills) - 0.02 * max(0.0, prev_health - health)
                prev_kills, prev_health = kills, health

            ep_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            total_kills += ep_kills
            print(f"episode {ep + 1:>3}/{args.episodes}: kills {ep_kills:.0f}  "
                  f"(mean {total_kills / (ep + 1):.2f})")
    finally:
        try:
            brain.stdin.write("\n")  # graceful shutdown
            brain.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        brain.terminate()
        game.close()

    print(f"\nDone. {total_kills} kills over {args.episodes} episodes "
          f"({total_kills / max(1, args.episodes):.2f}/episode).")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
