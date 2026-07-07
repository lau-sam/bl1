#!/usr/bin/env python3
"""Fair substrate comparison on real DOOM (ViZDoom), headless.

Runs a controlled sweep — {scenario} x {substrate} x {seed} — each from a *fresh*
brain, and reports converged kills/episode with error bars across seeds. The
displayed in-TUI "mean/ep" is cumulative from episode 1 (dragged down by the
cold start); this measures the mean over the last window instead, which is the
converged skill, and checks stationarity so we know the run was long enough.

Usage:
    # run the sweep (headless), then analyse
    python scripts/doom_compare.py --episodes 300 --seeds 1,2,3 --jobs 2
    # re-analyse existing logs without re-running
    python scripts/doom_compare.py --analyze-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BRIDGE = REPO / "scripts" / "vizdoom_bridge.py"


def run_one(out: Path, scenario: str, substrate: str, seed: int, episodes: int) -> Path:
    """Run one fresh headless session; return the log path."""
    tag = f"{scenario}_{substrate}_seed{seed}"
    log = out / f"{tag}.log"
    brain = out / f"brain_{tag}.yaml"
    brain.unlink(missing_ok=True)  # start cold — independent run
    cmd = [
        sys.executable, str(BRIDGE),
        "--scenario", scenario, "--no-window",
        "--episodes", str(episodes), "--seed", str(seed),
        "--brain-file", str(brain),
    ]
    if substrate == "reservoir":
        cmd += ["--reservoir", "--neurons", "800"]
    with log.open("w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, check=False)
    return log


def kills_from_log(log: Path) -> list[int]:
    """Extract per-episode kills from a bridge log."""
    ks: list[int] = []
    for line in log.read_text().splitlines():
        s = line.strip()
        if not s.startswith("episode "):
            continue
        toks = s.split()
        if "kills" in toks:
            i = toks.index("kills")
            try:
                ks.append(int(float(toks[i + 1])))
            except (ValueError, IndexError):
                pass
    return ks


def converged_mean(kills: list[int], window: int) -> tuple[float, float, bool]:
    """Return (mean of last `window`, mean of the window before it, stationary?).

    Stationary if the two windows agree within 15% — i.e. learning has plateaued
    and the last-window mean is a fair estimate of converged skill.
    """
    if len(kills) < 2 * window:
        window = max(1, len(kills) // 3)
    last = kills[-window:]
    prev = kills[-2 * window : -window] or last
    m_last = statistics.fmean(last)
    m_prev = statistics.fmean(prev)
    stationary = abs(m_last - m_prev) <= 0.15 * max(m_prev, 1e-9)
    return m_last, m_prev, stationary


def analyse(out: Path, window: int) -> None:
    logs = sorted(out.glob("*.log"))
    if not logs:
        sys.exit(f"no logs in {out} — run the sweep first")
    # group by (scenario, substrate) -> list of (seed, converged_mean, stationary)
    groups: dict[tuple[str, str], list[tuple[int, float, bool]]] = {}
    for log in logs:
        parts = log.stem.rsplit("_", 1)  # <scenario>_<substrate>, seed<K>
        if len(parts) != 2 or not parts[1].startswith("seed"):
            continue
        head, seedtok = parts
        # head == "<scenario>_<substrate>"; substrate is the last token
        scen, _, sub = head.rpartition("_")
        seed = int(seedtok[4:])
        kills = kills_from_log(log)
        if not kills:
            continue
        m_last, _, stat = converged_mean(kills, window)
        groups.setdefault((scen, sub), []).append((seed, m_last, stat))

    print(f"\nConverged kills/episode — mean over last {window} eps, across seeds\n")
    print(f"{'scenario':<20} {'substrate':<14} {'mean±std':<16} {'seeds':<20} conv?")
    print("-" * 82)
    for (scen, sub) in sorted(groups):
        rows = sorted(groups[(scen, sub)])
        means = [m for _, m, _ in rows]
        std = statistics.pstdev(means) if len(means) > 1 else 0.0
        per_seed = " ".join(f"s{sd}={m:.1f}" for sd, m, _ in rows)
        allconv = all(c for _, _, c in rows)
        print(
            f"{scen:<20} {sub:<14} "
            f"{statistics.fmean(means):.2f}±{std:.2f}".ljust(16)
            + f" {per_seed:<20} {'yes' if allconv else 'NO (extend)'}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--scenarios", default="defend_the_center,defend_the_line")
    ap.add_argument("--substrates", default="feedforward,reservoir")
    ap.add_argument("--jobs", type=int, default=2, help="parallel runs")
    ap.add_argument("--window", type=int, default=100, help="converged window (eps)")
    ap.add_argument("--out", default=None, help="output dir for logs + brains")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO / "results" / "doom_compare"
    out.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        seeds = [int(s) for s in args.seeds.split(",")]
        scenarios = args.scenarios.split(",")
        substrates = args.substrates.split(",")
        jobs = [
            (out, scen, sub, seed, args.episodes)
            for scen in scenarios
            for sub in substrates
            for seed in seeds
        ]
        print(f"running {len(jobs)} headless sessions ({args.episodes} eps each, "
              f"{args.jobs} parallel) → {out}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, *j): j for j in jobs}
            for fut in concurrent.futures.as_completed(futs):
                _, scen, sub, seed, _ = futs[fut]
                fut.result()
                print(f"  done: {scen} · {sub} · seed {seed}")

    analyse(out, args.window)


if __name__ == "__main__":
    main()
