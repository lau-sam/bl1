"""Dark-themed colour scheme for the BL-1 live monitor.

Defines per-channel-group colours matching the eight doom-neuron
functional groups, plus helpers for applying a dark DOOM-aesthetic
theme to matplotlib axes and figures.

The palette complements the publication-oriented colours in
:mod:`bl1.visualization._style` but is optimised for high-contrast
readability on a dark background during real-time monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

# -- Channel group colours (doom-neuron functional groups) ----------------

CHANNEL_COLORS: dict[str, str] = {
    "encoding": "#2196F3",  # blue
    "move_forward": "#4CAF50",  # green
    "move_backward": "#F44336",  # red
    "move_left": "#FF9800",  # orange
    "move_right": "#9C27B0",  # purple
    "turn_left": "#00BCD4",  # cyan
    "turn_right": "#E91E63",  # pink
    "attack": "#FFC107",  # amber
}

CHANNEL_GROUP_NAMES: list[str] = [
    "encoding",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "attack",
]

# Convenience list for indexed access (same order as CHANNEL_GROUP_NAMES).
CHANNEL_COLOR_LIST: list[str] = [CHANNEL_COLORS[n] for n in CHANNEL_GROUP_NAMES]

# -- Dark theme constants -------------------------------------------------

BG_COLOR = "#1a1a1a"
PANEL_BG = "#222222"
TEXT_COLOR = "#cccccc"
GRID_COLOR = "#333333"
SPINE_COLOR = "#444444"
ACCENT_GREEN = "#39ff14"  # DOOM-style neon green for highlights
MONITOR_DPI = 100


def apply_dark_theme(ax: matplotlib.axes.Axes) -> None:
    """Apply the dark monitor theme to a single *ax*.

    Sets background, spine, tick, and grid colours to match the dark
    DOOM aesthetic used during live monitoring sessions.
    """
    ax.set_facecolor(PANEL_BG)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)

    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)


def apply_monitor_style(fig: matplotlib.figure.Figure) -> None:
    """Apply the dark monitor theme to an entire *fig*.

    Calls :func:`apply_dark_theme` on every axes in the figure and sets
    the figure-level background colour.
    """
    fig.patch.set_facecolor(BG_COLOR)

    for ax in fig.get_axes():
        apply_dark_theme(ax)

    # Style any suptitle if present.
    if fig._suptitle is not None:  # noqa: SLF001
        fig._suptitle.set_color(TEXT_COLOR)  # noqa: SLF001
