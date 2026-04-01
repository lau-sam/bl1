"""RatInABox-style composable, streaming-aware visualization for BL-1.

This module provides live monitoring of simulated cortical cultures
during closed-loop experiments.  Unlike the static post-hoc plots in
:mod:`bl1.visualization`, every component here is designed for
**streaming**: O(1) append per tick, thread-safe access, and efficient
``render_frame()`` for MJPEG output.

Quick start::

    from bl1.monitor import ActivityMonitor

    mon = ActivityMonitor(n_neurons=100_000)

    # In the simulation loop:
    mon.record_tick(spike_counts, electrode_spikes=elec)

    # Composable single-panel plot (RatInABox pattern):
    fig, ax = mon.plot_raster(window_s=10.0)

    # Full dashboard:
    fig, axes = mon.plot_dashboard(window_s=10.0)

    # Numpy RGB frame for MJPEG streaming:
    rgb = mon.render_frame(width=640, height=480)

    # Replay a recorded session:
    from bl1.monitor import animate_session
    anim = animate_session(mon, speed_up=5.0, save_path="session.mp4")

    # MJPEG HTTP streaming server:
    from bl1.monitor import NeuralMJPEGServer
    server = NeuralMJPEGServer(port=12350)
    server.start()
"""

from bl1.monitor.activity import ActivityMonitor
from bl1.monitor.animation import animate_session
from bl1.monitor.logger import TensorBoardAdapter, UnifiedLogger
from bl1.monitor.mjpeg import NeuralMJPEGServer
from bl1.monitor.style import (
    CHANNEL_COLOR_LIST,
    CHANNEL_COLORS,
    CHANNEL_GROUP_NAMES,
    apply_dark_theme,
    apply_monitor_style,
)

__all__ = [
    "ActivityMonitor",
    "NeuralMJPEGServer",
    "TensorBoardAdapter",
    "UnifiedLogger",
    "animate_session",
    "CHANNEL_COLORS",
    "CHANNEL_COLOR_LIST",
    "CHANNEL_GROUP_NAMES",
    "apply_dark_theme",
    "apply_monitor_style",
]
