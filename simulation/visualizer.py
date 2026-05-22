"""
ANC Phase 1 — Real-time Visualization Dashboard
================================================
PyQtGraph-based GPU-accelerated window running at VIS_FPS (default 30 Hz).

Threading contract:
    • This module owns ZERO audio logic.
    • The Qt/PortAudio callbacks NEVER touch vis_q.  Only the inference thread
      does (via a non-blocking put_nowait in main_hardware._push_vis).
    • The QTimer callback (_tick) runs exclusively in the Qt main thread and
      is the ONLY reader of vis_q.

4 synchronized panels:
    Plot 1 — Raw Input          : dirty mic waveform from _input_q
    Plot 2 — AI Noise Estimate  : low-freq noise component extracted by model
    Plot 3 — Anti-Noise Signal  : phase-inverted waveform (× −1) sent to speaker
    Plot 4 — Virtual Clean Zone : mathematical preview of cancellation residual
                                  (raw + anti-noise = raw − noise_estimate),
                                  title updated with instantaneous dBFS metrics.
"""
from __future__ import annotations

import queue
from collections import namedtuple
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from configs.config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    VIS_FPS,
    VIS_YRANGE,
)
from utils.logger_setup import get_logger

log = get_logger(__name__)

# ── Shared data container ─────────────────────────────────────────────────────
# Created by the inference thread, consumed by the Qt timer — namedtuple is
# immutable so there is no need for a lock after construction.
VisFrame = namedtuple("VisFrame", ["raw", "noise", "anti", "clean"])

# ── Colour palette (dark professional theme) ──────────────────────────────────
_BG       = "#0d1117"   # window / plot background
_FG       = "#8b949e"   # axis lines, tick labels, grid
_C_RAW    = "#58a6ff"   # blue   — Plot 1: dirty mic
_C_NOISE  = "#ff7b72"   # red    — Plot 2: extracted noise (the enemy)
_C_ANTI   = "#3fb950"   # green  — Plot 3: anti-noise weapon
_C_CLEAN  = "#e3b341"   # amber  — Plot 4: virtual clean zone residual

# Apply global config BEFORE any widget is constructed (pyqtgraph requirement)
pg.setConfigOptions(
    antialias=True,
    background=_BG,
    foreground=_FG,
)
try:
    pg.setConfigOption("useOpenGL", True)   # GPU rendering on Jetson Maxwell GPU
except Exception:
    log.debug("OpenGL unavailable — falling back to software rendering.")

# ── Qt enum compatibility shims (PyQt5 vs PyQt6) ─────────────────────────────
# PyQt6 moved every enum into its own namespace class.
# We resolve all needed values once at import time so the rest of the code
# stays clean and works identically on both bindings.
try:
    # PyQt6 — enums live under their type class
    _Qt_DotLine      = QtCore.Qt.PenStyle.DotLine
    _Qt_PreciseTimer = QtCore.Qt.TimerType.PreciseTimer
except AttributeError:
    # PyQt5 — enums live directly on Qt
    _Qt_DotLine      = QtCore.Qt.DotLine
    _Qt_PreciseTimer = QtCore.Qt.PreciseTimer


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard window
# ══════════════════════════════════════════════════════════════════════════════

class ANCDashboard(pg.GraphicsLayoutWidget):
    """
    Main visualization window.  Must be instantiated and `show()`-ed in the
    Qt main thread (enforced by Qt itself).

    Parameters
    ----------
    vis_q       : queue.Queue[VisFrame]  Shared queue written by inference thread.
    stop_event  : threading.Event        Set by this window on close, and also
                                         monitored to auto-close when the pipeline dies.
    """

    def __init__(
        self,
        vis_q: queue.Queue,
        stop_event,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)

        self._vis_q      = vis_q
        self._stop_event = stop_event
        self._x_ms       = np.linspace(
            0, CHUNK_SIZE / SAMPLE_RATE * 1000, CHUNK_SIZE, endpoint=False, dtype=np.float32
        )
        self._frame_count = 0

        self.resize(1440, 860)
        self.setWindowTitle("ANC Phase 1  |  Real-time Noise Cancellation Dashboard")

        self._build_plots()
        self._start_timers()

        log.info(
            "ANCDashboard ready — %d plots @ %d FPS (interval=%d ms)",
            4, VIS_FPS, int(1000 / VIS_FPS),
        )

    # ── Plot construction ──────────────────────────────────────────────────────

    def _build_plots(self) -> None:
        label_style = {"color": _FG, "font-size": "9pt"}
        title_style = {"color": "#c9d1d9", "size": "10pt"}
        x_label     = f"Time within chunk (ms)  [chunk = {CHUNK_SIZE/SAMPLE_RATE*1e3:.1f} ms @ {SAMPLE_RATE} Hz]"
        zeros       = np.zeros(CHUNK_SIZE, dtype=np.float32)

        specs = [
            # (row, title, pen_color, y_label, fixed_y_range)
            (0, "Plot 1 — Raw Input  (dirty mic signal)",           _C_RAW,   "Amplitude", True),
            (1, "Plot 2 — AI Noise Estimate  (< 200 Hz band)",      _C_NOISE, "Amplitude", True),
            (2, "Plot 3 — Anti-Noise Signal  (× −1, sent to amp)",  _C_ANTI,  "Amplitude", True),
            (3, "Plot 4 — Virtual Clean Zone  (residual preview)",   _C_CLEAN, "Amplitude", False),
        ]

        self._plots  = []
        self._curves = []

        for row, title, color, y_label, fixed_y in specs:
            p = self.addPlot(row=row, col=0)
            p.setTitle(title, **title_style)
            p.setLabel("left",   y_label, **label_style)
            p.setLabel("bottom", x_label, **label_style)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setXRange(0, self._x_ms[-1], padding=0.01)
            p.getAxis("left").setWidth(50)
            p.getAxis("left").setStyle(tickFont=None)

            if fixed_y:
                p.setYRange(*VIS_YRANGE, padding=0)
                p.setMouseEnabled(x=False, y=False)
            else:
                # Plot 4: auto-range Y so we can see the (hopefully small) residual
                p.enableAutoRange(axis="y", enable=True)
                p.setMouseEnabled(x=False, y=True)

            pen   = pg.mkPen(color=color, width=1)
            curve = p.plot(x=self._x_ms, y=zeros, pen=pen)

            # Thin reference zero-line on every plot
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(_FG, width=1, style=_Qt_DotLine)))

            self._plots.append(p)
            self._curves.append(curve)

        # "Waiting for data" overlay on Plot 1 only
        self._wait_label = pg.TextItem("Waiting for audio data…", color=_FG, anchor=(0.5, 0.5))
        self._plots[0].addItem(self._wait_label)
        self._wait_label.setPos(self._x_ms[CHUNK_SIZE // 2], 0)
        self._data_arrived = False

    # ── Timer setup ────────────────────────────────────────────────────────────

    def _start_timers(self) -> None:
        interval_ms = max(1, round(1000 / VIS_FPS))

        # Primary render timer — fires every frame interval
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setTimerType(_Qt_PreciseTimer)
        self._render_timer.timeout.connect(self._tick)
        self._render_timer.start(interval_ms)

        # Heartbeat — keeps Python signal handlers alive inside Qt's C++ event loop
        # and auto-closes the window when the pipeline sets _stop_event.
        self._heartbeat = QtCore.QTimer(self)
        self._heartbeat.timeout.connect(self._on_heartbeat)
        self._heartbeat.start(150)   # 150 ms is fine; no visual impact

    # ── Per-frame update ───────────────────────────────────────────────────────

    def _tick(self) -> None:
        """
        Called every ~33 ms in the Qt main thread.

        Strategy: drain the entire vis_q and keep only the NEWEST frame.
        This means the GUI always shows the latest data and gracefully
        handles cases where inference is temporarily faster than rendering.
        """
        latest: Optional[VisFrame] = None
        try:
            while True:
                latest = self._vis_q.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            return   # no new data since last tick — keep current display

        # Remove "waiting" overlay on first real frame
        if not self._data_arrived:
            self._plots[0].removeItem(self._wait_label)
            self._data_arrived = True

        # ── setData() reuses the curve's internal buffer — zero allocation ────
        self._curves[0].setData(x=self._x_ms, y=latest.raw)
        self._curves[1].setData(x=self._x_ms, y=latest.noise)
        self._curves[2].setData(x=self._x_ms, y=latest.anti)
        self._curves[3].setData(x=self._x_ms, y=latest.clean)

        # ── Update Plot 4 title with live dB metrics ──────────────────────────
        self._plots[3].setTitle(
            self._make_p4_title(latest),
            **{"color": "#c9d1d9", "size": "10pt"},
        )

        self._frame_count += 1

    @staticmethod
    def _make_p4_title(frame: VisFrame) -> str:
        """
        Compute three complementary metrics for the Plot 4 title bar:

        • Noise Reduction (NR):
            How much quieter the clean residual is vs. the extracted noise.
            NR = 20·log10(RMS(noise) / RMS(clean))
            Positive value = noise is being suppressed (good).

        • Input level (dBFS):
            RMS of the raw mic signal, referenced to full-scale ±1.

        • Noise level (dBFS):
            RMS of the estimated noise, referenced to full-scale ±1.
        """
        _eps = 1e-10
        rms_raw   = float(np.sqrt(np.mean(frame.raw   ** 2))) + _eps
        rms_noise = float(np.sqrt(np.mean(frame.noise ** 2))) + _eps
        rms_clean = float(np.sqrt(np.mean(frame.clean ** 2))) + _eps

        nr_db       = 20.0 * np.log10(rms_noise / rms_clean)
        input_dbfs  = 20.0 * np.log10(rms_raw)
        noise_dbfs  = 20.0 * np.log10(rms_noise)

        return (
            f"Plot 4 — Virtual Clean Zone  |  "
            f"NR: {nr_db:+.1f} dB  ·  "
            f"Input: {input_dbfs:.1f} dBFS  ·  "
            f"Noise: {noise_dbfs:.1f} dBFS"
        )

    # ── Heartbeat & lifecycle ─────────────────────────────────────────────────

    def _on_heartbeat(self) -> None:
        """Auto-close the window if the pipeline has requested a stop."""
        if self._stop_event.is_set() and self.isVisible():
            log.info("Pipeline stop event detected — closing dashboard.")
            self.close()

    def stop_timers(self) -> None:
        """Stop Qt timers before the event loop exits."""
        self._render_timer.stop()
        self._heartbeat.stop()

    def closeEvent(self, event) -> None:
        """
        Called when the user closes the window or stop_timers triggers .close().
        Signal the pipeline to shut down, then accept the event.
        """
        log.info("Dashboard closeEvent — signalling pipeline shutdown.")
        self._stop_event.set()
        self.stop_timers()
        event.accept()
