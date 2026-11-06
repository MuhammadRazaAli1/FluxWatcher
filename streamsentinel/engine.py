"""Real-time engine: runs source -> detector -> metrics on a background thread."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import AppConfig
from .metrics import OnlineMetrics
from .models import OnlineAnomalyDetector
from .simulator import Sample, SensorStream

log = logging.getLogger(__name__)

MIN_SPEED, MAX_SPEED = 0.25, 4.0
MAX_CONSECUTIVE_ERRORS = 10


def _f(x: float, nd: int = 4) -> float:
    """Round and make JSON-safe."""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else 0.0


class StreamEngine:
    """Owns the stream source, the detector and the metrics; thread-safe.

    ``source`` must expose ``next() -> Sample``, ``sensor_names`` and optionally
    ``inject(kind)``; by default the built-in :class:`SensorStream` is used. Use
    :meth:`step` to advance deterministically (tests, headless), or
    :meth:`start` to run in real time on a background thread.
    """

    def __init__(self, cfg: AppConfig = AppConfig(), source: Optional[Any] = None):
        self.cfg = cfg
        self.source = source if source is not None else SensorStream(cfg.sim)
        self.names: Tuple[str, ...] = tuple(self.source.sensor_names)
        self.units: Tuple[str, ...] = tuple(s.unit for s in cfg.sim.sensors) \
            if source is None else ("",) * len(self.names)
        self.detector = OnlineAnomalyDetector(len(self.names), cfg.detector, seed=cfg.sim.seed)
        self.metrics = OnlineMetrics(cfg.metrics_grace)
        self._lock = threading.RLock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=cfg.history)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._paused = False
        self._speed = 1.0
        self._last_error: Optional[str] = None
        self._failed = False

    # --------------------------------------------------------------- stepping
    def step(self) -> Dict[str, Any]:
        """Advance one tick and return the JSON-friendly record."""
        with self._lock:
            sample: Sample = self.source.next()
            det = self.detector.update(sample.values)
            self.metrics.update(sample.label, det.alert, sample.t)
            record = {
                "t": int(sample.t),
                "v": [_f(v) for v in sample.values],
                "p": None if det.prediction is None else [_f(v) for v in det.prediction],
                "z": [_f(max(-99.0, min(99.0, v)), 2) for v in det.z],
                "s": _f(min(det.score, 10.0), 3),
                "f": int(det.flagged),
                "a": int(det.alert),
                "y": int(sample.label),
                "k": sample.kind,
                "ys": [int(i) for i in sample.sensors],
                "st": det.status,
            }
            self._history.append(record)
            return record

    # --------------------------------------------------------------- controls
    def inject(self, kind: Optional[str] = None) -> Tuple[bool, str]:
        inject = getattr(self.source, "inject", None)
        if inject is None:
            return False, "this data source does not support injection"
        with self._lock:
            return inject(kind)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_speed(self, speed: float) -> float:
        speed = float(speed)
        if not np.isfinite(speed):
            raise ValueError("speed must be a finite number")
        self._speed = min(MAX_SPEED, max(MIN_SPEED, speed))
        return self._speed

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    # ------------------------------------------------------------- thread I/O
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._failed = False
            self._thread = threading.Thread(target=self._run, name="stream-engine", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        errors = 0
        next_time = time.perf_counter()
        while not self._stop.is_set():
            if self._paused:
                self._stop.wait(0.05)
                next_time = time.perf_counter()
                continue
            try:
                self.step()
                errors = 0
            except Exception as exc:  # keep the service alive, but never hide the error
                errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                log.exception("engine step failed (%d/%d)", errors, MAX_CONSECUTIVE_ERRORS)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    self._failed = True
                    log.error("engine stopped after %d consecutive errors", errors)
                    return
            next_time += self.cfg.tick_interval / self._speed
            delay = next_time - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -1.0:          # fell far behind: resynchronise instead of bursting
                next_time = time.perf_counter()

    # ---------------------------------------------------------------- queries
    def snapshot(self, since: int = -1, limit: int = 400) -> Dict[str, Any]:
        """State for the dashboard: new points after tick ``since`` plus aggregates."""
        with self._lock:
            hist = self._history
            latest = hist[-1]["t"] if hist else -1
            reset = since > latest
            if reset:
                since = -1
            pts: List[Dict[str, Any]] = []
            for rec in reversed(hist):
                if rec["t"] <= since or len(pts) >= limit:
                    break
                pts.append(rec)
            pts.reverse()
            status = hist[-1]["st"] if hist else "starting"
            warm = self.cfg.detector.warmup_ticks
            return {
                "now": latest,
                "reset": reset,
                "points": pts,
                "sensors": [{"name": n, "unit": u} for n, u in zip(self.names, self.units)],
                "status": "error" if self._failed else status,
                "paused": self._paused,
                "speed": self._speed,
                "warmup_ticks": warm,
                "progress": min(1.0, (latest + 1) / warm) if warm else 1.0,
                "metrics": self.metrics.snapshot(),
                "error": self._last_error if self._failed else None,
            }
