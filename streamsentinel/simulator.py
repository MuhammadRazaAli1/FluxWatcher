"""Synthetic multi-sensor stream with labelled, injectable anomalies."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import ANOMALY_KINDS, SimConfig


@dataclass(frozen=True)
class Sample:
    """One tick of the stream."""

    t: int
    values: np.ndarray            # shape (n_sensors,)
    label: int                    # 1 while an anomaly episode is active
    kind: str                     # anomaly kind or ""
    sensors: Tuple[int, ...]      # indices of affected sensors (empty if normal)


class _Episode:
    """An anomaly episode affecting one or more sensors."""

    def __init__(self, kind: str, sensors: Tuple[int, ...], length: int,
                 magnitudes: np.ndarray, signs: np.ndarray, noise_scale: np.ndarray):
        self.kind = kind
        self.sensors = sensors
        self.length = length
        self.mag = magnitudes
        self.sign = signs
        self.noise_scale = noise_scale
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.length

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> None:
        for j, s in enumerate(self.sensors):
            if self.kind in ("spike", "level_shift"):
                values[s] += self.sign[j] * self.mag[j]
            elif self.kind == "drift":
                values[s] += self.sign[j] * self.mag[j] * (self.i + 1) / self.length
            elif self.kind == "variance_burst":
                values[s] += rng.normal(0.0, self.noise_scale[j])
        self.i += 1


class SensorStream:
    """Seasonal + correlated + autocorrelated-noise sensors with anomaly episodes.

    The stream is deterministic for a given seed. ``inject`` may be called from
    another thread; the request is applied at the next ``next()`` call.
    """

    def __init__(self, cfg: SimConfig = SimConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self._n = len(cfg.sensors)
        self._noise = np.zeros(self._n)
        self._load = 0.0
        self._episode: Optional[_Episode] = None
        self._pending: Optional[str] = None
        self._lock = threading.Lock()
        w = np.asarray(cfg.kind_weights, dtype=float)
        self._kind_p = w / w.sum()

    @property
    def sensor_names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.cfg.sensors)

    # ------------------------------------------------------------------ API
    def inject(self, kind: Optional[str] = None) -> Tuple[bool, str]:
        """Request an anomaly at the next tick. Returns (accepted, message)."""
        if kind is not None and kind not in ANOMALY_KINDS:
            return False, f"unknown anomaly kind '{kind}'"
        with self._lock:
            if self._episode is not None or self._pending is not None:
                return False, "an anomaly is already in progress"
            if self.t < self.cfg.clean_ticks:
                return False, "the system is still learning its baseline"
            self._pending = kind or "random"
        return True, "anomaly injected"

    def next(self) -> Sample:
        cfg = self.cfg
        t = self.t
        rng = self.rng

        self._load = 0.98 * self._load + 0.15 * rng.normal()
        self._noise = cfg.ar_phi * self._noise + np.array([s.noise for s in cfg.sensors]) \
            * rng.normal(size=self._n)
        clean = np.empty(self._n)
        for i, s in enumerate(cfg.sensors):
            season = s.amp * math.sin(2.0 * math.pi * t / s.period + s.phase)
            clean[i] = s.base + season + s.coupling * self._load + self._noise[i]

        values = clean
        label, kind, affected = 0, "", ()
        with self._lock:
            if self._episode is None and t >= cfg.clean_ticks:
                requested = self._pending
                self._pending = None
                if requested is not None:
                    chosen = None if requested == "random" else requested
                    self._episode = self._new_episode(chosen)
                elif rng.random() < cfg.anomaly_rate:
                    self._episode = self._new_episode(None)
            ep = self._episode
            if ep is not None:
                ep.perturb(values, rng)
                label, kind, affected = 1, ep.kind, ep.sensors
                if ep.done:
                    self._episode = None
        self.t += 1
        return Sample(t=t, values=values, label=label, kind=kind, sensors=tuple(affected))

    # ------------------------------------------------------------ internals
    def _new_episode(self, kind: Optional[str]) -> _Episode:
        rng = self.rng
        if kind is None:
            kind = str(rng.choice(ANOMALY_KINDS, p=self._kind_p))
        n_aff = 1 if (self._n == 1 or rng.random() < 0.75) else 2
        sensors = tuple(int(i) for i in rng.choice(self._n, size=n_aff, replace=False))
        specs = [self.cfg.sensors[i] for i in sensors]
        signs = rng.choice([-1.0, 1.0], size=n_aff)
        if kind == "spike":
            length = int(rng.integers(3, 6))
            mag = np.array([rng.uniform(8, 12) * s.noise_std for s in specs])
        elif kind == "level_shift":
            length = int(rng.integers(30, 61))
            mag = np.array([rng.uniform(1.5, 2.5) * s.amp for s in specs])
        elif kind == "drift":
            length = int(rng.integers(60, 121))
            mag = np.array([rng.uniform(2.0, 3.5) * s.amp for s in specs])
        else:  # variance_burst
            length = int(rng.integers(20, 41))
            mag = np.zeros(n_aff)
        noise_scale = np.array([rng.uniform(5, 8) * s.noise for s in specs])
        return _Episode(kind, sensors, length, mag, signs, noise_scale)
