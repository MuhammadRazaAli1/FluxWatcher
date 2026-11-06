"""Online learning components and the ensemble anomaly detector.

Pipeline per tick
-----------------
1. Standardise the raw reading (scaler calibrated on the first ticks, then frozen).
2. A multi-output Recursive Least Squares (RLS) autoregressive model forecasts every
   sensor from the last ``lags`` readings of *all* sensors. Its residual, expressed as
   a robust z-score (exponentially weighted statistics), measures how surprising the
   reading is given the recent dynamics.
3. An Isolation Forest, periodically refitted on a sliding window of clean feature
   vectors, scores how unusual the combined state (values, residual z, rolling |z|) is.
4. An operating-envelope check measures how far each reading lies outside the range
   seen in clean data (catches sustained level shifts and drifts that a forecaster
   adapts to).
5. The three normalised scores are combined with ``max``. A tick is *flagged* when the
   score reaches 1; an *alert* is raised after ``min_consecutive`` flagged ticks (or
   at once if the score reaches ``strong_score``) and cleared after ``cooldown``
   clean ticks.
6. Models only learn from ticks that are neither flagged nor inside an alert, so
   anomalies do not poison the baseline; after ``max_alert_run`` ticks they re-adapt
   (the "anomaly" is then treated as a new normal).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import DetectorConfig

log = logging.getLogger(__name__)


class RecursiveLeastSquares:
    """Multi-output RLS with exponential forgetting (shared input covariance)."""

    def __init__(self, n_inputs: int, n_outputs: int, forgetting: float = 0.999,
                 delta: float = 100.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lam = float(forgetting)
        self.delta = float(delta)
        self.max_trace = 10.0 * delta * n_inputs
        self.W = np.zeros((n_inputs, n_outputs))
        self.P = np.eye(n_inputs) * delta

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W

    def update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Learn one sample and return the a-priori prediction error."""
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        if not np.isfinite(denom) or denom <= 1e-12:
            self.P = np.eye(self.n_inputs) * self.delta
            return y - self.predict(x)
        k = Px / denom
        err = y - x @ self.W
        self.W = self.W + np.outer(k, err)
        P = (self.P - np.outer(k, Px)) / self.lam
        P = 0.5 * (P + P.T)
        trace = float(np.trace(P))
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(self.W)):
            log.warning("RLS numerical reset")
            self.P = np.eye(self.n_inputs) * self.delta
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if trace > self.max_trace:          # covariance wind-up guard
                P *= self.max_trace / trace
            self.P = P
        return err


class EwmStats:
    """Exponentially weighted mean / variance with a cumulative warm start."""

    def __init__(self, dim: int, alpha: float):
        self.alpha = alpha
        self.n = 0
        self.mean = np.zeros(dim)
        self.var = np.zeros(dim)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        a = max(1.0 / self.n, self.alpha)
        d = x - self.mean
        self.mean = self.mean + a * d
        self.var = (1.0 - a) * (self.var + a * d * d)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


@dataclass(frozen=True)
class Detection:
    """Result of processing one tick."""

    tick: int
    status: str                          # calibrating | learning | monitoring
    prediction: Optional[np.ndarray]     # one-step forecast in raw units
    z: np.ndarray                        # residual z-score per sensor
    iso_score: float                     # normalised Isolation Forest score
    env_score: float                     # normalised operating-envelope excursion
    score: float                         # combined anomaly score (>=1 -> flagged)
    flagged: bool
    alert: bool


class OnlineAnomalyDetector:
    """Streaming detector combining an RLS forecaster and an Isolation Forest."""

    def __init__(self, n_sensors: int, cfg: DetectorConfig = DetectorConfig(), seed: int = 0):
        if n_sensors < 1:
            raise ValueError("n_sensors must be >= 1")
        self.S = n_sensors
        self.cfg = cfg
        self._seed = seed
        self.tick = 0
        self._scaler = StandardScaler()
        self._mean = np.zeros(n_sensors)
        self._scale = np.ones(n_sensors)
        self._frozen = False
        self.rls = RecursiveLeastSquares(1 + cfg.lags * n_sensors, n_sensors,
                                         cfg.forgetting, cfg.p_init)
        self._hist: Deque[np.ndarray] = deque(maxlen=cfg.lags)
        self._stats = EwmStats(n_sensors, cfg.resid_alpha)
        self._zhist: Deque[np.ndarray] = deque(maxlen=cfg.roll)
        self._buf: Deque[np.ndarray] = deque(maxlen=cfg.iso_window)
        self._iso: Optional[Tuple[IsolationForest, float, float]] = None
        self._env: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._fits = 0
        self._last_fit = 0
        self._pred_ticks = 0
        self._consec_flag = 0
        self._consec_clear = 0
        self._alert = False
        self._alert_run = 0
        self._last_raw: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- helpers
    def _validate(self, raw) -> np.ndarray:
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.shape != (self.S,):
            raise ValueError(f"expected {self.S} readings, got {arr.shape[0]}")
        bad = ~np.isfinite(arr)
        if bad.any():                       # forward-fill missing / corrupt readings
            if self._last_raw is None:
                raise ValueError("non-finite reading before any valid sample")
            arr = np.where(bad, self._last_raw, arr)
        self._last_raw = arr.copy()
        return arr

    def _design(self) -> np.ndarray:
        lagged = np.concatenate(list(reversed(self._hist)))
        return np.concatenate(([1.0], lagged))

    def _fit_iso(self, tick: int) -> None:
        X = np.vstack(self._buf)
        model = IsolationForest(n_estimators=self.cfg.iso_trees,
                                max_samples=min(256, len(X)), contamination="auto",
                                random_state=self._seed + self._fits, n_jobs=1)
        model.fit(X)
        scores = -model.score_samples(X)
        med = float(np.median(scores))
        thr = float(np.quantile(scores, self.cfg.iso_quantile))
        thr = max(thr, med + 1e-3)
        self._iso = (model, med, thr)
        q = self.cfg.envelope_quantile
        vals = X[:, :self.S]
        self._env = (np.quantile(vals, q, axis=0), np.quantile(vals, 1.0 - q, axis=0))
        self._fits += 1
        self._last_fit = tick

    # ------------------------------------------------------------------- main
    def update(self, raw) -> Detection:
        """Process one reading and return the detection result."""
        raw = self._validate(raw)
        cfg = self.cfg
        tick = self.tick
        self.tick += 1
        zeros = np.zeros(self.S)

        if tick < cfg.scaler_ticks:                       # phase 1: calibrate scaler
            self._scaler.partial_fit(raw.reshape(1, -1))
            return Detection(tick, "calibrating", None, zeros, 0.0, 0.0, 0.0, False, False)
        if not self._frozen:
            self._mean = self._scaler.mean_.copy()
            self._scale = np.maximum(self._scaler.scale_.copy(), 1e-8)
            self._frozen = True

        scaled = (raw - self._mean) / self._scale
        x = pred_scaled = None
        resid = zeros
        if len(self._hist) == cfg.lags:
            x = self._design()
            pred_scaled = self.rls.predict(x)
            resid = scaled - pred_scaled
            self._pred_ticks += 1

        stats_valid = self._stats.n >= cfg.min_stats
        if stats_valid and pred_scaled is not None:
            z = (resid - self._stats.mean) / np.maximum(self._stats.std, 1e-3)
        else:
            z = zeros
        if pred_scaled is not None:
            self._zhist.append(np.abs(z))
        roll = np.mean(self._zhist, axis=0) if self._zhist else zeros
        feat = np.concatenate([scaled, np.clip(z, -20, 20), np.minimum(roll, 20)])

        monitoring = tick >= cfg.warmup_ticks and pred_scaled is not None and stats_valid
        if monitoring:                                    # (re)fit Isolation Forest
            due = self._iso is None or (tick - self._last_fit) >= cfg.retrain_every
            if due and len(self._buf) >= cfg.min_iso_samples:
                self._fit_iso(tick)

        iso_norm = env_norm = 0.0
        if monitoring and self._iso is not None and self._env is not None:
            model, med, thr = self._iso
            s = float(-model.score_samples(feat.reshape(1, -1))[0])
            iso_norm = max(0.0, (s - med) / ((thr - med) * cfg.iso_margin))
            lo, hi = self._env
            width = np.maximum(hi - lo, 1e-6)
            excursion = np.maximum(np.maximum(lo - scaled, scaled - hi), 0.0)
            env_norm = float(np.max(excursion / (cfg.envelope_margin * width)))
        z_ratio = float(np.max(np.abs(z))) / cfg.z_threshold if (monitoring and stats_valid) else 0.0
        active = monitoring and self._iso is not None
        score = max(z_ratio, iso_norm, env_norm) if active else 0.0
        flagged = bool(active and score >= 1.0)

        # alert state machine
        if flagged:
            self._consec_flag += 1
            self._consec_clear = 0
        else:
            self._consec_flag = 0
            self._consec_clear += 1
        strong = flagged and score >= cfg.strong_score
        if not self._alert and (self._consec_flag >= cfg.min_consecutive or strong):
            self._alert, self._alert_run = True, 0
        elif self._alert and self._consec_clear >= cfg.cooldown:
            self._alert, self._alert_run = False, 0
        if self._alert:
            self._alert_run += 1

        # learning (skip anomalous ticks so they do not poison the baseline)
        learn = (not (flagged or self._alert)) or self._alert_run > cfg.max_alert_run
        if learn and pred_scaled is not None:
            self.rls.update(x, scaled)
            if self._pred_ticks > cfg.burn_in:
                if stats_valid:
                    self._buf.append(feat)
                self._stats.update(resid)
        self._hist.append(scaled)

        prediction = None if pred_scaled is None else pred_scaled * self._scale + self._mean
        status = "monitoring" if active else "learning"
        return Detection(tick, status, prediction, np.asarray(z, dtype=float),
                         float(iso_norm), float(env_norm), float(score), flagged,
                         bool(self._alert))
