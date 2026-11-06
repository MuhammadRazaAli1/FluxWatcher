"""Typed, validated configuration for every component of StreamSentinel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

ANOMALY_KINDS: Tuple[str, ...] = ("spike", "level_shift", "drift", "variance_burst")


@dataclass(frozen=True)
class SensorSpec:
    """Statistical description of one simulated sensor."""

    name: str
    unit: str
    base: float          # mean operating level
    amp: float           # amplitude of the seasonal cycle
    period: float        # seasonal period in ticks
    noise: float         # innovation std of the AR(1) measurement noise
    coupling: float      # sensitivity to the shared latent "load" factor
    phase: float = 0.0   # seasonal phase offset (radians)

    @property
    def noise_std(self) -> float:
        """Marginal std of the AR(1) noise (phi = 0.6 -> 1.25 * innovation std)."""
        return self.noise * 1.25


DEFAULT_SENSORS: Tuple[SensorSpec, ...] = (
    SensorSpec("Temperature", "°C", 70.0, 4.0, 100.0, 0.35, 0.8, 0.0),
    SensorSpec("Vibration", "mm/s", 2.0, 0.4, 64.0, 0.05, 0.05, 1.3),
    SensorSpec("Pressure", "kPa", 101.0, 3.0, 128.0, 0.30, 0.5, 2.1),
)


@dataclass(frozen=True)
class SimConfig:
    """Settings of the synthetic sensor stream."""

    seed: int = 42
    sensors: Tuple[SensorSpec, ...] = DEFAULT_SENSORS
    clean_ticks: int = 450           # no anomalies before this tick (model warm-up)
    anomaly_rate: float = 0.003      # chance per tick to start an episode
    kind_weights: Tuple[float, ...] = (0.35, 0.25, 0.20, 0.20)
    ar_phi: float = 0.6              # AR(1) coefficient of measurement noise

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ValueError("at least one sensor is required")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError("anomaly_rate must be within [0, 1]")
        if len(self.kind_weights) != len(ANOMALY_KINDS) or min(self.kind_weights) < 0 \
                or sum(self.kind_weights) <= 0:
            raise ValueError("kind_weights must hold one non-negative weight per anomaly kind")
        if self.clean_ticks < 0:
            raise ValueError("clean_ticks must be >= 0")
        if not 0.0 <= self.ar_phi < 1.0:
            raise ValueError("ar_phi must be within [0, 1)")


@dataclass(frozen=True)
class DetectorConfig:
    """Hyper-parameters of the online anomaly detector."""

    scaler_ticks: int = 60        # ticks used to calibrate the (then frozen) scaler
    warmup_ticks: int = 400       # ticks before alerting starts
    lags: int = 6                 # autoregressive lags used by the forecaster
    forgetting: float = 0.999     # RLS forgetting factor
    p_init: float = 100.0         # RLS initial covariance scale
    burn_in: int = 80             # forecast ticks ignored while RLS converges
    resid_alpha: float = 0.01     # EWM smoothing of residual statistics
    min_stats: int = 30           # residual samples needed before z-scores are valid
    z_threshold: float = 4.5      # |z| of residual that counts as anomalous
    roll: int = 5                 # window of the rolling |z| feature
    iso_window: int = 600         # clean feature vectors kept for IsolationForest
    iso_trees: int = 60
    iso_quantile: float = 0.997   # training-score quantile used as alert threshold
    iso_margin: float = 1.3       # safety factor on the Isolation Forest threshold
    retrain_every: int = 300      # ticks between IsolationForest refits
    min_iso_samples: int = 200
    envelope_quantile: float = 0.005   # operating range = [q, 1-q] of clean readings
    envelope_margin: float = 0.30      # excursion (fraction of range) that flags a reading
    strong_score: float = 1.5     # a single tick this anomalous raises an alert at once
    min_consecutive: int = 2      # flagged ticks required to raise an alert
    cooldown: int = 3             # clear ticks required to drop an alert
    max_alert_run: int = 80       # after this many alert ticks the model re-adapts

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise ValueError("lags must be >= 1")
        if not 0.9 < self.forgetting <= 1.0:
            raise ValueError("forgetting must be within (0.9, 1]")
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be > 0")
        if not 0.5 < self.iso_quantile < 1.0:
            raise ValueError("iso_quantile must be within (0.5, 1)")
        if not 0.0 < self.envelope_quantile < 0.5 or self.envelope_margin <= 0:
            raise ValueError("invalid envelope settings")
        if self.iso_margin < 1.0:
            raise ValueError("iso_margin must be >= 1")
        if self.strong_score < 1.0:
            raise ValueError("strong_score must be >= 1")
        if self.min_consecutive < 1 or self.cooldown < 1:
            raise ValueError("min_consecutive and cooldown must be >= 1")
        needed = self.scaler_ticks + self.lags + self.burn_in + self.min_stats
        if self.warmup_ticks < needed:
            raise ValueError(f"warmup_ticks must be >= {needed} for these settings")
        if self.min_iso_samples > self.iso_window:
            raise ValueError("min_iso_samples cannot exceed iso_window")


@dataclass(frozen=True)
class AppConfig:
    """Top level configuration."""

    sim: SimConfig = field(default_factory=SimConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tick_interval: float = 0.05      # seconds between ticks at 1x speed (20 Hz)
    history: int = 1200              # records kept in memory for the dashboard
    metrics_grace: int = 10          # ticks after an episode ignored by point metrics

    def __post_init__(self) -> None:
        if self.tick_interval <= 0:
            raise ValueError("tick_interval must be > 0")
        if self.history < 50:
            raise ValueError("history must be >= 50")
        if self.sim.clean_ticks < self.detector.warmup_ticks:
            raise ValueError("sim.clean_ticks must be >= detector.warmup_ticks so the "
                             "baseline is learned from normal data")
