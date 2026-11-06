"""Run the pipeline as fast as possible (no server) and report the results."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import AppConfig
from .engine import StreamEngine


def run_headless(cfg: AppConfig, ticks: int) -> "tuple[pd.DataFrame, dict]":
    """Process ``ticks`` ticks synchronously. Returns (per-tick DataFrame, metrics)."""
    if ticks < 1:
        raise ValueError("ticks must be >= 1")
    engine = StreamEngine(cfg)
    names = engine.names
    rows = []
    for _ in range(ticks):
        r = engine.step()
        row = {"t": r["t"], "score": r["s"], "flagged": r["f"], "alert": r["a"],
               "label": r["y"], "kind": r["k"], "status": r["st"]}
        for i, n in enumerate(names):
            row[n] = r["v"][i]
            row[f"{n}_forecast"] = None if r["p"] is None else r["p"][i]
            row[f"{n}_z"] = r["z"][i]
        rows.append(row)
    return pd.DataFrame(rows), engine.metrics.snapshot()


def format_report(metrics: dict) -> str:
    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{100 * x:5.1f}%"

    delay = metrics["mean_delay"]
    lines = [
        "StreamSentinel - evaluation report",
        "-" * 38,
        f"ticks processed        : {metrics['ticks']}",
        f"anomaly episodes       : {metrics['episodes']}",
        f"  detected             : {metrics['detected']}  ({pct(metrics['event_recall']).strip()})",
        f"  mean detection delay : {'n/a' if delay is None else f'{delay:.1f} ticks'}",
        f"alert events           : {metrics['alert_events']}",
        f"  false alert events   : {metrics['false_alert_events']}",
        f"point precision        : {pct(metrics['precision'])}",
        f"point recall           : {pct(metrics['recall'])}",
        f"point F1               : {pct(metrics['f1'])}",
        f"confusion (tp/fp/fn/tn): {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}",
    ]
    return "\n".join(lines)


def save_plot(df: pd.DataFrame, names, path: str) -> None:
    """Save an overview PNG (requires matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    fig, axes = plt.subplots(n + 1, 1, figsize=(13, 2.4 * (n + 1)), sharex=True)
    for ax, name in zip(axes[:-1], names):
        ax.plot(df["t"], df[name], lw=0.8, label=name)
        ax.plot(df["t"], df[f"{name}_forecast"], lw=0.6, alpha=0.6, label="forecast")
        _shade(ax, df)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=7)
    ax = axes[-1]
    ax.plot(df["t"], df["score"], lw=0.8, color="tab:purple")
    ax.axhline(1.0, color="tab:red", ls="--", lw=0.8)
    _shade(ax, df)
    ax.set_ylabel("anomaly score")
    ax.set_xlabel("tick")
    fig.suptitle("StreamSentinel  (red = alert, green strip = ground truth)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _shade(ax, df: pd.DataFrame) -> None:
    ax.fill_between(df["t"], 0, 1, where=df["alert"] > 0, color="tab:red", alpha=0.18,
                    transform=ax.get_xaxis_transform(), step="mid")
    ax.fill_between(df["t"], 0, 0.06, where=df["label"] > 0, color="tab:green", alpha=0.8,
                    transform=ax.get_xaxis_transform(), step="mid")
