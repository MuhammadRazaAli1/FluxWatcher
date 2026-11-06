#!/usr/bin/env python3
"""StreamSentinel entry point.

  python run.py                    live dashboard in your browser
  python run.py --headless         fast offline evaluation + report
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from streamsentinel import __version__
from streamsentinel.config import AppConfig, SimConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time multivariate anomaly detection.")
    p.add_argument("--version", action="version", version=f"StreamSentinel {__version__}")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="first port to try (default: 8000)")
    p.add_argument("--seed", type=int, default=42, help="random seed of the simulator")
    p.add_argument("--interval", type=float, default=0.05,
                   help="seconds per tick at 1x speed (default: 0.05 = 20 Hz)")
    p.add_argument("--anomaly-rate", type=float, default=0.003,
                   help="probability per tick that an anomaly starts (default: 0.003)")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.add_argument("--headless", action="store_true", help="no server: process a batch and report")
    p.add_argument("--ticks", type=int, default=3000, help="ticks to process in --headless mode")
    p.add_argument("--csv", metavar="FILE", help="(headless) save per-tick results as CSV")
    p.add_argument("--plot", metavar="FILE", help="(headless) save an overview plot (needs matplotlib)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = AppConfig(sim=SimConfig(seed=args.seed, anomaly_rate=args.anomaly_rate),
                        tick_interval=args.interval)
    except ValueError as exc:
        print(f"Invalid settings: {exc}", file=sys.stderr)
        return 2

    if args.headless:
        from streamsentinel.headless import format_report, run_headless, save_plot
        if args.ticks < 1:
            print("--ticks must be >= 1", file=sys.stderr)
            return 2
        print(f"Processing {args.ticks} ticks (seed={args.seed}) ...", flush=True)
        df, metrics = run_headless(cfg, args.ticks)
        print(format_report(metrics))
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"saved {args.csv}")
        if args.plot:
            try:
                names = [c for c in df.columns if f"{c}_forecast" in df.columns]
                save_plot(df, names, args.plot)
                print(f"saved {args.plot}")
            except ImportError:
                print("matplotlib is not installed: pip install matplotlib", file=sys.stderr)
                return 1
        return 0

    from streamsentinel.engine import StreamEngine
    from streamsentinel.server import serve
    try:
        serve(StreamEngine(cfg), host=args.host, port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print(f"Could not start the server: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
