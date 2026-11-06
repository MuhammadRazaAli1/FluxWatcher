# FluxWatcher

Real-time multivariate anomaly detection for streaming sensor data, with online
learning, a live dashboard and a reproducible evaluation harness.
Runs on numpy, pandas and scikit-learn only. No database, no web framework, no
internet connection needed.

## Quick start

```bash
pip install -r requirements.txt        # Python 3.9+
python run.py                          # opens http://localhost:8000
```

The first ~400 ticks (about 20 s) are the learning phase. After that the system is
"Monitoring". Click **Inject anomaly** to watch it react in real time.

Headless evaluation (no browser):

```bash
python run.py --headless --ticks 3000 --csv results.csv --plot overview.png
```

(`--plot` needs `pip install matplotlib`.)

Tests:

```bash
python -m unittest discover -s tests -t .
```

## How it works

```
 SensorStream --> OnlineAnomalyDetector --> OnlineMetrics
 (simulator)      |  1. scaler (calibrated, then frozen)
                  |  2. RLS forecaster -> residual z-score
                  |  3. Isolation Forest on [values, z, rolling |z|]
                  |  4. operating-envelope check
                  |  5. score = max(...); alert state machine
        StreamEngine (thread, lock)  -->  HTTP server  -->  dashboard
```

* The forecaster learns the dynamics online (recursive least squares with forgetting)
  and flags readings that surprise it.
* The Isolation Forest is refitted every 300 ticks on a window of clean data.
* The envelope check catches sustained shifts and drifts the forecaster adapts to.
* A tick is flagged at score >= 1. An alert needs 2 consecutive flags, or one tick at
  score >= 1.5, and clears after 3 clean ticks.
* Models do not learn from flagged ticks, so anomalies do not poison the baseline.

## Measured behaviour (simulated data)

About 95% of anomaly episodes are detected, with roughly 0-1 false alert events per
10,000 normal ticks. Point-wise recall is only ~35-60%, because the models adapt to a
sustained shift and stop flagging it; judge the system by episode detection.
Exact numbers vary by seed: run `python run.py --headless --seed N`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state?since=T` | new points after tick T + metrics |
| GET | `/api/health` | liveness |
| POST | `/api/inject` `{"kind": "spike"}` | kinds: spike, level_shift, drift, variance_burst |
| POST | `/api/pause`, `/api/resume` | control the stream |
| POST | `/api/speed` `{"speed": 2}` | 0.25 to 4 |

## Using your own data

`StreamEngine(cfg, source=my_source)` accepts any object with `sensor_names` and a
`next()` returning `streamsentinel.simulator.Sample` (set `label=0` if you have no
ground truth). Remember to match the detector's sensor count.

## Troubleshooting

* **Port busy**: the server tries the next 20 ports and prints the URL it chose.
* **Browser did not open**: use `--no-browser` and open the printed URL yourself.
* **Slow machine**: use `--interval 0.1` for 10 ticks per second.

## Known limitations

Built and tested on simulated data; real sensors will need threshold tuning
(`DetectorConfig`). The dashboard was verified with a scripted DOM test, not on every
browser. The default server binds to localhost only and has no authentication.
