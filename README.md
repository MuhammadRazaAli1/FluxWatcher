<div align="center">

# FluxWatcher

### Real-time multivariate anomaly detection for industrial sensor streams

Online-learning detection engine with a live monitoring dashboard.
Spot faults in machine data the moment they appear, with no labelled training data required.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-2ea44f?style=for-the-badge)](https://flux-watcher-rlr38x3tm-muhammad-raza-ali1.vercel.app/)

![JavaScript](https://img.shields.io/badge/JavaScript-engine-F7DF1E?logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?logo=vercel&logoColor=white)

</div>

---

## Live Demo

**[https://flux-watcher-rlr38x3tm-muhammad-raza-ali1.vercel.app/](https://flux-watcher-rlr38x3tm-muhammad-raza-ali1.vercel.app/)**

The demo runs entirely in your browser. It streams simulated factory sensors in real time, learns what "normal" looks like on the fly, and raises alerts when something deviates. You can inject faults, change sensitivity, pause or speed up the stream, and analyse your own CSV file locally (nothing is uploaded).

---

## Overview

Industrial machines produce continuous sensor data (temperature, vibration, pressure, current). Faults rarely announce themselves: a bearing starts to wear, a valve drifts, a sensor jumps. Watching every signal by hand does not scale, and labelled failure data is usually unavailable.

FluxWatcher learns the normal behaviour of a machine **online**, directly from its own data stream, and flags readings that do not fit, together with the sensor most likely responsible.

## Key Features

| Feature | Description |
|---|---|
| **Real-time detection** | Processes every reading as it arrives (20 readings per second in the demo) |
| **Online learning** | No offline training step; the baseline is learned from the first ~400 normal readings and keeps adapting |
| **Three complementary detectors** | Forecasting residuals, Isolation Forest, and an operating-envelope check, combined into one anomaly score |
| **Poisoning protection** | The models do not learn from readings flagged as anomalous, so faults do not corrupt the baseline |
| **Explainable alerts** | Each alert reports its score, the dominant reason, and the suspect sensors |
| **Live metrics** | Precision, recall, F1, detection delay and false-alert counts against known faults |
| **Bring your own data** | Load a CSV export from a SCADA or historian system; missing values, text columns and regional number formats are handled |
| **Alert export** | Download the full alert log as CSV |
| **Sensitivity control** | Low, Normal and High presets to trade fewer false alarms against earlier warnings |
| **Privacy by design** | The web edition processes data locally in the browser |

## How It Works

```
 Sensor stream ──► Scaler ──► RLS forecaster ──► residual z-score ─┐
 (simulator or                                                      │
  your CSV)                 ──► Isolation Forest (sliding window) ──┼──► combined score ──► alert
                                                                    │     state machine
                            ──► Operating-envelope check ───────────┘
```

1. **Forecasting residuals.** A multi-output recursive-least-squares model predicts each sensor's next value from the recent history of all sensors. A large, unexpected error signals a spike or a sudden change.
2. **Isolation Forest.** Periodically refitted on a sliding window of clean data, it scores how unusual the combined state of all sensors is.
3. **Operating envelope.** Compares each reading with the range seen during normal operation. This catches sustained level shifts and slow drifts that the forecaster would otherwise adapt to.
4. **Alert logic.** The three scores are normalised and combined. An alert is raised after consecutive flagged readings (or immediately for a very strong one) and cleared after a cool-down period.

## Detected Fault Types

| Fault | Description |
|---|---|
| **Spike** | A short, sharp jump in one or more sensors |
| **Level shift** | A sudden, sustained offset from the normal level |
| **Drift** | A gradual deviation that builds up over time |
| **Variance burst** | A period of unusually noisy readings |

## Evaluation

Measured on seeded simulations with known, injected faults. The simulator and the detector use independent random streams, and results are reproducible.

| Metric | Result |
|---|---|
| Fault episodes detected (browser engine, 5 simulation runs of 3,500 readings) | **38 of 38** |
| False alert events in those runs | **0** |
| Alert events on 20,000 readings of fault-free data (8 runs) | **0** |
| Fault episodes detected (Python engine, 5 runs) | 41 of 43 (about 95%) |
| Automated tests | 47 Python tests, 13 JavaScript engine tests, and a full dashboard smoke test |

> **Note on interpretation.** Results come from simulated data. Because the models adapt to a sustained fault, point-by-point recall is lower (roughly 35-60%) than episode-level detection; the system is best judged by whether each fault episode is caught and how quickly. Real plants have operating modes, start-ups and shutdowns that need tuning, and a pilot in monitoring-only mode is recommended before acting on alerts.

## Use Cases

- Predictive maintenance for motors, pumps, compressors and fans
- Condition monitoring of production lines
- Early warning from utility and process sensors (temperature, pressure, flow)
- Quality control of IoT and gateway data streams
- Teaching and demonstrating streaming machine-learning concepts

## Architecture

FluxWatcher ships in two editions that share the same algorithms:

| Edition | Purpose | Runs on |
|---|---|---|
| **Web edition** (`index.html` + `engine.js`) | Interactive demo and local CSV analysis, dependency-free | Any browser; hosted on Vercel |
| **Python edition** (`streamsentinel/`) | Long-running service with a built-in dashboard, alert log and webhook, and data-source adapters (CSV, Modbus TCP, OPC UA, MQTT) | A factory PC or server |

Both editions are covered by automated tests (`tests/` for Python; a JavaScript engine suite and a dashboard smoke test for the web edition).

## Tech Stack

- **Algorithms:** recursive least squares, Isolation Forest, exponentially weighted statistics, exponential-smoothing style state machine
- **Web edition:** vanilla JavaScript (no framework, no dependencies), HTML5 canvas charts
- **Python edition:** NumPy, pandas, scikit-learn, standard-library HTTP server
- **Hosting:** Vercel (static)

## Limitations

- Validated on simulated data; thresholds should be tuned on real plant data.
- FluxWatcher is a decision-support tool. It is **not** a safety system and must not replace protection relays, interlocks or emergency stops.
- The web edition runs only while the browser tab is open and cannot reach private plant networks; live PLC, OPC UA and MQTT connections belong to the Python edition, whose adapters have not been verified against real hardware.

## Author

**Muhammad Raza Ali** — [@MuhammadRazaAli1](https://github.com/MuhammadRazaAli1)
