/* StreamSentinel - browser/Node edition of the real-time anomaly detection engine.
 * Same algorithms as the Python package: RLS forecaster + Isolation Forest + operating
 * envelope, online learning, alert state machine, live metrics. No dependencies.
 */
(function (root) {
"use strict";

// ------------------------------------------------------------------ config
const SENSOR_SPECS = [
  { name: "Temperature", unit: "°C", base: 70.0, amp: 4.0, period: 100, noise: 0.35, coupling: 0.8, phase: 0.0 },
  { name: "Vibration", unit: "mm/s", base: 2.0, amp: 0.4, period: 64, noise: 0.05, coupling: 0.05, phase: 1.3 },
  { name: "Pressure", unit: "kPa", base: 101.0, amp: 3.0, period: 128, noise: 0.30, coupling: 0.5, phase: 2.1 },
];
const KINDS = ["spike", "level_shift", "drift", "variance_burst"];
const DEFAULT_SIM = { seed: 42, cleanTicks: 450, anomalyRate: 0.003, kindWeights: [0.35, 0.25, 0.20, 0.20], arPhi: 0.6 };
const DEFAULT_DET = {
  scalerTicks: 60, warmupTicks: 400, lags: 6, forgetting: 0.999, pInit: 100.0, burnIn: 80,
  residAlpha: 0.01, minStats: 30, zThreshold: 4.5, roll: 5, isoWindow: 600, isoTrees: 60,
  isoQuantile: 0.997, isoMargin: 1.3, retrainEvery: 300, minIsoSamples: 200,
  envQuantile: 0.005, envMargin: 0.30, strongScore: 1.5, minConsecutive: 2, cooldown: 3, maxAlertRun: 80,
};
const SENSITIVITY = {
  low: { zThreshold: 6.0, isoMargin: 1.8, envMargin: 0.45 },
  normal: {},
  high: { zThreshold: 3.5, isoMargin: 1.0, envMargin: 0.20 },
};
const MIN_SPEED = 0.25, MAX_SPEED = 4.0, MAX_CATCHUP = 60, MAX_ERRORS = 10;

function detectorConfig(preset, over) {
  if (preset !== undefined && preset !== null && !(preset in SENSITIVITY)) throw new Error("unknown sensitivity: " + preset);
  const c = Object.assign({}, DEFAULT_DET, SENSITIVITY[preset || "normal"], over || {});
  const need = c.scalerTicks + c.lags + c.burnIn + c.minStats;
  if (c.warmupTicks < need) throw new Error("warmupTicks must be >= " + need);
  if (c.minIsoSamples > c.isoWindow) throw new Error("minIsoSamples cannot exceed isoWindow");
  return c;
}

class SourceExhausted extends Error { constructor(m) { super(m || "end of data"); this.name = "SourceExhausted"; } }
class SourceError extends Error { constructor(m) { super(m); this.name = "SourceError"; } }

// -------------------------------------------------------------------- RNG
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
class Rng {
  constructor(seed) { this._u = mulberry32(seed >>> 0); this._spare = null; }
  uniform(a, b) { return a === undefined ? this._u() : a + (b - a) * this._u(); }
  normal() {
    if (this._spare !== null) { const s = this._spare; this._spare = null; return s; }
    let u = 0; while (u === 0) u = this._u();
    const v = this._u(), r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  int(lo, hi) { return lo + Math.floor(this._u() * (hi - lo)); }           // [lo, hi)
  weighted(p) { const r = this._u(); let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (r < c) return i; } return p.length - 1; }
  sample(n, k) {                                                           // k distinct of [0, n)
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) { const j = i + Math.floor(this._u() * (n - i)); const t = a[i]; a[i] = a[j]; a[j] = t; }
    return a.slice(0, k);
  }
}

// -------------------------------------------------------------- simulator
class Episode {
  constructor(kind, sensors, length, mag, signs, noiseScale) {
    this.kind = kind; this.sensors = sensors; this.length = length; this.mag = mag;
    this.signs = signs; this.noiseScale = noiseScale; this.i = 0;
  }
  perturb(values, rng) {
    for (let j = 0; j < this.sensors.length; j++) {
      const s = this.sensors[j];
      if (this.kind === "spike" || this.kind === "level_shift") values[s] += this.signs[j] * this.mag[j];
      else if (this.kind === "drift") values[s] += this.signs[j] * this.mag[j] * (this.i + 1) / this.length;
      else values[s] += rng.normal() * this.noiseScale[j];
    }
    this.i++;
  }
}

class SimSource {
  constructor(opts) {
    this.cfg = Object.assign({}, DEFAULT_SIM, opts || {});
    this.specs = SENSOR_SPECS;
    this.sensorNames = this.specs.map(s => s.name);
    this.units = this.specs.map(s => s.unit);
    this.hasLabels = true; this.loop = false;
    this.reset();
  }
  reset() {
    this.rng = new Rng(this.cfg.seed);
    this.t = 0; this.load = 0; this.episode = null; this.pending = null;
    this.noise = new Float64Array(this.specs.length);
    const w = this.cfg.kindWeights, sum = w.reduce((a, b) => a + b, 0);
    this.kindP = w.map(x => x / sum);
  }
  inject(kind) {
    if (kind !== null && kind !== undefined && kind !== "" && KINDS.indexOf(kind) < 0) return [false, "unknown anomaly kind '" + kind + "'"];
    if (this.episode !== null || this.pending !== null) return [false, "an anomaly is already in progress"];
    if (this.t < this.cfg.cleanTicks) return [false, "the system is still learning its baseline"];
    this.pending = kind ? kind : "random";
    return [true, "anomaly injected"];
  }
  next() {
    const cfg = this.cfg, rng = this.rng, n = this.specs.length, t = this.t;
    this.load = 0.98 * this.load + 0.15 * rng.normal();
    for (let i = 0; i < n; i++) this.noise[i] = cfg.arPhi * this.noise[i] + this.specs[i].noise * rng.normal();
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const s = this.specs[i];
      values[i] = s.base + s.amp * Math.sin(2 * Math.PI * t / s.period + s.phase) + s.coupling * this.load + this.noise[i];
    }
    if (this.episode === null && t >= cfg.cleanTicks) {
      const requested = this.pending; this.pending = null;
      if (requested !== null) this.episode = this._newEpisode(requested === "random" ? null : requested);
      else if (rng.uniform() < cfg.anomalyRate) this.episode = this._newEpisode(null);
    }
    let label = 0, kind = "", affected = [];
    const ep = this.episode;
    if (ep !== null) {
      ep.perturb(values, rng);
      label = 1; kind = ep.kind; affected = ep.sensors.slice();
      if (ep.i >= ep.length) this.episode = null;
    }
    this.t++;
    return { t: t, values: values, label: label, kind: kind, sensors: affected, ts: null };
  }
  _newEpisode(kind) {
    const rng = this.rng, n = this.specs.length;
    if (kind === null) kind = KINDS[rng.weighted(this.kindP)];
    const nAff = (n === 1 || rng.uniform() < 0.75) ? 1 : 2;
    const sensors = rng.sample(n, nAff);
    const signs = sensors.map(() => (rng.uniform() < 0.5 ? -1 : 1));
    const specs = sensors.map(i => this.specs[i]);
    let length, mag;
    if (kind === "spike") { length = rng.int(3, 6); mag = specs.map(s => rng.uniform(8, 12) * s.noise * 1.25); }
    else if (kind === "level_shift") { length = rng.int(30, 61); mag = specs.map(s => rng.uniform(1.5, 2.5) * s.amp); }
    else if (kind === "drift") { length = rng.int(60, 121); mag = specs.map(s => rng.uniform(2.0, 3.5) * s.amp); }
    else { length = rng.int(20, 41); mag = specs.map(() => 0); }
    const noiseScale = specs.map(s => rng.uniform(5, 8) * s.noise);
    return new Episode(kind, sensors, length, mag, signs, noiseScale);
  }
}

// ------------------------------------------------------------ CSV source
function parseCsvText(text) {
  if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
  const firstLine = text.split(/\r?\n/, 1)[0] || "";
  const counts = { ",": 0, ";": 0, "\t": 0, "|": 0 };
  let inQ = false;
  for (const ch of firstLine) { if (ch === '"') inQ = !inQ; else if (!inQ && ch in counts) counts[ch]++; }
  let delim = ","; let best = -1;
  for (const d of Object.keys(counts)) if (counts[d] > best) { best = counts[d]; delim = d; }
  const rows = []; let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === delim) { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.length > 1 || row[0].trim() !== "") rows.push(row);
      row = [];
    } else cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); if (row.length > 1 || row[0].trim() !== "") rows.push(row); }
  return { delim: delim, rows: rows };
}

class CsvSource {
  /** Replays a CSV text row by row. Rows are processed in FILE ORDER (timestamps are only displayed). */
  constructor(text, opts) {
    opts = opts || {};
    this.loop = !!opts.loop; this.hasLabels = false; this.warnings = []; this.name = opts.name || "data.csv";
    const parsed = parseCsvText(String(text));
    if (parsed.rows.length < 3) throw new SourceError("the file needs a header row and at least 2 data rows");
    const header = parsed.rows[0].map(h => h.trim());
    const body = parsed.rows.slice(1);
    const nCols = header.length;
    const timeIdx = header.findIndex(h => /^(date[ _-]?time|timestamp|time|date|ts)$/i.test(h));
    const labelIdx = header.findIndex(h => /^(label|fault|is_?anomaly|anomaly)$/i.test(h));
    const num = (s) => {
      if (s === undefined) return NaN;
      s = s.trim();
      if (s === "") return NaN;
      if (parsed.delim === ";" && /^-?\d+,\d+$/.test(s)) s = s.replace(",", ".");
      const v = Number(s);
      return Number.isFinite(v) ? v : NaN;
    };
    const cols = []; const names = [];
    for (let c = 0; c < nCols; c++) {
      if (c === timeIdx || c === labelIdx) continue;
      const vals = body.map(r => num(r[c]));
      const ok = vals.filter(v => !Number.isNaN(v)).length;
      if (ok / body.length < 0.5) { this.warnings.push("column '" + header[c] + "' skipped (not numeric)"); continue; }
      const miss = body.length - ok;
      if (miss) this.warnings.push("column '" + header[c] + "': " + miss + " missing values were forward-filled");
      let first = vals.findIndex(v => !Number.isNaN(v));
      for (let i = 0; i < first; i++) vals[i] = vals[first];
      for (let i = first + 1; i < vals.length; i++) if (Number.isNaN(vals[i])) vals[i] = vals[i - 1];
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      if (vals.every(v => v === vals[0])) this.warnings.push("column '" + header[c] + "' is constant - it carries no information");
      cols.push(vals); names.push(header[c]);
    }
    if (!cols.length) throw new SourceError("no numeric sensor columns found");
    if (body.length < 600) this.warnings.push("only " + body.length + " rows: the detector needs ~400 normal rows to learn the baseline (>= 600 recommended)");
    this._values = body.map((_, i) => Float64Array.from(cols.map(c => c[i])));
    this._ts = timeIdx >= 0 ? body.map(r => (r[timeIdx] || "").trim()) : null;
    if (labelIdx >= 0) { this._labels = body.map(r => (num(r[labelIdx]) > 0 ? 1 : 0)); this.hasLabels = true; } else this._labels = null;
    this.sensorNames = names; this.units = names.map(() => "");
    this.rows = body.length; this.delimiter = parsed.delim;
    this.reset();
  }
  reset() { this._i = 0; this._t = 0; }
  describe() {
    return "Loaded " + this.rows + " rows, " + this.sensorNames.length + " sensors: " + this.sensorNames.join(", ") +
      " | labels: " + (this.hasLabels ? "yes" : "no (alerts only)") + " | rows are processed in file order";
  }
  next() {
    if (this._i >= this.rows) { if (!this.loop) throw new SourceExhausted(); this._i = 0; }
    const label = this._labels ? this._labels[this._i] : 0;
    const s = { t: this._t, values: Float64Array.from(this._values[this._i]), label: label,
      kind: label ? "recorded fault" : "", sensors: [], ts: this._ts ? this._ts[this._i] : null };
    this._i++; this._t++;
    return s;
  }
}

// -------------------------------------------------------- math / models
function quantileSorted(a, q) {
  const pos = q * (a.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
const sortedCopy = (arr) => Float64Array.from(arr).sort();

class RLS {
  constructor(nIn, nOut, forgetting, delta) {
    this.d = nIn; this.S = nOut; this.lam = forgetting; this.delta = delta; this.maxTrace = 10 * delta * nIn;
    this.W = new Float64Array(nIn * nOut); this.P = new Float64Array(nIn * nIn); this._resetP();
  }
  _resetP() { this.P.fill(0); for (let i = 0; i < this.d; i++) this.P[i * this.d + i] = this.delta; }
  predict(x) {
    const out = new Float64Array(this.S);
    for (let i = 0; i < this.d; i++) { const xi = x[i]; if (xi !== 0) for (let j = 0; j < this.S; j++) out[j] += xi * this.W[i * this.S + j]; }
    return out;
  }
  update(x, y) {
    const d = this.d, S = this.S, P = this.P;
    const Px = new Float64Array(d);
    for (let i = 0; i < d; i++) { let s = 0; for (let k = 0; k < d; k++) s += P[i * d + k] * x[k]; Px[i] = s; }
    let denom = this.lam; for (let i = 0; i < d; i++) denom += x[i] * Px[i];
    const pred = this.predict(x);
    const err = new Float64Array(S); for (let j = 0; j < S; j++) err[j] = y[j] - pred[j];
    if (!Number.isFinite(denom) || denom <= 1e-12) { this._resetP(); return err; }
    const k = new Float64Array(d); for (let i = 0; i < d; i++) k[i] = Px[i] / denom;
    for (let i = 0; i < d; i++) for (let j = 0; j < S; j++) this.W[i * S + j] += k[i] * err[j];
    const N = new Float64Array(d * d);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) N[i * d + j] = (P[i * d + j] - k[i] * Px[j]) / this.lam;
    let trace = 0, finite = true;
    for (let i = 0; i < d; i++) for (let j = i; j < d; j++) {
      const v = 0.5 * (N[i * d + j] + N[j * d + i]); N[i * d + j] = v; N[j * d + i] = v;
      if (!Number.isFinite(v)) finite = false;
    }
    for (let i = 0; i < d; i++) trace += N[i * d + i];
    for (let i = 0; i < this.W.length; i++) if (!Number.isFinite(this.W[i])) { finite = false; this.W[i] = 0; }
    if (!finite) { this._resetP(); return err; }
    if (trace > this.maxTrace) { const f = this.maxTrace / trace; for (let i = 0; i < N.length; i++) N[i] *= f; }
    this.P = N;
    return err;
  }
}

class EwmStats {
  constructor(dim, alpha) { this.alpha = alpha; this.n = 0; this.mean = new Float64Array(dim); this.v = new Float64Array(dim); }
  update(x) {
    this.n++;
    const a = Math.max(1 / this.n, this.alpha);
    for (let i = 0; i < x.length; i++) {
      const d = x[i] - this.mean[i];
      this.mean[i] += a * d;
      this.v[i] = (1 - a) * (this.v[i] + a * d * d);
    }
  }
  std(i) { return Math.sqrt(this.v[i]); }
}

function cFactor(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
}
class IsolationForest {
  constructor(nTrees, rng) { this.nTrees = nTrees; this.rng = rng; this.trees = []; this.psi = 0; }
  fit(X) {
    const n = X.length, d = X[0].length, rng = this.rng;
    this.psi = Math.min(256, n);
    const limit = Math.ceil(Math.log2(this.psi));
    const build = (idx, depth) => {
      if (depth >= limit || idx.length <= 1) return { size: idx.length };
      let f = -1, lo = 0, hi = 0;
      const order = rng.sample(d, d);
      for (const cand of order) {
        let mn = Infinity, mx = -Infinity;
        for (const i of idx) { const v = X[i][cand]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mx > mn) { f = cand; lo = mn; hi = mx; break; }
      }
      if (f < 0) return { size: idx.length };
      const p = rng.uniform(lo, hi);
      const L = [], R = [];
      for (const i of idx) (X[i][f] < p ? L : R).push(i);
      if (!L.length || !R.length) return { size: idx.length };
      return { f: f, p: p, l: build(L, depth + 1), r: build(R, depth + 1) };
    };
    this.trees = [];
    for (let t = 0; t < this.nTrees; t++) this.trees.push(build(rng.sample(n, this.psi), 0));
  }
  score(x) {                       // in (0, 1); higher = more anomalous
    let total = 0;
    for (const tree of this.trees) {
      let node = tree, depth = 0;
      while (node.size === undefined) { node = x[node.f] < node.p ? node.l : node.r; depth++; }
      total += depth + cFactor(node.size);
    }
    return Math.pow(2, -(total / this.trees.length) / cFactor(this.psi));
  }
}

class OnlineAnomalyDetector {
  constructor(nSensors, cfg, seed) {
    if (nSensors < 1) throw new Error("nSensors must be >= 1");
    this.S = nSensors; this.cfg = cfg; this.seed = seed || 0; this.tick = 0;
    this._n = 0; this._m = new Float64Array(nSensors); this._m2 = new Float64Array(nSensors);
    this.mean = new Float64Array(nSensors); this.scale = new Float64Array(nSensors).fill(1); this.frozen = false;
    this.rls = new RLS(1 + cfg.lags * nSensors, nSensors, cfg.forgetting, cfg.pInit);
    this.hist = []; this.stats = new EwmStats(nSensors, cfg.residAlpha); this.zhist = []; this.buf = [];
    this.iso = null; this.env = null; this.fits = 0; this.lastFit = 0; this.predTicks = 0;
    this.consecFlag = 0; this.consecClear = 0; this.alert = false; this.alertRun = 0; this.lastRaw = null;
  }
  _validate(raw) {
    if (!raw || raw.length !== this.S) throw new Error("expected " + this.S + " readings, got " + (raw ? raw.length : 0));
    const arr = Float64Array.from(raw);
    let bad = false;
    for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) bad = true;
    if (bad) {
      if (this.lastRaw === null) throw new Error("non-finite reading before any valid sample");
      for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) arr[i] = this.lastRaw[i];
    }
    this.lastRaw = Float64Array.from(arr);
    return arr;
  }
  _design() {
    const x = new Float64Array(1 + this.cfg.lags * this.S); x[0] = 1;
    let p = 1;
    for (let h = this.hist.length - 1; h >= 0; h--) for (let j = 0; j < this.S; j++) x[p++] = this.hist[h][j];
    return x;
  }
  _fitIso(tick) {
    const cfg = this.cfg, S = this.S, X = this.buf;
    const forest = new IsolationForest(cfg.isoTrees, new Rng(this.seed + 7919 * (this.fits + 1)));
    forest.fit(X);
    const scores = sortedCopy(X.map(r => forest.score(r)));
    const med = quantileSorted(scores, 0.5);
    const thr = Math.max(quantileSorted(scores, cfg.isoQuantile), med + 1e-3);
    this.iso = { forest: forest, med: med, thr: thr };
    const lo = new Float64Array(S), hi = new Float64Array(S);
    for (let j = 0; j < S; j++) {
      const col = sortedCopy(X.map(r => r[j]));
      lo[j] = quantileSorted(col, cfg.envQuantile); hi[j] = quantileSorted(col, 1 - cfg.envQuantile);
    }
    this.env = { lo: lo, hi: hi };
    this.fits++; this.lastFit = tick;
  }
  update(rawIn) {
    const S = this.S, cfg = this.cfg;
    const raw = this._validate(rawIn);
    const tick = this.tick++;
    const zeros = new Float64Array(S);
    if (tick < cfg.scalerTicks) {
      this._n++;
      for (let i = 0; i < S; i++) { const d = raw[i] - this._m[i]; this._m[i] += d / this._n; this._m2[i] += d * (raw[i] - this._m[i]); }
      return { tick: tick, status: "calibrating", prediction: null, z: zeros, isoScore: 0, envScore: 0, score: 0, flagged: false, alert: false };
    }
    if (!this.frozen) {
      for (let i = 0; i < S; i++) {
        this.mean[i] = this._m[i];
        const sd = Math.sqrt(this._m2[i] / this._n);
        this.scale[i] = Math.max(sd < 1e-12 ? 1 : sd, 1e-8);
      }
      this.frozen = true;
    }
    const scaled = new Float64Array(S);
    for (let i = 0; i < S; i++) scaled[i] = (raw[i] - this.mean[i]) / this.scale[i];
    let x = null, pred = null, resid = zeros;
    if (this.hist.length === cfg.lags) {
      x = this._design(); pred = this.rls.predict(x);
      resid = new Float64Array(S); for (let i = 0; i < S; i++) resid[i] = scaled[i] - pred[i];
      this.predTicks++;
    }
    const statsValid = this.stats.n >= cfg.minStats;
    let z = zeros;
    if (statsValid && pred !== null) {
      z = new Float64Array(S);
      for (let i = 0; i < S; i++) z[i] = (resid[i] - this.stats.mean[i]) / Math.max(this.stats.std(i), 1e-3);
    }
    if (pred !== null) {
      this.zhist.push(Float64Array.from(z, v => Math.abs(v)));
      if (this.zhist.length > cfg.roll) this.zhist.shift();
    }
    const roll = new Float64Array(S);
    if (this.zhist.length) { for (const h of this.zhist) for (let i = 0; i < S; i++) roll[i] += h[i]; for (let i = 0; i < S; i++) roll[i] /= this.zhist.length; }
    const feat = new Float64Array(3 * S);
    for (let i = 0; i < S; i++) { feat[i] = scaled[i]; feat[S + i] = Math.max(-20, Math.min(20, z[i])); feat[2 * S + i] = Math.min(roll[i], 20); }

    const monitoring = tick >= cfg.warmupTicks && pred !== null && statsValid;
    if (monitoring) {
      const due = this.iso === null || (tick - this.lastFit) >= cfg.retrainEvery;
      if (due && this.buf.length >= cfg.minIsoSamples) this._fitIso(tick);
    }
    let isoNorm = 0, envNorm = 0;
    if (monitoring && this.iso !== null) {
      const s = this.iso.forest.score(feat);
      isoNorm = Math.max(0, (s - this.iso.med) / ((this.iso.thr - this.iso.med) * cfg.isoMargin));
      for (let i = 0; i < S; i++) {
        const width = Math.max(this.env.hi[i] - this.env.lo[i], 1e-6);
        const exc = Math.max(this.env.lo[i] - scaled[i], scaled[i] - this.env.hi[i], 0);
        envNorm = Math.max(envNorm, exc / (cfg.envMargin * width));
      }
    }
    let zMax = 0; for (let i = 0; i < S; i++) zMax = Math.max(zMax, Math.abs(z[i]));
    const zRatio = (monitoring && statsValid) ? zMax / cfg.zThreshold : 0;
    const active = monitoring && this.iso !== null;
    const score = active ? Math.max(zRatio, isoNorm, envNorm) : 0;
    const flagged = active && score >= 1.0;

    if (flagged) { this.consecFlag++; this.consecClear = 0; } else { this.consecFlag = 0; this.consecClear++; }
    const strong = flagged && score >= cfg.strongScore;
    if (!this.alert && (this.consecFlag >= cfg.minConsecutive || strong)) { this.alert = true; this.alertRun = 0; }
    else if (this.alert && this.consecClear >= cfg.cooldown) { this.alert = false; this.alertRun = 0; }
    if (this.alert) this.alertRun++;

    const learn = !(flagged || this.alert) || this.alertRun > cfg.maxAlertRun;
    if (learn && pred !== null) {
      this.rls.update(x, scaled);
      if (this.predTicks > cfg.burnIn) {
        if (statsValid) { this.buf.push(feat); if (this.buf.length > cfg.isoWindow) this.buf.shift(); }
        this.stats.update(resid);
      }
    }
    this.hist.push(scaled); if (this.hist.length > cfg.lags) this.hist.shift();

    let prediction = null;
    if (pred !== null) { prediction = new Float64Array(S); for (let i = 0; i < S; i++) prediction[i] = pred[i] * this.scale[i] + this.mean[i]; }
    return { tick: tick, status: active ? "monitoring" : "learning", prediction: prediction, z: z,
      isoScore: isoNorm, envScore: envNorm, score: score, flagged: flagged, alert: this.alert };
  }
}

// ---------------------------------------------------------------- metrics
const ratio = (n, d) => (d <= 0 ? null : n / d);
class OnlineMetrics {
  constructor(grace) { this.grace = grace === undefined ? 10 : grace; this.reset(); }
  reset() {
    this.ticks = 0; this.tp = this.fp = this.fn = this.tn = 0; this.episodes = 0; this.detected = 0;
    this.delays = []; this.falseAlertEvents = 0; this.alertEvents = 0;
    this._prevLabel = 0; this._prevAlert = false; this._epStart = 0; this._epDetected = false; this._graceLeft = 0;
  }
  update(label, alert, t) {
    if (t === undefined) t = this.ticks;
    this.ticks++; label = label ? 1 : 0; alert = !!alert;
    if (label && !this._prevLabel) { this._epStart = t; this._epDetected = false; }
    if (label && alert && !this._epDetected) { this._epDetected = true; this.delays.push(t - this._epStart); }
    if (!label && this._prevLabel) { this.episodes++; if (this._epDetected) this.detected++; this._graceLeft = this.grace; }
    if (alert && !this._prevAlert) { this.alertEvents++; if (!label && this._graceLeft <= 0) this.falseAlertEvents++; }
    if (label) { if (alert) this.tp++; else this.fn++; }
    else if (this._graceLeft > 0) this._graceLeft--;
    else if (alert) this.fp++;
    else this.tn++;
    this._prevLabel = label; this._prevAlert = alert;
  }
  snapshot() {
    const precision = ratio(this.tp, this.tp + this.fp), recall = ratio(this.tp, this.tp + this.fn);
    let f1 = null;
    if (precision !== null && recall !== null && precision + recall > 0) f1 = 2 * precision * recall / (precision + recall);
    return {
      ticks: this.ticks, precision: precision, recall: recall, f1: f1, tp: this.tp, fp: this.fp, fn: this.fn, tn: this.tn,
      episodes: this.episodes, detected: this.detected, event_recall: ratio(this.detected, this.episodes),
      mean_delay: this.delays.length ? this.delays.reduce((a, b) => a + b, 0) / this.delays.length : null,
      alert_events: this.alertEvents, false_alert_events: this.falseAlertEvents,
    };
  }
}

// ----------------------------------------------------------------- engine
const rnd = (x, nd) => { x = Number(x); if (!Number.isFinite(x)) return 0; const f = Math.pow(10, nd === undefined ? 4 : nd); return Math.round(x * f) / f; };

class StreamEngine {
  constructor(opts) {
    const o = Object.assign({ source: null, sensitivity: "normal", detector: {}, history: 1200, metricsGrace: 10, tickInterval: 0.05, seed: 0 }, opts || {});
    this.source = o.source || new SimSource();
    this.names = this.source.sensorNames; this.units = this.source.units || this.names.map(() => "");
    this.hasLabels = this.source.hasLabels !== false;
    this.canInject = typeof this.source.inject === "function";
    this.cfg = detectorConfig(o.sensitivity, o.detector);
    this.sensitivity = o.sensitivity;
    this.detector = new OnlineAnomalyDetector(this.names.length, this.cfg, o.seed);
    this.metrics = new OnlineMetrics(o.metricsGrace);
    this.maxHistory = o.history; this.tickInterval = o.tickInterval;
    this.history = []; this.events = [];
    this.paused = false; this.speed = 1; this._acc = 0; this.finished = false; this.failed = false;
    this.lastError = null; this._errors = 0; this._prevAlert = false;
  }
  step() {
    const sample = this.source.next();
    const det = this.detector.update(sample.values);
    this.metrics.update(sample.label, det.alert, sample.t);
    const rec = {
      t: sample.t, v: Array.from(sample.values, v => rnd(v)), p: det.prediction ? Array.from(det.prediction, v => rnd(v)) : null,
      z: Array.from(det.z, v => rnd(Math.max(-99, Math.min(99, v)), 2)), s: rnd(Math.min(det.score, 10), 3),
      f: det.flagged ? 1 : 0, a: det.alert ? 1 : 0, y: sample.label, k: sample.kind, ys: sample.sensors.slice(),
      st: det.status, ts: sample.ts,
    };
    this.history.push(rec);
    if (this.history.length > this.maxHistory) this.history.shift();
    if (det.alert !== this._prevAlert) this._emit(det.alert, sample, det);
    this._prevAlert = det.alert;
    return rec;
  }
  _emit(raised, sample, det) {
    let zMax = 0; for (const v of det.z) zMax = Math.max(zMax, Math.abs(v));
    const comps = { forecast_residual: zMax / this.cfg.zThreshold, isolation_forest: det.isoScore, operating_envelope: det.envScore };
    const reason = Object.keys(comps).reduce((a, b) => (comps[b] > comps[a] ? b : a));
    let suspects = [];
    if (reason === "forecast_residual") {
      suspects = Array.from(det.z).map((v, i) => [Math.abs(v), i]).filter(p => p[0] >= 3).sort((a, b) => b[0] - a[0]).slice(0, 3).map(p => this.names[p[1]]);
    }
    const values = {}; this.names.forEach((n, i) => { values[n] = rnd(sample.values[i]); });
    this.events.push({ event: raised ? "alert_raised" : "alert_cleared", tick: sample.t, timestamp: sample.ts || ("tick " + sample.t),
      score: rnd(det.score, 3), reason: reason, suspect_sensors: suspects, values: values });
  }
  /** Advance the engine by dt seconds of wall-clock time (real-time pacing). */
  advance(dt) {
    if (this.failed || this.finished || this.paused) return 0;
    this._acc += dt * this.speed / this.tickInterval;
    let n = Math.floor(this._acc); this._acc -= n;
    if (n > MAX_CATCHUP) { n = MAX_CATCHUP; this._acc = 0; }
    let done = 0;
    for (let i = 0; i < n; i++) {
      try { this.step(); this._errors = 0; done++; }
      catch (e) {
        if (e instanceof SourceExhausted) { this.finished = true; break; }
        this._errors++; this.lastError = e.name + ": " + e.message;
        if (this._errors >= MAX_ERRORS) { this.failed = true; break; }
      }
    }
    return done;
  }
  inject(kind) {
    if (!this.canInject) return [false, "this data source does not support injection"];
    return this.source.inject(kind || null);
  }
  pause() { this.paused = true; }
  resume() { this.paused = false; }
  setSpeed(v) {
    v = Number(v);
    if (!Number.isFinite(v)) throw new Error("speed must be a finite number");
    this.speed = Math.min(MAX_SPEED, Math.max(MIN_SPEED, v)); return this.speed;
  }
  snapshot(since, limit) {
    since = since === undefined ? -1 : since; limit = limit || 400;
    const h = this.history, latest = h.length ? h[h.length - 1].t : -1;
    const reset = since > latest; if (reset) since = -1;
    const pts = [];
    for (let i = h.length - 1; i >= 0; i--) { if (h[i].t <= since || pts.length >= limit) break; pts.push(h[i]); }
    pts.reverse();
    let status = h.length ? h[h.length - 1].st : "starting";
    if (this.finished) status = "finished";
    const warm = this.cfg.warmupTicks;
    return {
      now: latest, reset: reset, points: pts,
      sensors: this.names.map((n, i) => ({ name: n, unit: this.units[i] || "" })),
      status: this.failed ? "error" : status, has_labels: this.hasLabels, can_inject: this.canInject,
      paused: this.paused, speed: this.speed, warmup_ticks: warm, progress: Math.min(1, (latest + 1) / warm),
      metrics: this.metrics.snapshot(), error: this.failed ? this.lastError : null,
    };
  }
  eventsCsv() {
    const q = (s) => { s = String(s); if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; return '"' + s.replace(/"/g, '""') + '"'; };
    const lines = ["timestamp,tick,event,score,reason,suspect_sensors," + this.names.map(q).join(",")];
    for (const e of this.events) lines.push([q(e.timestamp), e.tick, e.event, e.score, e.reason, q(e.suspect_sensors.join(" | ")),
      ...this.names.map(n => e.values[n])].join(","));
    return lines.join("\n");
  }
}

const api = { SimSource, CsvSource, StreamEngine, OnlineAnomalyDetector, OnlineMetrics, RLS, EwmStats, IsolationForest, Rng,
  SourceExhausted, SourceError, detectorConfig, parseCsvText, KINDS, SENSITIVITY, DEFAULT_DET, DEFAULT_SIM };
if (typeof module !== "undefined" && module.exports) module.exports = api; else root.StreamSentinel = api;
})(typeof self !== "undefined" ? self : this);
