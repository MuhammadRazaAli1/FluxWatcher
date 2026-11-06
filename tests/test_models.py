import unittest

import numpy as np

from streamsentinel.config import DetectorConfig, SimConfig
from streamsentinel.models import EwmStats, OnlineAnomalyDetector, RecursiveLeastSquares
from streamsentinel.simulator import SensorStream


class RlsTests(unittest.TestCase):
    def test_learns_linear_relationship(self):
        rng = np.random.default_rng(0)
        rls = RecursiveLeastSquares(3, 2, forgetting=0.999)
        true_w = np.array([[0.5, -1.0], [2.0, 0.0], [0.0, 3.0]])
        for _ in range(600):
            x = np.concatenate(([1.0], rng.normal(size=2)))
            y = x @ true_w + rng.normal(scale=0.01, size=2)
            rls.update(x, y)
        np.testing.assert_allclose(rls.W, true_w, atol=0.05)

    def test_stays_finite_with_degenerate_input(self):
        rls = RecursiveLeastSquares(4, 1, forgetting=0.99)
        x = np.array([1.0, 0.0, 0.0, 0.0])           # never excites 3 of 4 dimensions
        for _ in range(5000):
            rls.update(x, np.array([1.0]))
        self.assertTrue(np.all(np.isfinite(rls.W)))
        self.assertTrue(np.all(np.isfinite(rls.P)))
        self.assertLessEqual(np.trace(rls.P), rls.max_trace * 1.0001)


class EwmTests(unittest.TestCase):
    def test_converges_to_distribution(self):
        rng = np.random.default_rng(1)
        st = EwmStats(2, alpha=0.01)
        for x in rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(4000, 2)):
            st.update(x)
        np.testing.assert_allclose(st.mean, [3.0, -1.0], atol=0.5)
        np.testing.assert_allclose(st.std, [2.0, 0.5], rtol=0.3)


def clean_stream(seed=11, n=700):
    s = SensorStream(SimConfig(seed=seed, anomaly_rate=0.0))
    return [s.next().values for _ in range(n)], s


class DetectorTests(unittest.TestCase):
    def test_phases_and_finite_outputs(self):
        cfg = DetectorConfig()
        det = OnlineAnomalyDetector(3, cfg)
        data, _ = clean_stream(n=cfg.warmup_ticks + 60)
        statuses = []
        for x in data:
            r = det.update(x)
            statuses.append(r.status)
            self.assertTrue(np.isfinite(r.score) and np.isfinite(r.iso_score) and np.isfinite(r.env_score))
            self.assertTrue(np.all(np.isfinite(r.z)))
            if r.prediction is not None:
                self.assertTrue(np.all(np.isfinite(r.prediction)))
        self.assertTrue(all(s == "calibrating" for s in statuses[:cfg.scaler_ticks]))
        self.assertEqual(statuses[cfg.scaler_ticks], "learning")
        self.assertEqual(statuses[-1], "monitoring")
        self.assertNotIn("monitoring", statuses[:cfg.warmup_ticks])

    def test_no_alerts_on_clean_data_and_forecast_accuracy(self):
        det = OnlineAnomalyDetector(3, DetectorConfig())
        data, s = clean_stream(n=1200)
        alerts, errs = 0, []
        for i, x in enumerate(data):
            r = det.update(x)
            alerts += r.alert
            if i > 400 and r.prediction is not None:
                errs.append(np.abs(r.prediction - x) / np.array([sp.amp for sp in s.cfg.sensors]))
        self.assertLessEqual(alerts, 3)
        self.assertLess(np.mean(errs), 0.15)       # forecast error << seasonal amplitude

    def test_detects_injected_spike_and_level_shift(self):
        for kind in ("spike", "level_shift", "variance_burst"):
            s = SensorStream(SimConfig(seed=21, anomaly_rate=0.0, clean_ticks=10))
            det = OnlineAnomalyDetector(3, DetectorConfig())
            for _ in range(700):
                det.update(s.next().values)
            ok, _ = s.inject(kind)
            self.assertTrue(ok)
            detected = False
            for _ in range(80):
                x = s.next()
                r = det.update(x.values)
                if x.label and r.alert:
                    detected = True
            self.assertTrue(detected, kind)

    def test_missing_values_are_forward_filled(self):
        det = OnlineAnomalyDetector(3, DetectorConfig())
        det.update([1.0, 2.0, 3.0])
        r = det.update([float("nan"), 2.0, float("inf")])
        self.assertTrue(np.all(np.isfinite(r.z)))
        with self.assertRaises(ValueError):
            OnlineAnomalyDetector(2).update([float("nan"), 1.0])   # nothing to fill from
        with self.assertRaises(ValueError):
            det.update([1.0, 2.0])                                 # wrong width
        with self.assertRaises(ValueError):
            OnlineAnomalyDetector(0)

    def test_constant_signal_does_not_crash(self):
        det = OnlineAnomalyDetector(2, DetectorConfig())
        for _ in range(500):
            r = det.update([5.0, 5.0])
            self.assertTrue(np.isfinite(r.score))


if __name__ == "__main__":
    unittest.main()
