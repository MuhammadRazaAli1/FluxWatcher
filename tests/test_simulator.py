import unittest

import numpy as np

from streamsentinel.config import SimConfig
from streamsentinel.simulator import SensorStream


class SimulatorTests(unittest.TestCase):
    def test_shapes_and_finite(self):
        s = SensorStream(SimConfig(seed=1))
        for _ in range(500):
            x = s.next()
            self.assertEqual(x.values.shape, (3,))
            self.assertTrue(np.all(np.isfinite(x.values)))

    def test_deterministic_for_seed(self):
        a = [SensorStream(SimConfig(seed=7)).next().values for _ in range(1)]
        s1, s2 = SensorStream(SimConfig(seed=7)), SensorStream(SimConfig(seed=7))
        for _ in range(800):
            x, y = s1.next(), s2.next()
            np.testing.assert_array_equal(x.values, y.values)
            self.assertEqual((x.label, x.kind), (y.label, y.kind))
        self.assertTrue(a)

    def test_no_anomalies_during_clean_phase(self):
        s = SensorStream(SimConfig(seed=3, clean_ticks=400, anomaly_rate=1.0))
        for _ in range(400):
            self.assertEqual(s.next().label, 0)
        self.assertEqual(s.next().label, 1)      # rate 1.0 -> starts immediately

    def test_zero_rate_never_anomalous(self):
        s = SensorStream(SimConfig(anomaly_rate=0.0))
        self.assertEqual(sum(s.next().label for _ in range(1500)), 0)

    def test_inject_each_kind_and_episode_ends(self):
        for kind in ("spike", "level_shift", "drift", "variance_burst"):
            s = SensorStream(SimConfig(seed=5, anomaly_rate=0.0, clean_ticks=10))
            for _ in range(10):
                s.next()
            ok, _ = s.inject(kind)
            self.assertTrue(ok, kind)
            labels = [s.next() for _ in range(200)]
            active = [x for x in labels if x.label]
            self.assertGreater(len(active), 2, kind)
            self.assertTrue(all(x.kind == kind for x in active))
            self.assertTrue(all(len(x.sensors) >= 1 for x in active))
            self.assertEqual(labels[-1].label, 0)    # episode finished

    def test_inject_rejections(self):
        s = SensorStream(SimConfig(anomaly_rate=0.0, clean_ticks=50))
        self.assertFalse(s.inject("spike")[0])               # still in clean phase
        for _ in range(60):
            s.next()
        self.assertFalse(s.inject("nonsense")[0])
        self.assertTrue(s.inject("drift")[0])
        self.assertFalse(s.inject("spike")[0])               # already pending
        s.next()
        self.assertFalse(s.inject("spike")[0])               # already running

    def test_spike_is_visible_in_data(self):
        s = SensorStream(SimConfig(seed=2, anomaly_rate=0.0, clean_ticks=5))
        for _ in range(300):
            s.next()
        s.inject("spike")
        xs = [s.next() for _ in range(6)]
        dev = max(abs(x.values - np.array([sp.base for sp in s.cfg.sensors])).max() for x in xs)
        self.assertGreater(dev, 0.0)
        self.assertEqual(xs[0].label, 1)


if __name__ == "__main__":
    unittest.main()
