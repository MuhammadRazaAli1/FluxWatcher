import unittest

from streamsentinel.config import AppConfig, DetectorConfig, SimConfig


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = AppConfig()
        self.assertGreaterEqual(cfg.sim.clean_ticks, cfg.detector.warmup_ticks)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            SimConfig(anomaly_rate=1.5)
        with self.assertRaises(ValueError):
            SimConfig(kind_weights=(1, 1))
        with self.assertRaises(ValueError):
            DetectorConfig(lags=0)
        with self.assertRaises(ValueError):
            DetectorConfig(warmup_ticks=10)
        with self.assertRaises(ValueError):
            DetectorConfig(iso_quantile=1.2)
        with self.assertRaises(ValueError):
            AppConfig(tick_interval=0)
        with self.assertRaises(ValueError):
            AppConfig(sim=SimConfig(clean_ticks=10))


if __name__ == "__main__":
    unittest.main()
