"""End-to-end quality gate: the full pipeline must actually find the anomalies."""
import unittest

import numpy as np

from streamsentinel.config import AppConfig, SimConfig
from streamsentinel.headless import format_report, run_headless


class PipelineTests(unittest.TestCase):
    def test_end_to_end_quality(self):
        df, m = run_headless(AppConfig(sim=SimConfig(seed=1)), 3000)
        self.assertEqual(len(df), 3000)
        self.assertGreaterEqual(m["episodes"], 3)
        self.assertGreaterEqual(m["detected"] / m["episodes"], 0.75)
        self.assertLessEqual(m["false_alert_events"], 2)
        self.assertGreaterEqual(m["precision"], 0.8)
        # never alerts before the warm-up has finished
        self.assertEqual(int(df.loc[df["t"] < 300, "alert"].sum()), 0)
        self.assertTrue(np.isfinite(df["score"]).all())
        self.assertIn("false alert events", format_report(m))

    def test_headless_is_reproducible(self):
        a, ma = run_headless(AppConfig(sim=SimConfig(seed=5)), 450)
        b, mb = run_headless(AppConfig(sim=SimConfig(seed=5)), 450)
        np.testing.assert_allclose(a["score"].to_numpy(), b["score"].to_numpy())
        self.assertEqual(ma, mb)
        with self.assertRaises(ValueError):
            run_headless(AppConfig(), 0)


if __name__ == "__main__":
    unittest.main()
