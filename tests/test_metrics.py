import unittest

from streamsentinel.metrics import OnlineMetrics


class MetricsTests(unittest.TestCase):
    def test_hand_computed_scenario(self):
        m = OnlineMetrics(grace=2)
        seq = [(0, 0), (0, 0), (1, 0), (1, 1), (1, 1), (0, 1), (0, 0), (0, 0), (0, 1), (0, 0)]
        for t, (label, alert) in enumerate(seq):
            m.update(label, alert, t)
        s = m.snapshot()
        self.assertEqual((s["tp"], s["fn"], s["fp"], s["tn"]), (2, 1, 1, 4))
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["f1"], 2 / 3)
        self.assertEqual((s["episodes"], s["detected"]), (1, 1))
        self.assertEqual(s["mean_delay"], 1)
        self.assertEqual(s["alert_events"], 2)
        self.assertEqual(s["false_alert_events"], 1)
        self.assertEqual(s["ticks"], 10)

    def test_missed_episode_and_empty_cases(self):
        m = OnlineMetrics(grace=0)
        s = m.snapshot()
        self.assertIsNone(s["precision"])
        self.assertIsNone(s["recall"])
        self.assertIsNone(s["f1"])
        self.assertIsNone(s["mean_delay"])
        for label in (0, 1, 1, 0, 0):
            m.update(label, 0)
        s = m.snapshot()
        self.assertEqual((s["episodes"], s["detected"]), (1, 0))
        self.assertEqual(s["recall"], 0.0)
        self.assertIsNone(s["precision"])       # no positive predictions
        self.assertIsNone(s["f1"])

    def test_alert_during_grace_is_not_false_event(self):
        m = OnlineMetrics(grace=3)
        for label, alert in [(1, 1), (0, 0), (0, 1), (0, 1), (0, 0)]:
            m.update(label, alert)
        self.assertEqual(m.snapshot()["false_alert_events"], 0)


if __name__ == "__main__":
    unittest.main()
