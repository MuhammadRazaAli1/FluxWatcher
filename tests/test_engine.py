import time
import unittest

import numpy as np

from streamsentinel.config import AppConfig, SimConfig
from streamsentinel.engine import MAX_SPEED, MIN_SPEED, StreamEngine
from streamsentinel.simulator import Sample


class FakeSource:
    sensor_names = ("a", "b")

    def __init__(self, fail=False):
        self.t, self.fail = 0, fail

    def next(self):
        if self.fail:
            raise RuntimeError("boom")
        self.t += 1
        return Sample(self.t - 1, np.array([np.sin(self.t / 9), np.cos(self.t / 7)]), 0, "", ())


class EngineTests(unittest.TestCase):
    def test_step_record_and_snapshot(self):
        e = StreamEngine(AppConfig())
        for _ in range(20):
            rec = e.step()
        for key in ("t", "v", "p", "z", "s", "f", "a", "y", "k", "ys", "st"):
            self.assertIn(key, rec)
        snap = e.snapshot(since=-1)
        self.assertEqual(snap["now"], 19)
        self.assertEqual(len(snap["points"]), 20)
        self.assertEqual([s["name"] for s in snap["sensors"]], ["Temperature", "Vibration", "Pressure"])
        part = e.snapshot(since=14)
        self.assertEqual([p["t"] for p in part["points"]], [15, 16, 17, 18, 19])
        self.assertFalse(part["reset"])
        self.assertTrue(e.snapshot(since=500)["reset"])
        self.assertEqual(len(e.snapshot(since=-1, limit=5)["points"]), 5)

    def test_history_is_bounded(self):
        e = StreamEngine(AppConfig(history=60))
        for _ in range(200):
            e.step()
        self.assertEqual(len(e.snapshot(since=-1, limit=1000)["points"]), 60)

    def test_speed_clamped_and_validated(self):
        e = StreamEngine(AppConfig())
        self.assertEqual(e.set_speed(100), MAX_SPEED)
        self.assertEqual(e.set_speed(0), MIN_SPEED)
        with self.assertRaises(ValueError):
            e.set_speed(float("nan"))

    def test_threaded_run_pause_resume_stop(self):
        e = StreamEngine(AppConfig(tick_interval=0.002), source=FakeSource())
        e.start()
        e.start()                                  # idempotent
        time.sleep(0.4)
        n1 = e.snapshot()["now"]
        self.assertGreater(n1, 20)
        e.pause()
        time.sleep(0.15)
        n2 = e.snapshot()["now"]
        time.sleep(0.2)
        self.assertEqual(e.snapshot()["now"], n2)
        self.assertTrue(e.snapshot()["paused"])
        e.resume()
        time.sleep(0.3)
        self.assertGreater(e.snapshot()["now"], n2)
        e.stop()
        self.assertFalse(e.running)

    def test_engine_reports_persistent_failure(self):
        e = StreamEngine(AppConfig(tick_interval=0.001), source=FakeSource(fail=True))
        with self.assertLogs("streamsentinel.engine", level="ERROR"):
            e.start()
            deadline = time.time() + 5
            while e.running and time.time() < deadline:
                time.sleep(0.02)
        self.assertFalse(e.running)
        snap = e.snapshot()
        self.assertEqual(snap["status"], "error")
        self.assertIn("boom", snap["error"])

    def test_inject_passthrough(self):
        e = StreamEngine(AppConfig())
        self.assertFalse(e.inject("spike")[0])               # baseline not learned yet
        for _ in range(AppConfig().sim.clean_ticks + 1):
            e.step()
        self.assertTrue(e.inject("spike")[0])
        labels = [e.step()["y"] for _ in range(6)]
        self.assertIn(1, labels)
        self.assertFalse(StreamEngine(AppConfig(), source=FakeSource()).inject("spike")[0])


if __name__ == "__main__":
    unittest.main()
