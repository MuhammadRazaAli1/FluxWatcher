import json
import socket
import threading
import unittest
import urllib.error
import urllib.request

from streamsentinel.config import AppConfig
from streamsentinel.engine import StreamEngine
from streamsentinel.server import create_server, json_safe


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = StreamEngine(AppConfig())
        for _ in range(40):
            cls.engine.step()
        cls.server = create_server(cls.engine, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def req(self, path, data=None, method=None, raw=None):
        body = raw if raw is not None else (None if data is None else json.dumps(data).encode())
        r = urllib.request.Request(self.base + path, data=body, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def test_dashboard_served(self):
        status, headers, body = self.req("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"StreamSentinel", body)

    def test_health_and_state(self):
        self.assertEqual(json.loads(self.req("/api/health")[2])["ok"], True)
        status, _, body = self.req("/api/state?since=-1")
        d = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(d["now"], 39)
        self.assertEqual(len(d["points"]), 40)
        self.assertIn("metrics", d)

    def test_bad_requests(self):
        self.assertEqual(self.req("/nope")[0], 404)
        self.assertEqual(self.req("/api/state?since=abc")[0], 400)
        self.assertEqual(self.req("/api/inject", raw=b"not json", method="POST")[0], 400)
        self.assertEqual(self.req("/api/inject", raw=b"[1,2]", method="POST")[0], 400)
        self.assertEqual(self.req("/api/inject", {"kind": 5}, method="POST")[0], 400)
        self.assertEqual(self.req("/api/speed", {"speed": "fast"}, method="POST")[0], 400)
        self.assertEqual(self.req("/api/speed", {}, method="POST")[0], 400)
        self.assertEqual(self.req("/api/unknown", {}, method="POST")[0], 404)
        self.assertEqual(self.req("/api/state", method="PUT", raw=b"{}")[0], 405)

    def test_controls(self):
        s, _, b = self.req("/api/pause", {}, method="POST")
        self.assertTrue(json.loads(b)["paused"])
        self.assertTrue(self.engine.paused)
        s, _, b = self.req("/api/resume", {}, method="POST")
        self.assertFalse(self.engine.paused)
        s, _, b = self.req("/api/speed", {"speed": 2}, method="POST")
        self.assertEqual(json.loads(b)["speed"], 2.0)
        s, _, b = self.req("/api/inject", {"kind": "bogus"}, method="POST")
        self.assertEqual(s, 200)
        self.assertFalse(json.loads(b)["ok"])

    def test_json_safe_and_port_fallback(self):
        self.assertEqual(json_safe({"a": float("nan"), "b": [1.0, float("inf")]}), {"a": None, "b": [1.0, None]})
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        busy = blocker.getsockname()[1]
        try:
            srv = create_server(self.engine, "127.0.0.1", busy)
            self.assertNotEqual(srv.server_address[1], busy)
            srv.server_close()
        finally:
            blocker.close()


if __name__ == "__main__":
    unittest.main()
