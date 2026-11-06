"""Tiny dependency-free HTTP server exposing the engine and the dashboard."""
from __future__ import annotations

import json
import logging
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .engine import StreamEngine

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 4096


def json_safe(obj: Any) -> Any:
    """Recursively replace NaN / inf with None so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def make_handler(engine: StreamEngine, dashboard_html: bytes):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StreamSentinel/1.0"
        protocol_version = "HTTP/1.1"

        # -- plumbing ------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:   # keep console quiet
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                            # client went away

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json({"ok": False, "error": message}, status)

        def _read_json(self) -> Optional[dict]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self.close_connection = True        # unread body: do not reuse the socket
                self._error(413 if length > MAX_BODY else 400, "invalid body length")
                return None
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(400, "body must be valid JSON")
                return None
            if not isinstance(data, dict):
                self._error(400, "body must be a JSON object")
                return None
            return data

        # -- routes --------------------------------------------------------
        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            try:
                url = urlparse(self.path)
                if url.path in ("/", "/index.html"):
                    self._send(200, dashboard_html, "text/html; charset=utf-8")
                elif url.path == "/api/health":
                    self._json({"ok": True, "running": engine.running})
                elif url.path == "/api/state":
                    q = parse_qs(url.query)
                    try:
                        since = int(q.get("since", ["-1"])[0])
                    except ValueError:
                        return self._error(400, "'since' must be an integer")
                    self._json(engine.snapshot(since=since))
                else:
                    self._error(404, "not found")
            except Exception:                                   # pragma: no cover
                log.exception("GET %s failed", self.path)
                self._error(500, "internal error")

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                if path not in ("/api/inject", "/api/pause", "/api/resume", "/api/speed"):
                    return self._error(404, "not found")
                data = self._read_json()
                if data is None:
                    return
                if path == "/api/inject":
                    kind = data.get("kind")
                    if kind is not None and not isinstance(kind, str):
                        return self._error(400, "'kind' must be a string")
                    ok, message = engine.inject(kind or None)
                    self._json({"ok": ok, "message": message})
                elif path == "/api/pause":
                    engine.pause()
                    self._json({"ok": True, "paused": True})
                elif path == "/api/resume":
                    engine.resume()
                    self._json({"ok": True, "paused": False})
                else:
                    try:
                        speed = engine.set_speed(data["speed"])
                    except (KeyError, TypeError, ValueError):
                        return self._error(400, "'speed' must be a number")
                    self._json({"ok": True, "speed": speed})
            except Exception:                                   # pragma: no cover
                log.exception("POST %s failed", self.path)
                self._error(500, "internal error")

        def do_PUT(self) -> None:
            self._error(405, "method not allowed")

        do_DELETE = do_PATCH = do_PUT

    return Handler


def create_server(engine: StreamEngine, host: str = "127.0.0.1", port: int = 8000,
                  max_tries: int = 20) -> ThreadingHTTPServer:
    """Bind the server, trying the next ports if ``port`` is busy (port 0 = any)."""
    html = (STATIC_DIR / "dashboard.html").read_bytes()
    handler = make_handler(engine, html)
    last_error: Optional[OSError] = None
    candidates = [0] if port == 0 else range(port, min(port + max_tries, 65536))
    for p in candidates:
        try:
            server = ThreadingHTTPServer((host, p), handler)
            server.daemon_threads = True
            return server
        except OSError as exc:
            last_error = exc
    raise OSError(f"could not bind any port in {port}..{port + max_tries - 1}: {last_error}")


def serve(engine: StreamEngine, host: str = "127.0.0.1", port: int = 8000,
          open_browser: bool = True) -> None:
    """Start the engine and serve until Ctrl+C."""
    server = create_server(engine, host, port)
    bound_host, bound_port = server.server_address[:2]
    shown = "localhost" if bound_host in ("0.0.0.0", "127.0.0.1", "::") else bound_host
    url = f"http://{shown}:{bound_port}/"
    engine.start()
    print(f"\n  StreamSentinel is running -> {url}\n  Press Ctrl+C to stop.\n", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        server.server_close()
        engine.stop()
