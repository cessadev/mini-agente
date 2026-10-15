import http.server
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AGENT_PY = BASE_DIR / "agent.py"


class RateLimitHandler(http.server.BaseHTTPRequestHandler):
    """Immediately answers 429 with a long Retry-After header and a JSON body."""

    def do_POST(self):
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Retry-After", "27458")
        self.end_headers()
        body = json.dumps({
            "error": {
                "message": "Rate limit exceeded ... limit: 20 requests per day on Free Tier ...",
                "code": "too_many_requests",
            }
        }).encode("utf-8")
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Silence the server logs during the test
        pass


class HangingHandler(http.server.BaseHTTPRequestHandler):
    """Accepts the connection but never answers within the time window."""

    def do_POST(self):
        # Keeps the connection open, simulating a hung server
        try:
            for _ in range(40):
                time.sleep(0.5)
        except Exception:
            pass

    def log_message(self, format, *args):
        # Silence the server logs during the test
        pass


class TestFailFast(unittest.TestCase):
    def test_case_1_rate_limit_429(self):
        """
        Case 1: the server answers 429 with Retry-After: 27458.
        Expected: the subprocess ends in under 10 s with exit code 2 and 'RATE LIMIT/QUOTA' in stderr.
        """
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RateLimitHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            env = os.environ.copy()
            env["AGENT_BASE_URL"] = f"http://127.0.0.1:{port}"
            env["GEMINI_API_KEY"] = "fake_test_key"

            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(AGENT_PY)],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,  # The test's own safety timeout
            )
            elapsed = time.time() - start_time

            self.assertLess(
                elapsed, 10.0, f"The subprocess took too long: {elapsed:.2f} s"
            )
            self.assertEqual(
                result.returncode,
                2,
                f"Expected exit code 2, got {result.returncode}. Stderr: {result.stderr}",
            )
            self.assertIn(
                "RATE LIMIT/QUOTA",
                result.stderr,
                f"'RATE LIMIT/QUOTA' not found in stderr. Stderr was: {result.stderr}",
            )
            self.assertIn(
                "27458",
                result.stderr,
                f"Retry-After not found in stderr. Stderr was: {result.stderr}",
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_case_2_timeout(self):
        """
        Case 2: the server accepts the connection and never answers; with AGENT_API_TIMEOUT=1.
        Expected: it ends in under 10 s with exit code 2 and 'TIMEOUT' in stderr.
        """
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HangingHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            env = os.environ.copy()
            env["AGENT_BASE_URL"] = f"http://127.0.0.1:{port}"
            env["GEMINI_API_KEY"] = "fake_test_key"
            env["AGENT_API_TIMEOUT"] = "1"

            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(AGENT_PY)],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,  # The test's own safety timeout
            )
            elapsed = time.time() - start_time

            self.assertLess(
                elapsed, 10.0, f"The subprocess took too long: {elapsed:.2f} s"
            )
            self.assertEqual(
                result.returncode,
                2,
                f"Expected exit code 2, got {result.returncode}. Stderr: {result.stderr}",
            )
            self.assertIn(
                "TIMEOUT",
                result.stderr,
                f"'TIMEOUT' not found in stderr. Stderr was: {result.stderr}",
            )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
