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
AGENTE_PY = BASE_DIR / "agente.py"


class RateLimitHandler(http.server.BaseHTTPRequestHandler):
    """Responde inmediatamente 429 con cabecera Retry-After larga y cuerpo JSON."""

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
        # Silenciar logs del servidor durante la prueba
        pass


class HangingHandler(http.server.BaseHTTPRequestHandler):
    """Acepta la conexión pero nunca responde dentro de la ventana de tiempo."""

    def do_POST(self):
        # Mantiene la conexión abierta simulando cuelgue del servidor
        try:
            for _ in range(40):
                time.sleep(0.5)
        except Exception:
            pass

    def log_message(self, format, *args):
        # Silenciar logs del servidor durante la prueba
        pass


class TestFailFast(unittest.TestCase):
    def test_case_1_rate_limit_429(self):
        """
        Caso 1: el servidor responde 429 con Retry-After: 27458.
        Esperado: el subproceso termina en menos de 10 s con código 2 y 'LÍMITE/CUOTA' en stderr.
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
                [sys.executable, str(AGENTE_PY)],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,  # Timeout de seguridad propio en el test
            )
            elapsed = time.time() - start_time

            self.assertLess(
                elapsed, 10.0, f"El subproceso tardó demasiado: {elapsed:.2f} s"
            )
            self.assertEqual(
                result.returncode,
                2,
                f"Código de salida esperado 2, obtenido {result.returncode}. Stderr: {result.stderr}",
            )
            self.assertIn(
                "LÍMITE/CUOTA",
                result.stderr,
                f"No se encontró 'LÍMITE/CUOTA' en stderr. Stderr obtenido: {result.stderr}",
            )
            self.assertIn(
                "27458",
                result.stderr,
                f"No se encontró Retry-After en stderr. Stderr obtenido: {result.stderr}",
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_case_2_timeout(self):
        """
        Caso 2: el servidor acepta la conexión y nunca responde; con AGENT_API_TIMEOUT=1.
        Esperado: termina en menos de 10 s con código 2 y 'TIMEOUT' en stderr.
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
                [sys.executable, str(AGENTE_PY)],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,  # Timeout de seguridad propio en el test
            )
            elapsed = time.time() - start_time

            self.assertLess(
                elapsed, 10.0, f"El subproceso tardó demasiado: {elapsed:.2f} s"
            )
            self.assertEqual(
                result.returncode,
                2,
                f"Código de salida esperado 2, obtenido {result.returncode}. Stderr: {result.stderr}",
            )
            self.assertIn(
                "TIMEOUT",
                result.stderr,
                f"No se encontró 'TIMEOUT' en stderr. Stderr obtenido: {result.stderr}",
            )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
