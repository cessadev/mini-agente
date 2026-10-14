"""Servidor HTTP local que imita la API Interactions de Gemini con respuestas guionizadas.

Sirve para probar el CLI completo (bucle + herramientas + salida) sin red ni cuota.
Un elemento del guion puede ser (código_http, cuerpo) para simular errores como el 429.
Guarda cada petición recibida para poder comprobar qué le envía realmente el agente al modelo.
"""
import http.server
import json
import threading


def tool_call_response(n, calls, usage_tokens=10):
    """Respuesta en la que el modelo pide herramientas. calls = [(call_id, name, args), ...]"""
    return {
        "id": f"interaction_{n}",
        "status": "requires_action",
        "steps": [
            {"type": "function_call", "id": cid, "name": name, "arguments": args}
            for cid, name, args in calls
        ],
        "usage": {"total_tokens": usage_tokens},
    }


def text_response(n, text, usage_tokens=10):
    """Respuesta final de texto."""
    return {
        "id": f"interaction_{n}",
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {"total_tokens": usage_tokens},
    }


def quota_exceeded_response():
    """(429, cuerpo) que imita 'cuota diaria agotada'. Úsalo como elemento del guion."""
    return (429, {"error": {"message": "Rate limit exceeded ... limit: 20 requests per day on Free Tier ..."}})


class FakeGemini:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []  # cuerpos JSON recibidos, en orden
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                item = outer.responses.pop(0) if outer.responses else text_response(99, "(sin guion)")
                status, payload = item if isinstance(item, tuple) else (200, item)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
