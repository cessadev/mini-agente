"""Local HTTP server that imitates the Gemini Interactions API with scripted responses.

It is used to test the whole CLI (loop + tools + output) without network or quota.
An element of the script can be (http_code, body) to simulate errors such as a 429.
It stores every request received so that it is possible to check what the agent really sends to the model.
"""
import http.server
import json
import threading


def tool_call_response(n, calls, usage_tokens=10):
    """Response in which the model asks for tools. calls = [(call_id, name, args), ...]"""
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
    """Final text response."""
    return {
        "id": f"interaction_{n}",
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {"total_tokens": usage_tokens},
    }


def quota_exceeded_response():
    """(429, body) that imitates 'daily quota exhausted'. Use it as an element of the script."""
    return (429, {"error": {"message": "Rate limit exceeded ... limit: 20 requests per day on Free Tier ..."}})


class FakeGemini:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []  # JSON bodies received, in order
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
