"""Capa del modelo: aísla TODO lo específico de un proveedor (hoy, Gemini).

El bucle del agente solo conoce estos tipos neutros. Para probar el bucle sin API se
sustituye el cliente por uno falso; para cambiar de proveedor se escribe otro cliente.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class ModelError(Exception):
    """Fallo de infraestructura al llamar al modelo (cuota, timeout, red, API).

    str(e) es el mensaje listo para mostrar. La biblioteca lanza; quien decide
    terminar el proceso (sys.exit) es el punto de entrada, no el código reutilizable.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    output: Dict[str, Any]


@dataclass
class ModelTurn:
    id: str
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    total_tokens: Optional[int] = None


class ModelClient(Protocol):
    def start(self, prompt: str, tools: List[dict], system_instruction: Optional[str]) -> ModelTurn: ...

    def send_results(
        self, previous_id: str, results: List[ToolResult], tools: List[dict],
        system_instruction: Optional[str],
    ) -> ModelTurn: ...


def _interaction_text(interaction) -> str:
    """Extrae el texto de respuesta de una interacción de Gemini."""
    if getattr(interaction, "output_text", None):
        return interaction.output_text

    texts = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", "") == "model_output" and getattr(step, "content", None):
            for part in step.content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                elif hasattr(part, "text") and part.text:
                    texts.append(part.text)
    return "\n".join(texts)


class GeminiClient:
    """Cliente de la API Interactions de Gemini.

    Los reintentos del SDK están desactivados a propósito: si no, un 429 con un
    Retry-After largo dejaría el proceso dormido. La política de reintentos vive
    en un solo sitio (evals.py), no escondida en el SDK.
    """

    def __init__(self, model: str, timeout: float = 60.0, base_url: Optional[str] = None):
        from google import genai
        from google.genai import types

        http_options_kwargs: Dict[str, Any] = {"retry_options": types.HttpRetryOptions(attempts=0)}
        if base_url:
            http_options_kwargs["base_url"] = base_url
        self.model = model
        self.timeout = timeout
        self._client = genai.Client(http_options=types.HttpOptions(**http_options_kwargs))
        # Desactivar explícitamente reintentos en el recurso interactions
        # para evitar que utils.retry capture 429 y duerma con la cabecera Retry-After
        self._client.interactions.sdk_configuration.retry_config = None

    # -- llamadas -----------------------------------------------------------------
    def start(self, prompt, tools, system_instruction=None) -> ModelTurn:
        kwargs: Dict[str, Any] = {"model": self.model, "input": prompt, "tools": tools}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        return self._to_turn(self._create(**kwargs))

    def send_results(self, previous_id, results, tools, system_instruction=None) -> ModelTurn:
        payload = [
            {
                "type": "function_result",
                "name": r.name,
                "call_id": r.call_id,
                "result": [{"type": "text", "text": json.dumps(r.output, ensure_ascii=False)}],
            }
            for r in results
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model, "input": payload, "tools": tools,
            "previous_interaction_id": previous_id,
        }
        if system_instruction:  # igual que las herramientas: se reenvía en cada turno
            kwargs["system_instruction"] = system_instruction
        return self._to_turn(self._create(**kwargs))

    # -- internos -----------------------------------------------------------------
    def _to_turn(self, interaction) -> ModelTurn:
        steps = getattr(interaction, "steps", []) or []
        calls = [
            ToolCall(
                id=getattr(s, "id", ""),
                name=getattr(s, "name", ""),
                arguments=getattr(s, "arguments", {}) or {},
            )
            for s in steps
            if getattr(s, "type", "") == "function_call"
        ]
        usage = getattr(interaction, "usage", None)
        return ModelTurn(
            id=interaction.id,
            text=_interaction_text(interaction),
            tool_calls=calls,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )

    def _create(self, **kwargs):
        """Envuelve ÚNICAMENTE la llamada a la API y traduce sus fallos a ModelError:
        - 429: "LÍMITE/CUOTA:" más mensaje del servidor y Retry-After si existe.
        - Timeout: "TIMEOUT:".  - Error de red: "RED:".  - Otro error: "ERROR API:".
        """
        import httpx

        try:
            return self._client.interactions.create(timeout=self.timeout, **kwargs)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code is None and hasattr(e, "response"):
                status_code = getattr(e.response, "status_code", None)

            is_429 = status_code == 429 or type(e).__name__ == "RateLimitError" or "429" in str(e)
            if is_429:
                server_msg = None
                if hasattr(e, "body") and isinstance(e.body, dict):
                    error_field = e.body.get("error", {})
                    if isinstance(error_field, dict):
                        server_msg = error_field.get("message")
                    elif isinstance(error_field, str):
                        server_msg = error_field
                if not server_msg and hasattr(e, "message"):
                    server_msg = e.message
                if not server_msg:
                    server_msg = str(e)

                retry_after = None
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("retry-after")

                retry_info = f" (Retry-After: {retry_after})" if retry_after else ""
                raise ModelError(f"LÍMITE/CUOTA: {server_msg}{retry_info}") from e

            is_timeout = (
                type(e).__name__ in ("APITimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout")
                or isinstance(e, (TimeoutError, httpx.TimeoutException))
            )
            if is_timeout:
                raise ModelError(f"TIMEOUT: {getattr(e, 'message', str(e))}") from e

            is_network = (
                type(e).__name__ in ("APIConnectionError", "NetworkError", "ConnectError")
                or isinstance(e, (ConnectionError, httpx.NetworkError, httpx.RequestError))
            )
            if is_network:
                raise ModelError(f"RED: {getattr(e, 'message', str(e))}") from e

            raise ModelError(f"ERROR API: {type(e).__name__}: {getattr(e, 'message', str(e))}") from e
