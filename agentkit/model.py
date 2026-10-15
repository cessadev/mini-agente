"""Model layer: isolates EVERYTHING provider-specific (today, Gemini).

The agent loop only knows these neutral types. To test the loop without an API the client is
replaced by a fake one; to change provider, another client is written.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class ModelError(Exception):
    """Infrastructure failure when calling the model (quota, timeout, network, API).

    str(e) is the message ready to be shown. The library raises; whoever decides to end
    the process (sys.exit) is the entry point, not the reusable code.
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
    """Extracts the answer text from a Gemini interaction."""
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
    """Client for the Gemini Interactions API.

    The SDK retries are disabled on purpose: otherwise a 429 with a long Retry-After
    would leave the process asleep. The retry policy lives in a single place (evals.py),
    not hidden inside the SDK.
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
        # Explicitly disable retries on the interactions resource so that
        # utils.retry does not catch a 429 and sleep following the Retry-After header
        self._client.interactions.sdk_configuration.retry_config = None

    # -- calls --------------------------------------------------------------------
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
        if system_instruction:  # same as the tools: it is re-sent on every turn
            kwargs["system_instruction"] = system_instruction
        return self._to_turn(self._create(**kwargs))

    # -- internals ----------------------------------------------------------------
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
        """Wraps ONLY the API call and translates its failures into ModelError:
        - 429: "RATE LIMIT/QUOTA:" plus the server message and Retry-After if present.
        - Timeout: "TIMEOUT:".  - Network error: "NETWORK:".  - Any other error: "API ERROR:".
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
                raise ModelError(f"RATE LIMIT/QUOTA: {server_msg}{retry_info}") from e

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
                raise ModelError(f"NETWORK: {getattr(e, 'message', str(e))}") from e

            raise ModelError(f"API ERROR: {type(e).__name__}: {getattr(e, 'message', str(e))}") from e
