"""Pieces of the application layer shared by ALL agents (they are not the core).

The core (agentkit) knows nothing about .env, the console or which provider is used by default;
that is decided here, once, so that each agent is just its definition.
"""
import os
from pathlib import Path
from typing import Optional

from agentkit import GeminiClient

ROOT = Path(__file__).resolve().parent.parent

# Optional loading of environment variables from .env (no extra dependencies)
_env_file = ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

MODEL = os.getenv("MODEL", "gemini-3.8-flash")
MAX_ITERATIONS = 8

try:
    API_TIMEOUT = float(os.getenv("AGENT_API_TIMEOUT", "60"))
except (ValueError, TypeError):
    API_TIMEOUT = 60.0


def make_model() -> GeminiClient:
    return GeminiClient(MODEL, timeout=API_TIMEOUT, base_url=os.getenv("AGENT_BASE_URL"))


def system_prompt(default: Optional[str] = None) -> Optional[str]:
    """System instruction of the agent. AGENT_SYSTEM_PROMPT takes precedence over the agent's
    default; if it is defined but empty it means 'no system instruction' (useful to compare
    with and without a prompt)."""
    if "AGENT_SYSTEM_PROMPT" in os.environ:
        return os.environ["AGENT_SYSTEM_PROMPT"] or None
    return default


def print_event(event: dict) -> None:
    """Translates the agent events into the console output."""
    kind = event["type"]
    if kind == "iteration":
        print(f"\n[Iteration {event['iteration']}/{event['max']}] "
              f"Function requests detected ({event['calls']}):")
    elif kind == "tool_call":
        print(f"  ⚡ Running: {event['name']}({event['arguments']})")
    elif kind == "tool_result":
        print(f"  📥 Result: {event['result']}")
    elif kind == "sending_results":
        print("  🔄 Sending results to the Gemini API...")
    elif kind == "final":
        print(f"\n[Turn {event['turn']}] The model requested no more functions.")
        print("\n" + "=" * 65)
        print("📝 FINAL MODEL ANSWER:")
        print("=" * 65)
        print(event["text"])
    elif kind == "limit_reached":
        print(f"\n⚠️ Iteration limit of {event['max']} reached without a final answer.")
        if event["text"]:
            print("\nLast text received:")
            print(event["text"])
