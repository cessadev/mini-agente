"""Agent definitions. Adding an agent = adding ONE file in this folder.

Contract of an agent module (enforced by test_agents.py):
  TITLE, DEFAULT_BASE_DIR, DEFAULT_QUESTION   used by the CLI
  build_agent(base_dir=None, on_event=...)    returns an agentkit.Agent
"""
import importlib
import pkgutil
from pathlib import Path

DEFAULT_AGENT = "files"
_NOT_AGENTS = {"common"}


def available():
    return sorted(m.name for m in pkgutil.iter_modules([str(Path(__file__).parent)])
                  if m.name not in _NOT_AGENTS)


def load_agent(name: str):
    if name not in available():
        raise ValueError(f"Unknown agent: '{name}'. Available: {', '.join(available())}.")
    return importlib.import_module(f"agents.{name}")
