"""Definiciones de agentes. Añadir un agente = añadir UN archivo en esta carpeta.

Contrato de un módulo de agente (lo vigila test_agents.py):
  TITLE, DEFAULT_BASE_DIR, DEFAULT_QUESTION   para la CLI
  build_agent(base_dir=None, on_event=...)    devuelve un agentkit.Agent
"""
import importlib
import pkgutil
from pathlib import Path

DEFAULT_AGENT = "archivos"
_NOT_AGENTS = {"common"}


def available():
    return sorted(m.name for m in pkgutil.iter_modules([str(Path(__file__).parent)])
                  if m.name not in _NOT_AGENTS)


def load_agent(name: str):
    if name not in available():
        raise ValueError(f"Agente desconocido: '{name}'. Disponibles: {', '.join(available())}.")
    return importlib.import_module(f"agents.{name}")
