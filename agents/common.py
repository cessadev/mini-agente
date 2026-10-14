"""Piezas de la capa de aplicación que comparten TODOS los agentes (no son el núcleo).

El núcleo (agentkit) no sabe de .env, de consola ni de qué proveedor se usa por defecto;
eso se decide aquí, una sola vez, para que cada agente sea solo su definición.
"""
import os
from pathlib import Path
from typing import Optional

from agentkit import GeminiClient

ROOT = Path(__file__).resolve().parent.parent

# Carga de variables de entorno opcional desde .env (sin dependencias adicionales)
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
    """Instrucción de sistema del agente. AGENT_SYSTEM_PROMPT manda sobre el valor por defecto
    del agente; si está definida pero vacía, significa 'sin instrucción de sistema' (útil para
    comparar con y sin prompt)."""
    if "AGENT_SYSTEM_PROMPT" in os.environ:
        return os.environ["AGENT_SYSTEM_PROMPT"] or None
    return default


def print_event(event: dict) -> None:
    """Traduce los eventos del agente a la salida de consola de siempre."""
    kind = event["type"]
    if kind == "iteration":
        print(f"\n[Iteración {event['iteration']}/{event['max']}] "
              f"Solicitudes de función detectadas ({event['calls']}):")
    elif kind == "tool_call":
        print(f"  ⚡ Ejecutando: {event['name']}({event['arguments']})")
    elif kind == "tool_result":
        print(f"  📥 Resultado: {event['result']}")
    elif kind == "sending_results":
        print("  🔄 Enviando resultados a la Gemini API...")
    elif kind == "final":
        print(f"\n[Turno {event['turn']}] El modelo no solicitó más funciones.")
        print("\n" + "=" * 65)
        print("📝 RESPUESTA FINAL DEL MODELO:")
        print("=" * 65)
        print(event["text"])
    elif kind == "limit_reached":
        print(f"\n⚠️ Se alcanzó el límite de {event['max']} iteraciones sin respuesta final.")
        if event["text"]:
            print("\nÚltimo texto recibido:")
            print(event["text"])
