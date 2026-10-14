"""Agente #1: responde preguntas sobre los archivos de una carpeta (solo lectura)."""
from agentkit import Agent, make_file_tools

from .common import MAX_ITERATIONS, ROOT, make_model, print_event, system_prompt

TITLE = "AGENTE GEMINI"
DEFAULT_BASE_DIR = (ROOT / "muestra").resolve()
DEFAULT_QUESTION = (
    "Examina los archivos de la carpeta ./muestra y proporciona un resumen completo: "
    "qué archivos existen, de qué trata el proyecto y qué funciones o clases tiene cada uno."
)
DEFAULT_SYSTEM_PROMPT = None  # sin instrucción de sistema por defecto


def build_agent(base_dir=None, on_event=print_event) -> Agent:
    """Aquí se decide QUÉ agente es: qué modelo, qué herramientas, qué instrucciones.

    base_dir: carpeta a la que se confinan las herramientas (por defecto, ./muestra).
    on_event: quién recibe los eventos; None = silencio (lo usa evals.py).
    """
    if base_dir is None:
        base_dir, examples = DEFAULT_BASE_DIR, ("README.md", "calculadora.py")
    else:
        examples = ()  # los ejemplos de nombres de archivo son de ./muestra
    tools = make_file_tools(base_dir, examples=examples)
    return Agent(make_model(), tools, system_instruction=system_prompt(DEFAULT_SYSTEM_PROMPT),
                 max_iterations=MAX_ITERATIONS, on_event=on_event)
