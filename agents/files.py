"""Agent #1: answers questions about the files of a folder (read-only)."""
from agentkit import Agent, make_file_tools

from .common import MAX_ITERATIONS, ROOT, make_model, print_event, system_prompt

TITLE = "GEMINI AGENT"
DEFAULT_BASE_DIR = (ROOT / "muestra").resolve()
DEFAULT_QUESTION = (
    "Examina los archivos de la carpeta ./muestra y proporciona un resumen completo: "
    "qué archivos existen, de qué trata el proyecto y qué funciones o clases tiene cada uno."
)
DEFAULT_SYSTEM_PROMPT = None  # no system instruction by default


def build_agent(base_dir=None, on_event=print_event) -> Agent:
    """This is where WHICH agent it is gets decided: which model, which tools, which instructions.

    base_dir: folder the tools are confined to (default: ./muestra).
    on_event: who receives the events; None = silence (used by evals.py).
    """
    if base_dir is None:
        base_dir, examples = DEFAULT_BASE_DIR, ("README.md", "calculadora.py")
    else:
        examples = ()  # the example file names belong to ./muestra
    tools = make_file_tools(base_dir, examples=examples)
    return Agent(make_model(), tools, system_instruction=system_prompt(DEFAULT_SYSTEM_PROMPT),
                 max_iterations=MAX_ITERATIONS, on_event=on_event)
