"""Punto de entrada (CLI) de los agentes.

    python3 agente.py "pregunta"                  # agente por defecto: archivos
    AGENT=logs python3 agente.py "pregunta"       # otro agente (ver carpeta agents/)

La salida por pantalla y los códigos de salida son un contrato con evals.py (busca
"RESPUESTA FINAL DEL MODELO:" y trata el código 2 como fallo de infraestructura):
no los cambies sin cambiar evals.py.
"""
import os
import sys

from agentkit import ModelError
from agents import DEFAULT_AGENT, load_agent
from agents.common import MAX_ITERATIONS, MODEL, print_event  # noqa: F401  (re-exportados)
from agents.archivos import DEFAULT_BASE_DIR as BASE_DIR, build_agent  # noqa: F401  (compatibilidad)


def run_agent(prompt: str, agent_name: str = DEFAULT_AGENT) -> str:
    module = load_agent(agent_name)
    agent = module.build_agent()

    print("=" * 65)
    print(f"🤖 INICIANDO {module.TITLE}")
    print(f"Modelo: {MODEL}")
    print(f"Directorio restringido: {module.DEFAULT_BASE_DIR}")
    print(f"Límite máximo de iteraciones: {MAX_ITERATIONS}")
    print(f"Prompt: {prompt}")
    print("=" * 65)

    return agent.run(prompt).answer


def main():
    agent_name = os.getenv("AGENT", DEFAULT_AGENT)
    try:
        module = load_agent(agent_name)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else module.DEFAULT_QUESTION

    try:
        run_agent(user_prompt, agent_name)
    except ModelError as e:
        # Fallo de infraestructura: mensaje claro y código 2 (evals.py lo clasifica como ERROR)
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")


if __name__ == "__main__":
    main()
