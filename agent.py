"""Entry point (CLI) for the agents.

    python3 agent.py "question"                 # default agent: files
    AGENT=logs python3 agent.py "question"      # another agent (see the agents/ folder)

The console output and the exit codes are a contract with the tests (test_cli.py and
test_fail_fast.py): exit code 2 means an infrastructure failure (quota, timeout, network,
API error). Do not change them without updating those tests.
"""
import os
import sys

from agentkit import ModelError
from agents import DEFAULT_AGENT, load_agent
from agents.common import MAX_ITERATIONS, MODEL


def run_agent(prompt: str, agent_name: str = DEFAULT_AGENT) -> str:
    module = load_agent(agent_name)
    agent = module.build_agent()

    print("=" * 65)
    print(f"🤖 STARTING {module.TITLE}")
    print(f"Model: {MODEL}")
    print(f"Restricted directory: {module.DEFAULT_BASE_DIR}")
    print(f"Max iterations: {MAX_ITERATIONS}")
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
        # Infrastructure failure: clear message and exit code 2 (evals.py reports the same
        # ModelError as an ERROR, never as an agent FAIL)
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nExecution interrupted by the user.")


if __name__ == "__main__":
    main()
