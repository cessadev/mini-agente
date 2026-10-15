"""Contract of the CLI (agent.py) against a fake Gemini server: no network, no quota.

The console output is a contract (final-answer marker, "[Iteration", exit codes), so it is
guarded here to keep a format change from breaking things silently.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from fake_gemini import FakeGemini, text_response as R, tool_call_response as T

AGENT_PY = Path(__file__).resolve().parent / "agent.py"


def run_cli(responses, extra_env=None, prompt="test question"):
    with FakeGemini(responses) as srv:
        env = dict(os.environ, AGENT_BASE_URL=srv.base_url, GEMINI_API_KEY="clave_falsa", MODEL="modelo-falso")
        env.update(extra_env or {})
        p = subprocess.run([sys.executable, str(AGENT_PY), prompt],
                           capture_output=True, text=True, env=env, timeout=30)
        return p, srv.requests


class TestCliContract(unittest.TestCase):
    def test_normal_run_prints_markers_evals_rely_on(self):
        p, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "There are three files.")])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count("[Iteration"), 1)  # this is how iterations are counted
        self.assertIn("FINAL MODEL ANSWER:", p.stdout)
        self.assertTrue(p.stdout.rstrip().endswith("There are three files."))
        self.assertEqual(len(reqs), 2)

    def test_requests_declare_tools_and_chain_interactions(self):
        _, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "ok")])
        self.assertEqual([t["name"] for t in reqs[0]["tools"]], ["list_files", "read_file"])
        self.assertNotIn("previous_interaction_id", reqs[0])
        self.assertEqual(reqs[1]["previous_interaction_id"], "interaction_1")
        self.assertNotIn("system_instruction", reqs[0])  # by default, no system instruction

    def test_system_prompt_from_env_goes_on_every_request(self):
        _, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "ok")],
                          extra_env={"AGENT_SYSTEM_PROMPT": "You are read-only."})
        self.assertEqual([r.get("system_instruction") for r in reqs], ["You are read-only."] * 2)

    def test_path_traversal_is_blocked_end_to_end(self):
        p, reqs = run_cli([T(1, [("c1", "read_file", {"path": "../.gitignore"})]), R(2, "No puedo.")])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("denegado", p.stdout.lower())
        sent_back = json.dumps(reqs[1]["input"], ensure_ascii=False)
        self.assertIn("denegado", sent_back.lower())
        self.assertNotIn("__pycache__", sent_back)  # real content of the .gitignore: it must not come out

    def test_iteration_limit_has_no_final_answer_marker(self):
        responses = [T(i, [(f"c{i}", "list_files", {"path": "."})]) for i in range(1, 9)] + [R(99, "nunca")]
        p, reqs = run_cli(responses)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("Iteration limit of 8", p.stdout)
        self.assertNotIn("FINAL MODEL ANSWER:", p.stdout)
        self.assertEqual(p.stdout.count("[Iteration"), 8)
        self.assertEqual(len(reqs), 8)  # after the 8th round no more results are sent


if __name__ == "__main__":
    unittest.main()
