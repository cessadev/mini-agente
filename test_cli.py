"""Contrato del CLI (agente.py) contra un servidor Gemini falso: sin red y sin cuota.

evals.py depende de esta salida (marcador de respuesta final, "[Iteración", código 2),
así que se vigila aquí para que un cambio de formato no rompa las evals en silencio.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from fake_gemini import FakeGemini, text_response as R, tool_call_response as T

AGENTE_PY = Path(__file__).resolve().parent / "agente.py"


def run_cli(responses, extra_env=None, prompt="pregunta de prueba"):
    with FakeGemini(responses) as srv:
        env = dict(os.environ, AGENT_BASE_URL=srv.base_url, GEMINI_API_KEY="clave_falsa", MODEL="modelo-falso")
        env.update(extra_env or {})
        p = subprocess.run([sys.executable, str(AGENTE_PY), prompt],
                           capture_output=True, text=True, env=env, timeout=30)
        return p, srv.requests


class TestCliContract(unittest.TestCase):
    def test_normal_run_prints_markers_evals_rely_on(self):
        p, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "Hay tres archivos.")])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.count("[Iteración"), 1)  # evals.py cuenta iteraciones así
        self.assertIn("RESPUESTA FINAL DEL MODELO:", p.stdout)
        self.assertTrue(p.stdout.rstrip().endswith("Hay tres archivos."))
        self.assertEqual(len(reqs), 2)

    def test_requests_declare_tools_and_chain_interactions(self):
        _, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "ok")])
        self.assertEqual([t["name"] for t in reqs[0]["tools"]], ["list_files", "read_file"])
        self.assertNotIn("previous_interaction_id", reqs[0])
        self.assertEqual(reqs[1]["previous_interaction_id"], "interaction_1")
        self.assertNotIn("system_instruction", reqs[0])  # por defecto, sin instrucción de sistema

    def test_system_prompt_from_env_goes_on_every_request(self):
        _, reqs = run_cli([T(1, [("c1", "list_files", {"path": "."})]), R(2, "ok")],
                          extra_env={"AGENT_SYSTEM_PROMPT": "Eres de solo lectura."})
        self.assertEqual([r.get("system_instruction") for r in reqs], ["Eres de solo lectura."] * 2)

    def test_path_traversal_is_blocked_end_to_end(self):
        p, reqs = run_cli([T(1, [("c1", "read_file", {"path": "../.gitignore"})]), R(2, "No puedo.")])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("denegado", p.stdout.lower())
        sent_back = json.dumps(reqs[1]["input"], ensure_ascii=False)
        self.assertIn("denegado", sent_back.lower())
        self.assertNotIn("__pycache__", sent_back)  # contenido real del .gitignore: no debe salir

    def test_iteration_limit_has_no_final_answer_marker(self):
        responses = [T(i, [(f"c{i}", "list_files", {"path": "."})]) for i in range(1, 9)] + [R(99, "nunca")]
        p, reqs = run_cli(responses)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("límite de 8 iteraciones", p.stdout)
        self.assertNotIn("RESPUESTA FINAL DEL MODELO:", p.stdout)
        self.assertEqual(p.stdout.count("[Iteración"), 8)
        self.assertEqual(len(reqs), 8)  # tras la 8.ª ronda no se envían más resultados


if __name__ == "__main__":
    unittest.main()
