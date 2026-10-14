"""Pruebas del núcleo (agentkit). Todas sin red y sin cuota: el modelo es un guion en memoria.

Esto prueba TU CÓDIGO (bucle, herramientas, límites, sandbox), no al modelo real.
El comportamiento del modelo real se mide con evals.py.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from agentkit import (Agent, ModelTurn, Tool, ToolCall, ToolRegistry, make_file_tools,
                      resolve_safe_path, tool)


# ---------------------------------------------------------------------------
# Modelo falso: devuelve turnos guionizados y registra lo que recibe
# ---------------------------------------------------------------------------
class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.started = None
        self.sent = []

    def start(self, prompt, tools, system_instruction):
        self.started = (prompt, tools, system_instruction)
        return self.turns.pop(0)

    def send_results(self, previous_id, results, tools, system_instruction):
        self.sent.append((previous_id, results, system_instruction))
        return self.turns.pop(0)


def asks(turn_id, *calls, tokens=None):
    """Turno en el que el modelo pide herramientas: calls = (call_id, name, args)."""
    return ModelTurn(turn_id, "", [ToolCall(c, n, a) for c, n, a in calls], tokens)


def says(turn_id, text, tokens=None):
    return ModelTurn(turn_id, text, [], tokens)


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------
class TestToolDeclarations(unittest.TestCase):
    def test_generated_schema_equals_the_old_handwritten_one(self):
        """Lo que el modelo ve de las herramientas no cambió tras el refactor (golden)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "muestra"
            base.mkdir()
            tools = make_file_tools(base, examples=("README.md", "calculadora.py"))
        declarations = {t.name: t.declaration() for t in tools}

        self.assertEqual(declarations["list_files"], {
            "type": "function",
            "name": "list_files",
            "description": (
                "Lista los archivos y directorios dentro de la carpeta ./muestra o de una subcarpeta relativa. "
                "Usa path='.' para listar la raíz de ./muestra."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": "Ruta relativa dentro de ./muestra a listar. Por defecto '.' (raíz de ./muestra).",
                }},
            },
        })
        self.assertEqual(declarations["read_file"], {
            "type": "function",
            "name": "read_file",
            "description": "Lee el contenido completo de un archivo de texto dentro de la carpeta ./muestra.",
            "parameters": {
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": ("Ruta relativa del archivo a leer dentro de ./muestra "
                                    "(por ejemplo: 'README.md', 'calculadora.py')."),
                }},
                "required": ["path"],
            },
        })

    def test_decorator_builds_types_and_required_from_signature(self):
        @tool(description="Suma", params={"a": "primero", "b": "segundo"})
        def sumar(a: int, b: float = 1.0, flag: bool = False) -> dict:
            return {"r": a + b}

        props = sumar.parameters["properties"]
        self.assertEqual(props["a"], {"type": "integer", "description": "primero"})
        self.assertEqual(props["b"], {"type": "number", "description": "segundo"})
        self.assertEqual(props["flag"], {"type": "boolean"})
        self.assertEqual(sumar.parameters["required"], ["a"])

    def test_decorator_rejects_unannotated_parameter(self):
        with self.assertRaises(TypeError):
            @tool(description="x")
            def mala(a):  # sin anotación
                return {}

    def test_decorator_rejects_description_for_unknown_parameter(self):
        with self.assertRaises(ValueError):
            @tool(description="x", params={"fantasma": "no existe"})
            def f(a: str) -> dict:
                return {}


class TestRegistry(unittest.TestCase):
    def setUp(self):
        @tool(description="Eco", params={"texto": "qué repetir"})
        def eco(texto: str) -> dict:
            return {"eco": texto}

        @tool(description="Rompe")
        def rota() -> dict:
            raise RuntimeError("bug en la herramienta")

        self.registry = ToolRegistry([eco, rota])

    def test_executes_known_tool(self):
        self.assertEqual(self.registry.execute("eco", {"texto": "hola"}), {"eco": "hola"})

    def test_unknown_tool_returns_error_as_data(self):
        self.assertIn("no reconocida", self.registry.execute("borrar_todo", {})["error"])

    def test_missing_required_argument_returns_error(self):
        self.assertIn("'texto' es requerido", self.registry.execute("eco", {})["error"])
        self.assertIn("'texto' es requerido", self.registry.execute("eco", {"texto": ""})["error"])

    def test_invented_arguments_are_ignored(self):
        self.assertEqual(self.registry.execute("eco", {"texto": "x", "inventado": 1}), {"eco": "x"})

    def test_non_dict_arguments_do_not_crash(self):
        self.assertIn("error", self.registry.execute("eco", "no soy un dict"))

    def test_tool_exception_becomes_error_not_crash(self):
        self.assertIn("bug en la herramienta", self.registry.execute("rota", {})["error"])

    def test_duplicate_names_are_rejected(self):
        @tool(description="otra", name="eco")
        def otra(texto: str) -> dict:
            return {}
        with self.assertRaises(ValueError):
            self.registry.register(otra)


# ---------------------------------------------------------------------------
# Sandbox de rutas
# ---------------------------------------------------------------------------
class TestSandbox(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.base = root / "muestra"
        (self.base / "sub").mkdir(parents=True)
        (self.base / "a.txt").write_text("contenido a", encoding="utf-8")
        (self.base / "sub" / "b.txt").write_text("contenido b", encoding="utf-8")
        self.secret = root / "secreto.txt"
        self.secret.write_text("CLAVE-SECRETA", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_allows_files_inside(self):
        self.assertEqual(resolve_safe_path(self.base, "a.txt"), (self.base / "a.txt").resolve())
        self.assertEqual(resolve_safe_path(self.base, "sub/b.txt"), (self.base / "sub" / "b.txt").resolve())

    def test_normalizes_dot_and_base_name_prefix(self):
        base = self.base.resolve()
        self.assertEqual(resolve_safe_path(self.base, "."), base)
        self.assertEqual(resolve_safe_path(self.base, ""), base)
        self.assertEqual(resolve_safe_path(self.base, "./a.txt"), base / "a.txt")
        self.assertEqual(resolve_safe_path(self.base, "muestra/a.txt"), base / "a.txt")
        self.assertEqual(resolve_safe_path(self.base, "muestra"), base)

    def test_blocks_parent_traversal(self):
        for p in ("../secreto.txt", "sub/../../secreto.txt", "./../secreto.txt"):
            with self.assertRaises(PermissionError, msg=p):
                resolve_safe_path(self.base, p)

    def test_blocks_absolute_paths(self):
        with self.assertRaises(PermissionError):
            resolve_safe_path(self.base, str(self.secret))

    def test_blocks_symlink_escaping_the_folder(self):
        """Por esto se hace resolve() ANTES de comprobar: mirar la cadena no detectaría esto."""
        link = self.base / "atajo.txt"
        try:
            os.symlink(self.secret, link)
        except (OSError, NotImplementedError):
            self.skipTest("el sistema no permite crear enlaces simbólicos")
        with self.assertRaises(PermissionError):
            resolve_safe_path(self.base, "atajo.txt")


class TestFileTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "docs"
        self.base.mkdir()
        (self.base / "corto.txt").write_text("hola", encoding="utf-8")
        (self.base / "largo.txt").write_text("x" * 100, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_read_file_truncates_and_says_so(self):
        registry = ToolRegistry(make_file_tools(self.base, max_chars=10))
        out = registry.execute("read_file", {"path": "largo.txt"})
        self.assertEqual(out, {"content": "x" * 10, "truncated": True, "limit_chars": 10})

    def test_read_file_normal_output_has_no_extra_keys(self):
        registry = ToolRegistry(make_file_tools(self.base, max_chars=10))
        self.assertEqual(registry.execute("read_file", {"path": "corto.txt"}), {"content": "hola"})

    def test_list_files_marks_directories(self):
        (self.base / "carpeta").mkdir()
        registry = ToolRegistry(make_file_tools(self.base))
        self.assertEqual(registry.execute("list_files", {})["files"], ["carpeta/", "corto.txt", "largo.txt"])

    def test_descriptions_follow_the_configured_folder(self):
        tools = {t.name: t for t in make_file_tools(self.base)}
        self.assertIn("./docs", tools["read_file"].description)


# ---------------------------------------------------------------------------
# El bucle del agente
# ---------------------------------------------------------------------------
def echo_tool():
    @tool(description="Eco", params={"texto": "texto"})
    def eco(texto: str) -> dict:
        return {"eco": texto}
    return eco


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer_without_tools(self):
        model = ScriptedModel([says("t1", "hola")])
        result = Agent(model, [echo_tool()]).run("saluda")
        self.assertEqual((result.answer, result.status, result.iterations), ("hola", "ok", 0))
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(model.sent, [])

    def test_one_tool_round_then_answer(self):
        model = ScriptedModel([asks("t1", ("c1", "eco", {"texto": "ping"})), says("t2", "pong")])
        result = Agent(model, [echo_tool()]).run("haz eco")
        self.assertEqual((result.answer, result.status, result.iterations), ("pong", "ok", 1))
        self.assertEqual(result.tool_calls, [{"name": "eco", "arguments": {"texto": "ping"}, "result": {"eco": "ping"}}])
        previous_id, results, _ = model.sent[0]
        self.assertEqual(previous_id, "t1")
        self.assertEqual((results[0].call_id, results[0].name, results[0].output), ("c1", "eco", {"eco": "ping"}))

    def test_parallel_calls_are_all_executed_and_sent_together(self):
        model = ScriptedModel([
            asks("t1", ("c1", "eco", {"texto": "a"}), ("c2", "eco", {"texto": "b"})),
            says("t2", "listo"),
        ])
        result = Agent(model, [echo_tool()]).run("dos a la vez")
        self.assertEqual(result.iterations, 1)  # una ronda, aunque tenga dos peticiones
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual([r.call_id for r in model.sent[0][1]], ["c1", "c2"])

    def test_iteration_limit_stops_the_loop(self):
        turns = [asks(f"t{i}", (f"c{i}", "eco", {"texto": "x"})) for i in range(1, 10)]
        model = ScriptedModel(turns)
        result = Agent(model, [echo_tool()], max_iterations=3).run("sin fin")
        self.assertEqual((result.status, result.iterations), ("max_iterations", 3))
        self.assertEqual(len(result.tool_calls), 3)
        self.assertEqual(len(model.sent), 2)  # tras la 3.ª ronda ya no se envía nada más

    def test_tokens_are_summed_across_turns(self):
        model = ScriptedModel([asks("t1", ("c1", "eco", {"texto": "x"}), tokens=10), says("t2", "ok", tokens=25)])
        self.assertEqual(Agent(model, [echo_tool()]).run("q").total_tokens, 35)

    def test_tokens_are_none_when_the_provider_does_not_report(self):
        self.assertIsNone(Agent(ScriptedModel([says("t1", "ok")]), [echo_tool()]).run("q").total_tokens)

    def test_system_instruction_is_sent_on_every_turn(self):
        model = ScriptedModel([asks("t1", ("c1", "eco", {"texto": "x"})), says("t2", "ok")])
        Agent(model, [echo_tool()], system_instruction="Solo lectura.").run("q")
        self.assertEqual(model.started[2], "Solo lectura.")
        self.assertEqual(model.sent[0][2], "Solo lectura.")

    def test_declarations_are_passed_to_the_model(self):
        model = ScriptedModel([says("t1", "ok")])
        Agent(model, [echo_tool()]).run("q")
        self.assertEqual([d["name"] for d in model.started[1]], ["eco"])

    def test_events_follow_the_loop(self):
        events = []
        model = ScriptedModel([asks("t1", ("c1", "eco", {"texto": "x"})), says("t2", "fin")])
        Agent(model, [echo_tool()], on_event=events.append).run("q")
        self.assertEqual([e["type"] for e in events],
                         ["iteration", "tool_call", "tool_result", "sending_results", "final"])

    def test_path_traversal_attempt_is_blocked_and_leaves_a_trace(self):
        """Con el modelo falso se prueba que TU CÓDIGO frena el escape. Si el modelo REAL
        obedece o no una orden maliciosa es otra pregunta: la responden las evals."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "muestra"
            base.mkdir()
            (Path(tmp) / "secreto.txt").write_text("CLAVE-SECRETA", encoding="utf-8")
            model = ScriptedModel([
                asks("t1", ("c1", "read_file", {"path": "../secreto.txt"})),
                says("t2", "No puedo leer eso."),
            ])
            result = Agent(model, make_file_tools(base)).run("lee ../secreto.txt")

        attempt = result.tool_calls[0]
        self.assertIn("denegado", attempt["result"]["error"].lower())
        self.assertNotIn("CLAVE-SECRETA", json.dumps(result.tool_calls))
        self.assertNotIn("CLAVE-SECRETA", json.dumps([r.output for r in model.sent[0][1]]))


if __name__ == "__main__":
    unittest.main()
