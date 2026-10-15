"""Tests of the core (agentkit). All of them without network and without quota: the model is an in-memory script.

This tests YOUR CODE (loop, tools, limits, sandbox), not the real model.
The behaviour of the real model is measured with evals.py.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from agentkit import (Agent, ModelTurn, Tool, ToolCall, ToolRegistry, make_file_tools,
                      resolve_safe_path, tool)


# ---------------------------------------------------------------------------
# Fake model: returns scripted turns and records what it receives
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
    """Turn in which the model asks for tools: calls = (call_id, name, args)."""
    return ModelTurn(turn_id, "", [ToolCall(c, n, a) for c, n, a in calls], tokens)


def says(turn_id, text, tokens=None):
    return ModelTurn(turn_id, text, [], tokens)


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------
class TestToolDeclarations(unittest.TestCase):
    def test_generated_schema_equals_the_old_handwritten_one(self):
        """What the model sees of the tools did not change after the refactor (golden)."""
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
        @tool(description="Add", params={"a": "first", "b": "second"})
        def sumar(a: int, b: float = 1.0, flag: bool = False) -> dict:
            return {"r": a + b}

        props = sumar.parameters["properties"]
        self.assertEqual(props["a"], {"type": "integer", "description": "first"})
        self.assertEqual(props["b"], {"type": "number", "description": "second"})
        self.assertEqual(props["flag"], {"type": "boolean"})
        self.assertEqual(sumar.parameters["required"], ["a"])

    def test_decorator_rejects_unannotated_parameter(self):
        with self.assertRaises(TypeError):
            @tool(description="x")
            def mala(a):  # no annotation
                return {}

    def test_decorator_rejects_description_for_unknown_parameter(self):
        with self.assertRaises(ValueError):
            @tool(description="x", params={"fantasma": "no existe"})
            def f(a: str) -> dict:
                return {}


class TestRegistry(unittest.TestCase):
    def setUp(self):
        @tool(description="Echo", params={"text": "what to repeat"})
        def echo(text: str) -> dict:
            return {"echo": text}

        @tool(description="Breaks")
        def broken() -> dict:
            raise RuntimeError("bug in the tool")

        self.registry = ToolRegistry([echo, broken])

    def test_executes_known_tool(self):
        self.assertEqual(self.registry.execute("echo", {"text": "hello"}), {"echo": "hello"})

    def test_unknown_tool_returns_error_as_data(self):
        self.assertIn("no reconocida", self.registry.execute("borrar_todo", {})["error"])

    def test_missing_required_argument_returns_error(self):
        self.assertIn("'text' es requerido", self.registry.execute("echo", {})["error"])
        self.assertIn("'text' es requerido", self.registry.execute("echo", {"text": ""})["error"])

    def test_invented_arguments_are_ignored(self):
        self.assertEqual(self.registry.execute("echo", {"text": "x", "invented": 1}), {"echo": "x"})

    def test_non_dict_arguments_do_not_crash(self):
        self.assertIn("error", self.registry.execute("echo", "i am not a dict"))

    def test_tool_exception_becomes_error_not_crash(self):
        self.assertIn("bug in the tool", self.registry.execute("broken", {})["error"])

    def test_duplicate_names_are_rejected(self):
        @tool(description="another", name="echo")
        def another(text: str) -> dict:
            return {}
        with self.assertRaises(ValueError):
            self.registry.register(another)


# ---------------------------------------------------------------------------
# Path sandbox
# ---------------------------------------------------------------------------
class TestSandbox(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.base = root / "muestra"
        (self.base / "sub").mkdir(parents=True)
        (self.base / "a.txt").write_text("content a", encoding="utf-8")
        (self.base / "sub" / "b.txt").write_text("content b", encoding="utf-8")
        self.secret = root / "secret.txt"
        self.secret.write_text("SECRET-KEY", encoding="utf-8")
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
        for p in ("../secret.txt", "sub/../../secret.txt", "./../secret.txt"):
            with self.assertRaises(PermissionError, msg=p):
                resolve_safe_path(self.base, p)

    def test_blocks_absolute_paths(self):
        with self.assertRaises(PermissionError):
            resolve_safe_path(self.base, str(self.secret))

    def test_blocks_symlink_escaping_the_folder(self):
        """This is why resolve() is done BEFORE checking: looking at the string would not detect this."""
        link = self.base / "shortcut.txt"
        try:
            os.symlink(self.secret, link)
        except (OSError, NotImplementedError):
            self.skipTest("the system does not allow creating symlinks")
        with self.assertRaises(PermissionError):
            resolve_safe_path(self.base, "shortcut.txt")


class TestFileTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "docs"
        self.base.mkdir()
        (self.base / "short.txt").write_text("hello", encoding="utf-8")
        (self.base / "long.txt").write_text("x" * 100, encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_read_file_truncates_and_says_so(self):
        registry = ToolRegistry(make_file_tools(self.base, max_chars=10))
        out = registry.execute("read_file", {"path": "long.txt"})
        self.assertEqual(out, {"content": "x" * 10, "truncated": True, "limit_chars": 10})

    def test_read_file_normal_output_has_no_extra_keys(self):
        registry = ToolRegistry(make_file_tools(self.base, max_chars=10))
        self.assertEqual(registry.execute("read_file", {"path": "short.txt"}), {"content": "hello"})

    def test_list_files_marks_directories(self):
        (self.base / "folder").mkdir()
        registry = ToolRegistry(make_file_tools(self.base))
        self.assertEqual(registry.execute("list_files", {})["files"], ["folder/", "long.txt", "short.txt"])

    def test_descriptions_follow_the_configured_folder(self):
        tools = {t.name: t for t in make_file_tools(self.base)}
        self.assertIn("./docs", tools["read_file"].description)


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
def echo_tool():
    @tool(description="Echo", params={"text": "text"})
    def echo(text: str) -> dict:
        return {"echo": text}
    return echo


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer_without_tools(self):
        model = ScriptedModel([says("t1", "hello")])
        result = Agent(model, [echo_tool()]).run("say hello")
        self.assertEqual((result.answer, result.status, result.iterations), ("hello", "ok", 0))
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(model.sent, [])

    def test_one_tool_round_then_answer(self):
        model = ScriptedModel([asks("t1", ("c1", "echo", {"text": "ping"})), says("t2", "pong")])
        result = Agent(model, [echo_tool()]).run("do an echo")
        self.assertEqual((result.answer, result.status, result.iterations), ("pong", "ok", 1))
        self.assertEqual(result.tool_calls, [{"name": "echo", "arguments": {"text": "ping"}, "result": {"echo": "ping"}}])
        previous_id, results, _ = model.sent[0]
        self.assertEqual(previous_id, "t1")
        self.assertEqual((results[0].call_id, results[0].name, results[0].output), ("c1", "echo", {"echo": "ping"}))

    def test_parallel_calls_are_all_executed_and_sent_together(self):
        model = ScriptedModel([
            asks("t1", ("c1", "echo", {"text": "a"}), ("c2", "echo", {"text": "b"})),
            says("t2", "done"),
        ])
        result = Agent(model, [echo_tool()]).run("two at once")
        self.assertEqual(result.iterations, 1)  # one round, even though it has two requests
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual([r.call_id for r in model.sent[0][1]], ["c1", "c2"])

    def test_iteration_limit_stops_the_loop(self):
        turns = [asks(f"t{i}", (f"c{i}", "echo", {"text": "x"})) for i in range(1, 10)]
        model = ScriptedModel(turns)
        result = Agent(model, [echo_tool()], max_iterations=3).run("endless")
        self.assertEqual((result.status, result.iterations), ("max_iterations", 3))
        self.assertEqual(len(result.tool_calls), 3)
        self.assertEqual(len(model.sent), 2)  # after the 3rd round nothing else is sent

    def test_tokens_are_summed_across_turns(self):
        model = ScriptedModel([asks("t1", ("c1", "echo", {"text": "x"}), tokens=10), says("t2", "ok", tokens=25)])
        self.assertEqual(Agent(model, [echo_tool()]).run("q").total_tokens, 35)

    def test_tokens_are_none_when_the_provider_does_not_report(self):
        self.assertIsNone(Agent(ScriptedModel([says("t1", "ok")]), [echo_tool()]).run("q").total_tokens)

    def test_system_instruction_is_sent_on_every_turn(self):
        model = ScriptedModel([asks("t1", ("c1", "echo", {"text": "x"})), says("t2", "ok")])
        Agent(model, [echo_tool()], system_instruction="Solo lectura.").run("q")
        self.assertEqual(model.started[2], "Solo lectura.")
        self.assertEqual(model.sent[0][2], "Solo lectura.")

    def test_declarations_are_passed_to_the_model(self):
        model = ScriptedModel([says("t1", "ok")])
        Agent(model, [echo_tool()]).run("q")
        self.assertEqual([d["name"] for d in model.started[1]], ["echo"])

    def test_events_follow_the_loop(self):
        events = []
        model = ScriptedModel([asks("t1", ("c1", "echo", {"text": "x"})), says("t2", "end")])
        Agent(model, [echo_tool()], on_event=events.append).run("q")
        self.assertEqual([e["type"] for e in events],
                         ["iteration", "tool_call", "tool_result", "sending_results", "final"])

    def test_path_traversal_attempt_is_blocked_and_leaves_a_trace(self):
        """With the fake model it is tested that YOUR CODE stops the escape. Whether the REAL model
        obeys a malicious order or not is another question: the evals answer it."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "muestra"
            base.mkdir()
            (Path(tmp) / "secret.txt").write_text("SECRET-KEY", encoding="utf-8")
            model = ScriptedModel([
                asks("t1", ("c1", "read_file", {"path": "../secret.txt"})),
                says("t2", "No puedo leer eso."),
            ])
            result = Agent(model, make_file_tools(base)).run("read ../secret.txt")

        attempt = result.tool_calls[0]
        self.assertIn("denegado", attempt["result"]["error"].lower())
        self.assertNotIn("SECRET-KEY", json.dumps(result.tool_calls))
        self.assertNotIn("SECRET-KEY", json.dumps([r.output for r in model.sent[0][1]]))


if __name__ == "__main__":
    unittest.main()
