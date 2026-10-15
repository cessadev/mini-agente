"""The agent loop: model + tools + limits. It knows nothing about Gemini or files."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .model import ModelClient, ToolResult
from .tools import Tool, ToolRegistry


@dataclass
class AgentResult:
    answer: str
    status: str                      # "ok" | "max_iterations"
    iterations: int                  # rounds in which the model asked for tools
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # [{"name","arguments","result"}]
    total_tokens: Optional[int] = None  # sum over the turns; None if the provider does not report it


class Agent:
    """Model + tools + loop with an iteration cap.

    on_event: optional function that receives a dict for each step (observability).
    The agent prints nothing on its own; whoever uses it decides what to do with the events.
    """

    def __init__(
        self,
        model: ModelClient,
        tools: Union[ToolRegistry, Iterable[Tool]],
        system_instruction: Optional[str] = None,
        max_iterations: int = 8,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system_instruction = system_instruction
        self.max_iterations = max_iterations
        self._emit = on_event or (lambda event: None)

    def run(self, prompt: str) -> AgentResult:
        declarations = self.tools.declarations()
        tool_calls: List[Dict[str, Any]] = []
        tokens: List[int] = []

        def total() -> Optional[int]:
            return sum(tokens) if tokens else None

        def track(turn):
            if turn.total_tokens is not None:
                tokens.append(turn.total_tokens)
            return turn

        turn = track(self.model.start(prompt, declarations, self.system_instruction))

        for iteration in range(1, self.max_iterations + 1):
            # If the model asks for no functions, we have the final answer
            if not turn.tool_calls:
                self._emit({"type": "final", "turn": iteration, "text": turn.text})
                return AgentResult(turn.text, "ok", iteration - 1, tool_calls, total())

            self._emit({
                "type": "iteration", "iteration": iteration,
                "max": self.max_iterations, "calls": len(turn.tool_calls),
            })

            # Local execution of what the model asked for
            results: List[ToolResult] = []
            for call in turn.tool_calls:
                self._emit({"type": "tool_call", "name": call.name, "arguments": call.arguments})
                output = self.tools.execute(call.name, call.arguments)
                self._emit({"type": "tool_result", "name": call.name, "result": output})
                tool_calls.append({"name": call.name, "arguments": call.arguments, "result": output})
                results.append(ToolResult(call_id=call.id, name=call.name, output=output))

            # Iteration cap: checked before the next request
            if iteration >= self.max_iterations:
                self._emit({"type": "limit_reached", "max": self.max_iterations, "text": turn.text})
                return AgentResult(turn.text, "max_iterations", iteration, tool_calls, total())

            self._emit({"type": "sending_results"})
            turn = track(self.model.send_results(turn.id, results, declarations, self.system_instruction))

        # Not reachable: the loop always returns inside. Defense in case max_iterations < 1.
        return AgentResult("", "max_iterations", 0, tool_calls, total())
