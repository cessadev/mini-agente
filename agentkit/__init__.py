"""agentkit: esqueleto mínimo y reutilizable para agentes con herramientas."""
from .agent import Agent, AgentResult
from .fs_tools import make_file_tools
from .model import GeminiClient, ModelClient, ModelError, ModelTurn, ToolCall, ToolResult
from .sandbox import resolve_safe_path
from .tools import Tool, ToolRegistry, tool

__all__ = [
    "Agent", "AgentResult", "GeminiClient", "ModelClient", "ModelError", "ModelTurn",
    "Tool", "ToolCall", "ToolRegistry", "ToolResult", "make_file_tools",
    "resolve_safe_path", "tool",
]
