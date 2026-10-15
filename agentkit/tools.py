"""Agent tools.

A tool is ONE single thing: function + description + parameters. The schema the model sees
is generated from the function signature, so it cannot get out of sync with the code that
actually runs (before, they were three hand-written pieces kept separately).
"""
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, get_type_hints

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema de tipo "object"
    func: Callable[..., Any]

    def declaration(self) -> Dict[str, Any]:
        """All the model sees of the tool: name, description and parameters."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def required(self) -> List[str]:
        return list(self.parameters.get("required", []))

    @property
    def allowed(self) -> set:
        return set(self.parameters.get("properties", {}))


def tool(description: str, params: Optional[Dict[str, str]] = None, name: Optional[str] = None):
    """Turns a function into a Tool.

    `params` gives the description of each parameter. The types come from the annotations
    (str, int, float, bool) and the parameters without a default value are required.
    """
    params = params or {}

    def decorator(func: Callable[..., Any]) -> Tool:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for pname, p in sig.parameters.items():
            annotation = hints.get(pname)
            if annotation not in _JSON_TYPES:
                raise TypeError(
                    f"{func.__name__}: parameter '{pname}' needs a type annotation "
                    f"of str, int, float or bool."
                )
            prop: Dict[str, Any] = {"type": _JSON_TYPES[annotation]}
            if pname in params:
                prop["description"] = params[pname]
            properties[pname] = prop
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        unknown = set(params) - set(properties)
        if unknown:
            raise ValueError(f"{func.__name__}: descriptions for nonexistent parameters: {sorted(unknown)}")

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return Tool(name or func.__name__, description, schema, func)

    return decorator


class ToolRegistry:
    """Tool catalog + dispatcher. Replaces the old if/elif of execute_tool."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool: '{t.name}'.")
        self._tools[t.name] = t

    def names(self) -> List[str]:
        return list(self._tools)

    def declarations(self) -> List[Dict[str, Any]]:
        return [t.declaration() for t in self._tools.values()]

    def execute(self, name: str, args: Any) -> Dict[str, Any]:
        """Runs what the model asked for. It never raises: errors come back as data
        so that the model sees them and can react."""
        if not isinstance(args, dict):
            args = {}
        t = self._tools.get(name)
        if t is None:
            return {"error": f"Función no reconocida: '{name}'."}
        for p in t.required:
            if args.get(p) in (None, ""):
                return {"error": f"El argumento '{p}' es requerido para {name}."}
        kwargs = {k: v for k, v in args.items() if k in t.allowed}  # ignores invented arguments
        try:
            result = t.func(**kwargs)
        except Exception as e:  # safety net: a bug in a tool does not bring down the loop
            return {"error": f"Error al ejecutar {name}: {e}"}
        return result if isinstance(result, dict) else {"result": result}
