"""Herramientas del agente.

Una herramienta es UNA sola cosa: función + descripción + parámetros. El esquema que ve
el modelo se genera a partir de la firma de la función, así no puede desincronizarse del
código que realmente se ejecuta (antes eran tres piezas escritas a mano por separado).
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
        """Lo único que el modelo ve de la herramienta: nombre, descripción y parámetros."""
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
    """Convierte una función en una Tool.

    `params` da la descripción de cada parámetro. Los tipos salen de las anotaciones
    (str, int, float, bool) y los parámetros sin valor por defecto son obligatorios.
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
                    f"{func.__name__}: el parámetro '{pname}' necesita una anotación "
                    f"de tipo str, int, float o bool."
                )
            prop: Dict[str, Any] = {"type": _JSON_TYPES[annotation]}
            if pname in params:
                prop["description"] = params[pname]
            properties[pname] = prop
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        unknown = set(params) - set(properties)
        if unknown:
            raise ValueError(f"{func.__name__}: descripciones para parámetros inexistentes: {sorted(unknown)}")

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return Tool(name or func.__name__, description, schema, func)

    return decorator


class ToolRegistry:
    """Catálogo de herramientas + despachador. Reemplaza al if/elif de execute_tool."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Herramienta duplicada: '{t.name}'.")
        self._tools[t.name] = t

    def names(self) -> List[str]:
        return list(self._tools)

    def declarations(self) -> List[Dict[str, Any]]:
        return [t.declaration() for t in self._tools.values()]

    def execute(self, name: str, args: Any) -> Dict[str, Any]:
        """Ejecuta lo que pidió el modelo. Nunca lanza: los errores vuelven como datos
        para que el modelo los vea y pueda reaccionar."""
        if not isinstance(args, dict):
            args = {}
        t = self._tools.get(name)
        if t is None:
            return {"error": f"Función no reconocida: '{name}'."}
        for p in t.required:
            if args.get(p) in (None, ""):
                return {"error": f"El argumento '{p}' es requerido para {name}."}
        kwargs = {k: v for k, v in args.items() if k in t.allowed}  # ignora argumentos inventados
        try:
            result = t.func(**kwargs)
        except Exception as e:  # red de seguridad: un bug en una herramienta no tumba el bucle
            return {"error": f"Error al ejecutar {name}: {e}"}
        return result if isinstance(result, dict) else {"result": result}
