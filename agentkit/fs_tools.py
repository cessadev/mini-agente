"""Herramientas de solo lectura sobre una carpeta. Mínimo privilegio: no escriben nada."""
from pathlib import Path
from typing import Sequence

from .sandbox import resolve_safe_path
from .tools import Tool, tool


def make_file_tools(base_dir, max_chars: int = 20_000, examples: Sequence[str] = ()) -> "list[Tool]":
    """Devuelve [list_files, read_file] confinadas a `base_dir`.

    max_chars: tope de lo que read_file devuelve al modelo (evita llenar el contexto).
    examples:  nombres de archivo de ejemplo para la descripción de read_file.
    """
    base = Path(base_dir).resolve()
    rel = f"./{base.name}"
    ejemplos = ""
    if examples:
        ejemplos = " (por ejemplo: " + ", ".join(f"'{e}'" for e in examples) + ")"

    @tool(
        description=(
            f"Lista los archivos y directorios dentro de la carpeta {rel} o de una subcarpeta relativa. "
            f"Usa path='.' para listar la raíz de {rel}."
        ),
        params={"path": f"Ruta relativa dentro de {rel} a listar. Por defecto '.' (raíz de {rel})."},
    )
    def list_files(path: str = ".") -> dict:
        try:
            target = resolve_safe_path(base, path)
            if not target.exists():
                return {"error": f"La ruta '{path}' no existe."}
            if not target.is_dir():
                return {"error": f"La ruta '{path}' no es un directorio."}

            items = []
            for item in sorted(target.iterdir()):
                items.append(item.name + ("/" if item.is_dir() else ""))
            return {"files": items}
        except PermissionError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Error al listar archivos: {str(e)}"}

    @tool(
        description=f"Lee el contenido completo de un archivo de texto dentro de la carpeta {rel}.",
        params={"path": f"Ruta relativa del archivo a leer dentro de {rel}{ejemplos}."},
    )
    def read_file(path: str) -> dict:
        try:
            target = resolve_safe_path(base, path)
            if not target.exists():
                return {"error": f"El archivo '{path}' no existe."}
            if not target.is_file():
                return {"error": f"La ruta '{path}' no es un archivo regular."}

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars + 1)  # un carácter de más para saber si hubo recorte
            if len(content) > max_chars:
                return {"content": content[:max_chars], "truncated": True, "limit_chars": max_chars}
            return {"content": content}
        except PermissionError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Error al leer el archivo: {str(e)}"}

    return [list_files, read_file]
