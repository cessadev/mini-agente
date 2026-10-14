"""Restricción de rutas: la defensa real contra path traversal.

La salida del modelo es una entrada NO confiable. Por eso la ruta se resuelve primero
(resolve() colapsa los `..` y sigue los enlaces simbólicos) y solo después se comprueba
que sigue dentro de la carpeta permitida. Revisar la cadena buscando ".." sería eludible.
"""
from pathlib import Path


def resolve_safe_path(base_dir, rel_path: str = ".") -> Path:
    base = Path(base_dir).resolve()
    name = base.name

    if not rel_path or rel_path.strip() in (".", "./"):
        return base

    cleaned = rel_path.strip()

    # Normalizar si el modelo envía la ruta incluyendo la carpeta base
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith(f"{name}/"):
        cleaned = cleaned[len(name) + 1:]
    elif cleaned == name:
        cleaned = "."

    path_obj = Path(cleaned)
    if path_obj.is_absolute():
        raise PermissionError(f"Acceso denegado: rutas absolutas no permitidas ('{rel_path}').")

    target = (base / path_obj).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(
            f"Acceso denegado: '{rel_path}' intenta acceder fuera de la carpeta permitida ./{name}."
        )

    return target
