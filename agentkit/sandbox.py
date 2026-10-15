"""Path restriction: the real defense against path traversal.

The model's output is untrusted input. That is why the path is resolved first
(resolve() collapses `..` and follows symlinks) and only then checked to still be inside
the allowed folder. Checking the string for ".." would be bypassable.
"""
from pathlib import Path


def resolve_safe_path(base_dir, rel_path: str = ".") -> Path:
    base = Path(base_dir).resolve()
    name = base.name

    if not rel_path or rel_path.strip() in (".", "./"):
        return base

    cleaned = rel_path.strip()

    # Normalize if the model sends the path including the base folder
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
