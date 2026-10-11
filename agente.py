import json
import os
import sys
from pathlib import Path
import httpx
from google import genai
from google.genai import types

# Carga de variables de entorno opcional desde .env (sin dependencias adicionales)
_env_file = Path(__file__).parent / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

# Configuración básica
MODEL = os.getenv("MODEL", "gemini-3.8-flash")
MAX_ITERATIONS = 8
BASE_DIR = (Path(__file__).parent / "muestra").resolve()

try:
    API_TIMEOUT = float(os.getenv("AGENT_API_TIMEOUT", "60"))
except (ValueError, TypeError):
    API_TIMEOUT = 60.0


# ==============================================================================
# Herramientas locales y Sandboxing seguro en ./muestra
# ==============================================================================

def _resolve_safe_path(rel_path: str = ".") -> Path:
    """
    Resuelve una ruta garantizando que permanezca estrictamente dentro de ./muestra.
    Evita cualquier vulnerabilidad de path traversal o rutas absolutas fuera del directorio.
    """
    if not rel_path or rel_path.strip() in (".", "./"):
        return BASE_DIR

    cleaned = rel_path.strip()

    # Normalizar si el modelo envía la ruta incluyendo la carpeta base 'muestra'
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("muestra/"):
        cleaned = cleaned[len("muestra/") :]
    elif cleaned == "muestra":
        cleaned = "."

    path_obj = Path(cleaned)
    if path_obj.is_absolute():
        raise PermissionError(f"Acceso denegado: rutas absolutas no permitidas ('{rel_path}').")

    target = (BASE_DIR / path_obj).resolve()

    try:
        target.relative_to(BASE_DIR)
    except ValueError:
        raise PermissionError(
            f"Acceso denegado: '{rel_path}' intenta acceder fuera de la carpeta permitida ./muestra."
        )

    return target


def list_files(path: str = ".") -> dict:
    """
    Lista los nombres de archivos y subdirectorios dentro de ./muestra.
    """
    try:
        target = _resolve_safe_path(path)
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


def read_file(path: str) -> dict:
    """
    Lee y devuelve el contenido de texto de un archivo dentro de ./muestra.
    """
    try:
        target = _resolve_safe_path(path)
        if not target.exists():
            return {"error": f"El archivo '{path}' no existe."}
        if not target.is_file():
            return {"error": f"La ruta '{path}' no es un archivo regular."}

        content = target.read_text(encoding="utf-8", errors="replace")
        return {"content": content}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error al leer el archivo: {str(e)}"}


def execute_tool(name: str, args: dict) -> dict:
    """
    Ejecuta localmente la función solicitada por el modelo.
    """
    if not isinstance(args, dict):
        args = {}

    if name == "list_files":
        path = args.get("path", ".")
        return list_files(path=path)
    elif name == "read_file":
        path = args.get("path")
        if not path:
            return {"error": "El argumento 'path' es requerido para read_file."}
        return read_file(path=path)
    else:
        return {"error": f"Función no reconocida: '{name}'."}


# ==============================================================================
# Declaración de herramientas (según documentación oficial de Gemini API)
# ==============================================================================

LIST_FILES_TOOL = {
    "type": "function",
    "name": "list_files",
    "description": (
        "Lista los archivos y directorios dentro de la carpeta ./muestra o de una subcarpeta relativa. "
        "Usa path='.' para listar la raíz de ./muestra."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Ruta relativa dentro de ./muestra a listar. Por defecto '.' (raíz de ./muestra).",
            }
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "name": "read_file",
    "description": (
        "Lee el contenido completo de un archivo de texto dentro de la carpeta ./muestra."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Ruta relativa del archivo a leer dentro de ./muestra (por ejemplo: 'README.md', 'calculadora.py').",
            }
        },
        "required": ["path"],
    },
}

TOOLS = [LIST_FILES_TOOL, READ_FILE_TOOL]


# ==============================================================================
# Helper para obtener texto de respuesta del modelo
# ==============================================================================

def get_interaction_text(interaction) -> str:
    """Extrae el texto de respuesta del modelo."""
    if getattr(interaction, "output_text", None):
        return interaction.output_text

    texts = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", "") == "model_output" and getattr(step, "content", None):
            for part in step.content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                elif hasattr(part, "text") and part.text:
                    texts.append(part.text)
    return "\n".join(texts)


# ==============================================================================
# Llamada a la API con manejo de errores y fail-fast
# ==============================================================================

def _create_interaction(client: genai.Client, **kwargs):
    """
    Envuelve ÚNICAMENTE la llamada a client.interactions.create.
    Aplica el timeout configurado y maneja excepciones específicas de la API:
    - 429: "LÍMITE/CUOTA:" más mensaje del servidor y Retry-After si existe.
    - Timeout: "TIMEOUT:".
    - Error de red: "RED:".
    - Otro error de API: "ERROR API:" con tipo y mensaje.
    En todos estos casos finaliza con sys.exit(2).
    """
    try:
        return client.interactions.create(timeout=API_TIMEOUT, **kwargs)
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code is None and hasattr(e, "response"):
            status_code = getattr(e.response, "status_code", None)

        is_429 = (
            status_code == 429
            or type(e).__name__ == "RateLimitError"
            or "429" in str(e)
        )

        if is_429:
            server_msg = None
            if hasattr(e, "body") and isinstance(e.body, dict):
                error_field = e.body.get("error", {})
                if isinstance(error_field, dict):
                    server_msg = error_field.get("message")
                elif isinstance(error_field, str):
                    server_msg = error_field
            if not server_msg and hasattr(e, "message"):
                server_msg = e.message
            if not server_msg:
                server_msg = str(e)

            retry_after = None
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("retry-after")

            retry_info = f" (Retry-After: {retry_after})" if retry_after else ""
            print(f"LÍMITE/CUOTA: {server_msg}{retry_info}", file=sys.stderr)
            sys.exit(2)

        is_timeout = (
            type(e).__name__ in ("APITimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout")
            or isinstance(e, (TimeoutError, httpx.TimeoutException))
        )
        if is_timeout:
            print(f"TIMEOUT: {getattr(e, 'message', str(e))}", file=sys.stderr)
            sys.exit(2)

        is_network = (
            type(e).__name__ in ("APIConnectionError", "NetworkError", "ConnectError")
            or isinstance(e, (ConnectionError, httpx.NetworkError, httpx.RequestError))
        )
        if is_network:
            print(f"RED: {getattr(e, 'message', str(e))}", file=sys.stderr)
            sys.exit(2)

        # Cualquier otro error de la API
        print(f"ERROR API: {type(e).__name__}: {getattr(e, 'message', str(e))}", file=sys.stderr)
        sys.exit(2)


# ==============================================================================
# Bucle del Agente
# ==============================================================================

def run_agent(prompt: str) -> str:
    """
    Bucle principal del agente con Gemini API:
    - Configura el cliente con AGENT_BASE_URL (si está definida) y desactiva reintentos.
    - Envía el prompt y las herramientas disponibles con AGENT_API_TIMEOUT.
    - Ejecuta hasta 8 iteraciones:
      - Si el modelo responde directamente, devuelve la respuesta final.
      - Si el modelo pide list_files o read_file, ejecuta las funciones localmente,
        devuelve los resultados a la API y repite el bucle.
    """
    http_options_kwargs = {
        "retry_options": types.HttpRetryOptions(attempts=0),
    }
    agent_base_url = os.getenv("AGENT_BASE_URL")
    if agent_base_url:
        http_options_kwargs["base_url"] = agent_base_url

    http_options = types.HttpOptions(**http_options_kwargs)
    client = genai.Client(http_options=http_options)

    # Desactivar explícitamente reintentos en el recurso interactions
    # para evitar que utils.retry capture 429 y duerma con la cabecera Retry-After
    client.interactions.sdk_configuration.retry_config = None

    print("=" * 65)
    print("🤖 INICIANDO AGENTE GEMINI")
    print(f"Modelo: {MODEL}")
    print(f"Directorio restringido: {BASE_DIR}")
    print(f"Límite máximo de iteraciones: {MAX_ITERATIONS}")
    print(f"Prompt: {prompt}")
    print("=" * 65)

    # Llamada inicial con las herramientas
    interaction = _create_interaction(
        client,
        model=MODEL,
        input=prompt,
        tools=TOOLS,
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        steps = getattr(interaction, "steps", []) or []
        function_calls = [s for s in steps if getattr(s, "type", "") == "function_call"]

        # Si el modelo no pide funciones, tenemos la respuesta final
        if not function_calls:
            print(f"\n[Turno {iteration}] El modelo no solicitó más funciones.")
            print("\n" + "=" * 65)
            print("📝 RESPUESTA FINAL DEL MODELO:")
            print("=" * 65)
            final_text = get_interaction_text(interaction)
            print(final_text)
            return final_text

        print(f"\n[Iteración {iteration}/{MAX_ITERATIONS}] Solicitudes de función detectadas ({len(function_calls)}):")

        # Ejecución local de las funciones solicitadas
        function_results = []
        for fc in function_calls:
            name = getattr(fc, "name", "")
            args = getattr(fc, "arguments", {}) or {}
            call_id = getattr(fc, "id", "")

            print(f"  ⚡ Ejecutando: {name}({args})")
            tool_result = execute_tool(name, args)
            print(f"  📥 Resultado: {tool_result}")

            function_results.append({
                "type": "function_result",
                "name": name,
                "call_id": call_id,
                "result": [{"type": "text", "text": json.dumps(tool_result, ensure_ascii=False)}],
            })

        # Comprobar límite de 8 iteraciones antes de la siguiente petición
        if iteration >= MAX_ITERATIONS:
            print(f"\n⚠️ Se alcanzó el límite de {MAX_ITERATIONS} iteraciones sin respuesta final.")
            last_text = get_interaction_text(interaction)
            if last_text:
                print("\nÚltimo texto recibido:")
                print(last_text)
            return last_text

        # Enviar los resultados al modelo usando previous_interaction_id
        print("  🔄 Enviando resultados a la Gemini API...")
        interaction = _create_interaction(
            client,
            model=MODEL,
            input=function_results,
            tools=TOOLS,
            previous_interaction_id=interaction.id,
        )

    return ""


def main():
    default_prompt = (
        "Examina los archivos de la carpeta ./muestra y proporciona un resumen completo: "
        "qué archivos existen, de qué trata el proyecto y qué funciones o clases tiene cada uno."
    )
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt

    try:
        run_agent(user_prompt)
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")


if __name__ == "__main__":
    main()
