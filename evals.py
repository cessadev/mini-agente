import json, os, subprocess, sys, time
from pathlib import Path

RUNS = int(os.getenv("RUNS", "1"))
MODEL = os.getenv("MODEL", "default")
DELAY = float(os.getenv("DELAY", "6"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
RETRIES = int(os.getenv("RETRIES", "2"))
MARK = "RESPUESTA FINAL DEL MODELO:"
OUT = Path(f"resultados_{MODEL}.json")


def extract_answer(stdout):
    if MARK not in stdout:
        return None
    lines = stdout.split(MARK, 1)[1].strip().splitlines()
    if lines and set(lines[0].strip()) == {"="}:
        lines = lines[1:]
    return "\n".join(lines).strip()


def grade(answer, case):
    if answer is None:
        return False, "sin respuesta final"
    a = answer.lower()
    for s in case.get("all_of", []):
        if s.lower() not in a:
            return False, f"falta: {s}"
    any_of = case.get("any_of", [])
    if any_of and not any(s.lower() in a for s in any_of):
        return False, "ninguna de las esperadas"
    for s in case.get("none_of", []):
        if s.lower() in a:
            return False, f"contiene prohibido: {s}"
    return True, "ok"


def run_once(question):
    """Devuelve (stdout, error). error != '' significa fallo de infraestructura."""
    try:
        p = subprocess.run([sys.executable, "agente.py", question],
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "timeout (evals)"
    if p.returncode != 0:
        msg = p.stderr.strip()
        # código 2 = error controlado de agente.py: el mensaje útil está al inicio
        return None, "error: " + (msg[:300] if p.returncode == 2 else msg[-300:])
    return p.stdout, ""


def is_daily_quota(err):
    e = err.lower()
    return "límite/cuota" in e and "per day" in e


cases = json.loads(Path("evals.json").read_text(encoding="utf-8"))
done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

for case in cases:
    for run in range(RUNS):
        key = f"{case['id']}#{run}"
        if done.get(key, {}).get("status") in ("PASS", "FAIL"):
            continue  # ya medido
        out, err = None, ""
        for attempt in range(RETRIES + 1):
            out, err = run_once(case["question"])
            if out is not None or is_daily_quota(err):
                break
            if attempt < RETRIES:
                time.sleep(20 * (attempt + 1))
        if out is None and is_daily_quota(err):
            print(f"\nCUOTA DIARIA AGOTADA para {MODEL}: {err}")
            print("Ejecución detenida. Lo medido está guardado; repite el comando "
                  "cuando se reinicie la cuota, o usa otro modelo.")
            sys.exit(3)
        if out is None:
            done[key] = {"status": "ERROR", "detail": err, "iters": 0}
        else:
            ok, why = grade(extract_answer(out), case)
            done[key] = {"status": "PASS" if ok else "FAIL", "detail": why,
                         "iters": out.count("[Iteración")}
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        d = done[key]
        print(f"{d['status']:<5} {key:<18} iters={d['iters']}  {d['detail']}")
        time.sleep(DELAY)

vals = [done.get(f"{c['id']}#{r}", {"status": "ERROR"}) for c in cases for r in range(RUNS)]
n_pass = sum(v["status"] == "PASS" for v in vals)
n_fail = sum(v["status"] == "FAIL" for v in vals)
n_err = sum(v["status"] == "ERROR" for v in vals)
print(f"\nModelo: {MODEL} | PASS {n_pass} | FAIL {n_fail} | ERROR {n_err}")
if n_pass + n_fail:
    print(f"Precisión sobre respuestas válidas: {n_pass}/{n_pass + n_fail}")
if n_err:
    print("Hay ERRORES de infraestructura: repite el mismo comando, solo reintentará esos casos.")
