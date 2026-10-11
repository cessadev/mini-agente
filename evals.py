import json, os, subprocess, sys, time
from pathlib import Path

RUNS = int(os.getenv("RUNS", "1"))
MODEL = os.getenv("MODEL", "default")
MARK = "RESPUESTA FINAL DEL MODELO:"


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


cases = json.loads(Path("evals.json").read_text(encoding="utf-8"))
results = []
for case in cases:
    passes, iters, notes = 0, [], []
    for _ in range(RUNS):
        try:
            p = subprocess.run([sys.executable, "agente.py", case["question"]],
                               capture_output=True, text=True, timeout=180)
            out = p.stdout
            err = p.stderr[-150:] if p.returncode != 0 else ""
        except subprocess.TimeoutExpired:
            out, err = "", "timeout"
        ok, why = grade(extract_answer(out), case)
        passes += ok
        iters.append(out.count("[Iteración"))
        notes.append(why if not err else f"{why} | {err}")
        time.sleep(2)  # evita saturar la cuota gratuita
    results.append({"id": case["id"], "passes": passes, "runs": RUNS,
                    "avg_iters": sum(iters) / RUNS, "notes": notes})
    print(f"{'OK ' if passes == RUNS else 'FAIL'} {case['id']:<16} {passes}/{RUNS}  iters={sum(iters)/RUNS:.1f}  {notes[-1]}")

total = sum(r["passes"] for r in results)
print(f"\nModelo: {MODEL} | Aciertos: {total}/{len(cases) * RUNS}")
Path(f"resultados_{MODEL}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
