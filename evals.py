"""Agent evaluation: cases from evals.json -> results_<MODEL>[_<LABEL>].json

Environment variables:
  AGENT    agent to measure (agents/ folder); default "files". The cases of any other agent
           live in evals_<agent>.json and their results in results_<agent>_...
  MODEL    model to measure (it also names the results file)
  RUNS     repetitions per case (default 1)
  CASES    comma-separated ids to measure only some (saves quota)
  LABEL    experiment label; keeps results apart, e.g. LABEL=defensive
  DELAY    pause between calls (s)   RETRIES  retries on infrastructure errors
  REGRADE=1  re-grades answers already stored, without calling the API

The agent runs in this very process (not in a subprocess) so that the tool TRACE can be read:
what it tried to do, not only what it said.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from agents import DEFAULT_AGENT
from grading import grade, needs_trace

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

AGENT = os.getenv("AGENT", DEFAULT_AGENT)
CASES_FILE = ROOT / ("evals.json" if AGENT == DEFAULT_AGENT else f"evals_{AGENT}.json")
RUNS = int(os.getenv("RUNS", "1"))
MODEL = os.getenv("MODEL", "default")
LABEL = os.getenv("LABEL", "").strip()
DELAY = float(os.getenv("DELAY", "6"))
RETRIES = int(os.getenv("RETRIES", "2"))
ONLY = {c.strip() for c in os.getenv("CASES", "").split(",") if c.strip()}
_prefix = "" if AGENT == DEFAULT_AGENT else f"{AGENT}_"
OUT = Path(f"results_{_prefix}{MODEL}{'_' + LABEL if LABEL else ''}.json")


def load_cases():
    if not CASES_FILE.is_file():
        sys.exit(f"No cases for agent '{AGENT}': {CASES_FILE.name} is missing.")
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if ONLY:
        unknown = ONLY - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"CASES contains unknown ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in ONLY]
    for c in cases:
        if "dir" in c and not (FIXTURES / c["dir"]).is_dir():
            sys.exit(f"Case '{c['id']}' uses folder fixtures/{c['dir']}, which does not exist.")
    return cases


def plain(value):
    """Converts to plain JSON (the arguments returned by the SDK may be custom types)."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def run_once(case):
    """Returns (AgentResult, ''), or (None, message) on an infrastructure failure."""
    from agentkit import ModelError
    from agents import load_agent  # late import: REGRADE needs neither a key nor the network

    base_dir = FIXTURES / case["dir"] if "dir" in case else None
    try:
        return load_agent(AGENT).build_agent(base_dir=base_dir, on_event=None).run(case["question"]), ""
    except ModelError as e:
        return None, "error: " + str(e)[:300]
    except Exception as e:  # e.g. the API key is missing
        return None, f"error: {type(e).__name__}: {str(e)[:300]}"


def is_daily_quota(err):
    e = err.lower()
    return "rate limit/quota" in e and "per day" in e


def to_entry(result, case):
    """AgentResult -> storable entry. It keeps the trace and the raw answer so that it can be
    re-graded for free (REGRADE) when a criterion changes."""
    answer = result.answer if result.status == "ok" else None
    calls = [
        {"name": c["name"], "arguments": plain(c["arguments"]),
         "result_preview": json.dumps(plain(c["result"]), ensure_ascii=False)[:300]}
        for c in result.tool_calls
    ]
    ok, why = grade(answer, case, calls)
    return {"status": "PASS" if ok else "FAIL", "detail": why, "iters": result.iterations,
            "tokens": result.total_tokens, "answer": answer, "tool_calls": calls}


def regrade(done, cases):
    """Re-grades what is already stored with the current evals.json. It does not call the API."""
    by_id = {c["id"]: c for c in cases}
    not_regradable = []
    for key, entry in done.items():
        if entry.get("status") not in ("PASS", "FAIL"):
            continue
        case = by_id.get(key.rsplit("#", 1)[0])
        if case is None:
            continue
        if "answer" not in entry or (needs_trace(case) and "tool_calls" not in entry):
            not_regradable.append(key)
            continue
        ok, why = grade(entry["answer"], case, entry.get("tool_calls"))
        new_status = "PASS" if ok else "FAIL"
        if new_status != entry["status"]:
            print(f"{key}: {entry['status']} -> {new_status}")
        entry["status"], entry["detail"] = new_status, why
    return not_regradable


def summarize(done, cases):
    keys = [(c, f"{c['id']}#{r}") for c in cases for r in range(RUNS)]
    vals = [(c, done.get(k, {"status": "ERROR"})) for c, k in keys]
    n_pass = sum(v["status"] == "PASS" for _, v in vals)
    n_fail = sum(v["status"] == "FAIL" for _, v in vals)
    n_err = sum(v["status"] == "ERROR" for _, v in vals)
    tag = (f" [{AGENT}]" if AGENT != DEFAULT_AGENT else "") + (f" [{LABEL}]" if LABEL else "")
    print(f"\nModel: {MODEL}{tag} | PASS {n_pass} | FAIL {n_fail} | ERROR {n_err}")
    if n_pass + n_fail:
        print(f"Accuracy over valid answers: {n_pass}/{n_pass + n_fail}")

    groups = defaultdict(list)
    for c, v in vals:
        groups[c.get("group", "basic")].append(v)
    if len(groups) > 1:
        print("\nBy group (PASS/valid, mean iterations):")
        for g, vs in groups.items():
            valid = [v for v in vs if v["status"] in ("PASS", "FAIL")]
            if not valid:
                continue
            passed = sum(v["status"] == "PASS" for v in valid)
            mean_iters = sum(v.get("iters", 0) for v in valid) / len(valid)
            print(f"  {g:<12} {passed}/{len(valid)}   iters≈{mean_iters:.1f}")
    if n_err:
        print("\nThere are infrastructure ERRORS: repeat the same command; only those cases will be retried.")


def main():
    cases = load_cases()
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    if os.getenv("REGRADE") == "1":
        not_regradable = regrade(done, cases)
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        for key in not_regradable:
            print(f"{key}: no stored answer/trace (cannot be re-graded)")
        counts = [v.get("status") for v in done.values()]
        print(f"\nModel: {MODEL} | PASS {counts.count('PASS')} | "
              f"FAIL {counts.count('FAIL')} | ERROR {counts.count('ERROR')}")
        sys.exit(0)

    for case in cases:
        for run in range(RUNS):
            key = f"{case['id']}#{run}"
            if done.get(key, {}).get("status") in ("PASS", "FAIL"):
                continue  # already measured
            result, err = None, ""
            for attempt in range(RETRIES + 1):
                result, err = run_once(case)
                if result is not None or is_daily_quota(err):
                    break
                if attempt < RETRIES:
                    time.sleep(20 * (attempt + 1))
            if result is None and is_daily_quota(err):
                print(f"\nDAILY QUOTA EXHAUSTED for {MODEL}: {err}")
                print("Run stopped. What was measured is saved; repeat the command "
                      "when the quota resets, or use another model.")
                sys.exit(3)
            if result is None:
                done[key] = {"status": "ERROR", "detail": err, "iters": 0}
            else:
                done[key] = to_entry(result, case)
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
            d = done[key]
            print(f"{d['status']:<5} {key:<26} iters={d['iters']}  {d['detail']}")
            time.sleep(DELAY)

    summarize(done, cases)


if __name__ == "__main__":
    main()
