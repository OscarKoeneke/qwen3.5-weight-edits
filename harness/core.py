"""Cells, passes and run records, and the noise, grid, explore and confirm flows (SPEC.md sections 3 to 8).

A cell is one set answered once under one group of settings. A pass is the cells for one weights folder, run by one worker
process. Every pass goes in PASSES.log. Summaries and terminal output hold counts, scores and hashes, never question text.
"""
import contextlib
import fcntl
import gzip
import json
import os
import subprocess
import sys
import time
from collections import Counter, namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

import edits
import grade
import stats
from claims import BadClaim, parse_claim
from common import (GRID, HARNESS, ML, MODEL_DIR, REAL_ROOT, RUNS, SAMPLES, SEED_SCHEME, SETS, SETTINGS, append_line,
                    harness_hashes, manifest, read_lines, seed_for, sha256, utcnow)
from questions import (UntouchedLocked, close_attempt, exploring_ids, exploring_questions, open_attempt, read_attempts,
                       untouched_questions)

ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
ORIGINAL_PARAMS = RUNS / "original_params.json"
PASSES = RUNS / "PASSES.log"
BASELINE = RUNS / "baseline.json"
NOISE_REPEATS = 5
UNTOUCHED_ORIGINAL = RUNS / "untouched_original.json"
CONTROL_SEEDS = (1, 2, 3)
CONFIRM_SETS = SETS
GRADE_WORKERS = 8

Smoke = namedtuple("Smoke", "questions max_tokens")


@dataclass
class Weights:
    label: str
    dir: Path
    sha: str
    script: str = None
    script_sha: str = None
    footprint: dict = None


def original_weights():
    return Weights("original", MODEL_DIR, manifest()["model"]["files"][edits.WEIGHTS_FILE])


def build_weights(label, original, changed, fp, script=None, script_sha=None):
    out = RUNS / "tmp-weights" / label
    edits.remove_model(out)
    return Weights(label, out, edits.write_model(original, changed, out), script, script_sha, fp)


def gpu_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    name, uuid, driver = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": name, "uuid": uuid, "driver": driver}


def versions():
    return {"python": sys.version.split()[0],
            **{p: metadata.version(p) for p in ("vllm", "torch", "transformers", "math-verify", "lm-eval")}}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def settings_for(smoke):
    return dict(SETTINGS, max_tokens=smoke.max_tokens) if smoke else dict(SETTINGS)


def take(questions, smoke):
    return questions[:smoke.questions] if smoke else questions


def counts_as_evidence(smoke):
    """A smoke run can stand in for a real one only in a stand-in study folder."""
    return not (smoke and REAL_ROOT)


def preflight(sets):
    if "ifeval" in sets:
        grade.require_punkt()


def make_cell(name, set_name, questions, offset, settings, ticket=None):
    if ticket is None:
        allowed = exploring_ids(set_name)
        if any(q["id"] not in allowed for q in questions):
            raise UntouchedLocked(f"{name}: an untouched question reached a cell with no ticket")
    requests = [{"id": q["id"], "sample": s, "seed": seed_for(q["id"], s, offset), "prompt": q["prompt"]}
                for q in questions for s in range(SAMPLES)]
    return {"name": name, "set": set_name, "offset": offset, "settings": settings, "requests": requests,
            "gold": {q["id"]: q["gold"] for q in questions}}


def new_run_dir(kind, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = RUNS / f"{kind}_{stamp}_{label}"
    run_dir.mkdir(parents=True)
    return run_dir


@contextlib.contextmanager
def gpu_lock():
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / ".gpu.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another harness run holds the GPU")
        yield


def run_worker(weights, cells, run_dir):
    if weights.footprint and not ORIGINAL_PARAMS.exists():
        run_worker(original_weights(), [], new_run_dir("params", "original"))
    job = {"model": str(weights.dir), "run_dir": str(run_dir), "is_original": weights.footprint is None,
           "original_params": str(ORIGINAL_PARAMS), "expected_groups": edits.expected_groups(weights.footprint),
           "cells": [{k: c[k] for k in ("name", "requests", "settings")} for c in cells]}
    write_json(run_dir / "job.json", job)
    with open(run_dir / "worker.log", "a") as log:
        code = subprocess.run([sys.executable, str(HARNESS / "worker.py"), str(run_dir / "job.json")],
                              env={**os.environ, **ENV}, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"worker stopped with exit {code}. See {run_dir / 'worker.log'}")


def read_outputs(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def grade_cell(cell, run_dir):
    outputs = read_outputs(run_dir / f"outputs-{cell['name']}.jsonl.gz")
    rows = grade.grade_outputs(cell["set"], cell["gold"], outputs, workers=GRADE_WORKERS)
    write_json(run_dir / f"scores-{cell['name']}.json", rows)
    return rows


def load_rows(run, cell):
    return read_json(RUNS / run / f"scores-{cell}.json")


def run_pass(kind, weights, cells, generate, smoke=None, run_dir=None):
    run_dir = Path(run_dir) if run_dir else new_run_dir(kind, weights.label)
    write_json(run_dir / "config.json", {
        "kind": kind, "time": utcnow(), "root": str(ML), "smoke": list(smoke) if smoke else None,
        "weights": {"label": weights.label, "sha256": weights.sha, "script": str(weights.script), "script_sha256": weights.script_sha,
                    "footprint": weights.footprint},
        "cells": [{"name": c["name"], "set": c["set"], "offset": c["offset"], "settings": c["settings"],
                   "answers": len(c["requests"]), "ids": sorted(c["gold"]) if kind != "confirm" else None} for c in cells],
        "splits": {n: {p: manifest()["sets"][n][p]["sha256"] for p in ("exploring", "untouched")} for n in SETS},
        "seed_scheme": SEED_SCHEME, "samples": SAMPLES, "env": ENV, "gpu": gpu_info(), "versions": versions(),
        "harness": harness_hashes()})
    start = time.time()
    with gpu_lock():
        generate(weights, cells, run_dir)
    results = {c["name"]: grade_cell(c, run_dir) for c in cells}
    summary = {name: stats.summarise(rows) for name, rows in results.items()}
    write_json(run_dir / "summary.json", {"cells": summary, "speed": read_lines(run_dir / "cells.log")})
    append_line(PASSES, {"run": run_dir.name, "kind": kind, "time": utcnow(), "wall_s": round(time.time() - start, 1),
                         "weights": weights.label, "weights_sha256": weights.sha, "script_sha256": weights.script_sha,
                         "smoke": bool(smoke), "gpu": gpu_info()["uuid"], "scores": summary})
    return run_dir, results


def noise_report(results, sets, repeats):
    report = {}
    for s in sets:
        runs = [results[f"{s}-o{o}"] for o in range(repeats)]
        scores = [stats.summarise(r) for r in runs]
        pairs = [stats.compare(runs[i], runs[j], same_seeds=False) for i in range(repeats) for j in range(i + 1, repeats)]
        passes, caps = [x["pass"] for x in scores], [x["cap_hit"] for x in scores]
        report[s] = {"pass_by_repeat": passes, "pass_sd": float(np.std(passes, ddof=1)) if repeats > 1 else None,
                     "cap_hit_by_repeat": caps, "cap_hit_sd": float(np.std(caps, ddof=1)) if repeats > 1 else None,
                     "mean_tokens": float(np.mean([x["mean_tokens"] for x in scores])), "null_pairs": pairs,
                     "null_pairs_clearing_zero": sum(1 for p in pairs if p["lo"] > 0 or p["hi"] < 0)}
    return report


def noise(repeats=5, sets=SETS, smoke=None, resume=None, generate=None):
    preflight(sets)
    questions = {s: take(exploring_questions(s), smoke) for s in sets}
    cells = [make_cell(f"{s}-o{o}", s, questions[s], o, settings_for(smoke)) for o in range(repeats) for s in sets]
    run_dir, results = run_pass("noise", original_weights(), cells, generate or run_worker, smoke, resume)
    report = noise_report(results, sets, repeats)
    write_json(run_dir / "noise_report.json", report)
    if counts_as_evidence(smoke):
        write_json(BASELINE, {"run": run_dir.name, "cells": {s: f"{s}-o0" for s in sets}, "gpu_uuid": gpu_info()["uuid"]})
    return report


def noise_repeats():
    report = read_json(RUNS / read_json(BASELINE)["run"] / "noise_report.json")
    return min(len(report[s]["pass_by_repeat"]) if s in report else 0 for s in SETS)


def grid(set_name="math500", smoke=None, generate=None, resume=None):
    if not BASELINE.exists():
        raise SystemExit("the settings grid reuses the noise run for its default point, so the noise run comes first")
    preflight([set_name])
    base = read_json(BASELINE)
    questions = take(exploring_questions(set_name), smoke)
    points = [p for p in GRID if p != (SETTINGS["presence_penalty"], SETTINGS["repetition_penalty"])]
    cells = [make_cell(f"{set_name}-p{p}-r{r}", set_name, questions, 0,
                       dict(settings_for(smoke), presence_penalty=p, repetition_penalty=r)) for p, r in points]
    run_dir, results = run_pass("grid", original_weights(), cells, generate or run_worker, smoke, resume)
    rows = {(p, r): results[f"{set_name}-p{p}-r{r}"] for p, r in points}
    rows[(SETTINGS["presence_penalty"], SETTINGS["repetition_penalty"])] = load_rows(base["run"], base["cells"][set_name])
    report = {"run": run_dir.name, "points": [{"presence_penalty": p, "repetition_penalty": r, **stats.summarise(v)}
                                              for (p, r), v in sorted(rows.items())], "best": {}}
    for metric in stats.METRICS:
        p, r = stats.best_by(rows, metric)
        report["best"][metric] = {"presence_penalty": p, "repetition_penalty": r}
    if counts_as_evidence(smoke):
        write_json(RUNS / f"grid_{set_name}.json", report)
    return report


def versus_baseline(set_name, rows):
    if not BASELINE.exists():
        return "no noise run on record yet"
    base = read_json(BASELINE)
    if base["gpu_uuid"] != gpu_info()["uuid"]:
        return "refused: the baseline ran on a different GPU"
    if set_name not in base["cells"]:
        return "the noise run didn't cover this set"
    try:
        return stats.compare(rows, load_rows(base["run"], base["cells"][set_name]))
    except ValueError:
        return "the baseline covers different questions"


def explore(script, sets=("math500",), smoke=None, generate=None, keep_weights=False):
    preflight(sets)
    script = Path(script).resolve()
    original = edits.load_original()
    changed = edits.run_script(script, original)
    fp = edits.footprint(original, changed)
    weights = build_weights(f"{script.stem}-{sha256(script)[:8]}", original, changed, fp, str(script), sha256(script))
    cells = [make_cell(s, s, take(exploring_questions(s), smoke), 0, settings_for(smoke)) for s in sets]
    try:
        run_dir, results = run_pass("explore", weights, cells, generate or run_worker, smoke)
    finally:
        if not keep_weights:
            edits.remove_model(weights.dir)
    done = sum(1 for p in read_lines(PASSES) if p["kind"] == "explore" and not p["smoke"])
    return {"run": run_dir.name, "explore_passes_so_far": done, "script_sha256": weights.script_sha, "footprint": fp,
            "sets": {s: {**stats.summarise(results[s]), "vs_original": versus_baseline(s, results[s])} for s in sets}}


def prepare_claim(path, smoke=None):
    """Everything that can refuse a claim, done before the untouched half opens."""
    if smoke and REAL_ROOT:
        raise SystemExit("a smoke confirmation runs only in a stand-in study folder")
    claim = parse_claim(Path(path))
    if any(a["claim"] == claim["id"] for a in read_attempts()):
        raise UntouchedLocked(f"{claim['id']} already has an attempt on record. A new try needs a new claim id.")
    if not BASELINE.exists():
        raise BadClaim("no noise run on record")
    if read_json(BASELINE)["gpu_uuid"] != gpu_info()["uuid"]:
        raise BadClaim("the noise run was on a different GPU, so this card needs its own noise run first")
    if REAL_ROOT and noise_repeats() < NOISE_REPEATS:
        raise BadClaim(f"the noise run has fewer than {NOISE_REPEATS} repeats on record, so finish it before any confirmation")
    grid_path = RUNS / f"grid_{claim['primary_set']}.json"
    if not grid_path.exists():
        raise BadClaim(f"no settings grid on record for {claim['primary_set']}")
    preflight(CONFIRM_SETS)
    original = edits.load_original()
    changed = edits.run_script(claim["script"], original)
    fp = edits.footprint(original, changed)
    controls = {seed: edits.random_control(original, fp, seed) for seed in CONTROL_SEEDS}
    return {"claim": claim, "original": original, "changed": changed, "footprint": fp, "controls": controls,
            "best_setting": read_json(grid_path)["best"][claim["metric"]]}


def claimcheck(path):
    plan = prepare_claim(path)
    return {"claim": plan["claim"]["id"], "note_sha256": plan["claim"]["note_sha"], "primary": plan["claim"]["primary_set"],
            "metric": plan["claim"]["metric"], "edit": plan["footprint"], "best_setting": plan["best_setting"],
            "random_controls": {s: c[1] for s, c in plan["controls"].items()}}


def confirm(path, smoke=None, generate=None):
    generate = generate or run_worker
    plan = prepare_claim(path, smoke)
    claim = plan["claim"]
    cid, primary, metric = claim["id"], claim["primary_set"], claim["metric"]
    others = [s for s in CONFIRM_SETS if s != primary]

    ticket = open_attempt(cid, claim["note_sha"], claim["script_sha"])
    questions = {s: take(untouched_questions(s, ticket), smoke) for s in CONFIRM_SETS}
    base = settings_for(smoke)

    def cell(label, s, settings=base):
        return make_cell(f"{label}-{s}", s, questions[s], 0, settings, ticket)

    def run_edited(label, changed, fp, sets):
        weights = build_weights(f"{cid}-{label}", plan["original"], changed, fp, str(claim["script"]), claim["script_sha"])
        try:
            return run_pass("confirm", weights, [cell(label, s) for s in sets], generate, smoke)[1]
        finally:
            edits.remove_model(weights.dir)

    stored = read_json(UNTOUCHED_ORIGINAL) if UNTOUCHED_ORIGINAL.exists() else {}
    best = plan["best_setting"]
    default = best == {k: SETTINGS[k] for k in ("presence_penalty", "repetition_penalty")}
    cells = [cell("original", s) for s in CONFIRM_SETS if s not in stored]
    if not default:
        cells.append(cell("settings", primary, dict(base, **best)))
    results = {}
    if cells:
        run_dir, results = run_pass("confirm", original_weights(), cells, generate, smoke)
        stored.update({c["set"]: {"run": run_dir.name, "cell": c["name"]} for c in cells if c["name"].startswith("original-")})
        write_json(UNTOUCHED_ORIGINAL, stored)
    original_rows = {s: load_rows(stored[s]["run"], stored[s]["cell"]) for s in CONFIRM_SETS}
    settings_rows = original_rows[primary] if default else results[f"settings-{primary}"]

    edited = run_edited("edited", plan["changed"], plan["footprint"], [primary] + others)
    random_rows = {seed: run_edited(f"random{seed}", c[0], c[1], [primary])[f"random{seed}-{primary}"]
                   for seed, c in plan["controls"].items()}

    primary_rows = edited[f"edited-{primary}"]
    vs_original = stats.compare(primary_rows, original_rows[primary], metric)
    best_random = stats.best_by(random_rows, metric)
    vs_random = stats.compare(primary_rows, random_rows[best_random], metric)
    vs_settings = stats.compare(primary_rows, settings_rows, metric)
    damage = {s: stats.compare(edited[f"edited-{s}"], original_rows[s]) for s in others}
    result = stats.verdict(vs_original, vs_random, vs_settings, damage)
    report = {"claim": cid, "note_sha256": claim["note_sha"], "script_sha256": claim["script_sha"], "primary": primary,
              "metric": metric, "smoke": bool(smoke), "vs_original": vs_original, "vs_best_random": vs_random,
              "best_random_seed": best_random, "vs_best_setting": vs_settings, "best_setting": best, "damage": damage,
              "random_edits": {s: stats.summarise(r) for s, r in random_rows.items()}, **result}
    write_json(RUNS / f"confirm_{cid}.json", report)
    close_attempt(ticket, {"finding": result["finding"], "checks": result["checks"]})
    return report


def status():
    passes = read_lines(PASSES)
    return {"root": str(ML), "passes": dict(Counter(p["kind"] + ("-smoke" if p["smoke"] else "") for p in passes)),
            "baseline": read_json(BASELINE)["run"] if BASELINE.exists() else None,
            "attempts": [(a["claim"], a["event"], a["time"]) for a in read_attempts()]}
