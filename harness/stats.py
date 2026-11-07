"""Paired comparisons and the one rule for all claims (SPEC.md sections 5 to 7).

Resampling uses numpy's default_rng with the study seed. A question's score is the mean over its samples.
"""
import numpy as np

from common import SEED

RESAMPLES = 10000
METRICS = {"pass": lambda r: float(r["passed"]), "no_loop": lambda r: 0.0 if r["cap_hit"] else 1.0}


def question_scores(rows, metric="pass"):
    per_question = {}
    for r in rows:
        per_question.setdefault(r["id"], []).append(METRICS[metric](r))
    return {q: float(np.mean(v)) for q, v in per_question.items()}


def paired_gains(a, b, metric="pass", same_seeds=True):
    """a minus b per question. Runs must cover the same questions and samples, with the same seeds unless told otherwise
    (noise repeats differ on purpose)."""
    keys = ("id", "sample", "seed") if same_seeds else ("id", "sample")
    if sorted(tuple(r[k] for k in keys) for r in a) != sorted(tuple(r[k] for k in keys) for r in b):
        raise ValueError("runs aren't paired: their questions, samples or seeds differ")
    sa, sb = question_scores(a, metric), question_scores(b, metric)
    return np.array([sa[q] - sb[q] for q in sorted(sa)])


def interval(gains):
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(gains), size=(RESAMPLES, len(gains)))
    lo, hi = np.percentile(gains[picks].mean(axis=1), [0.5, 99.5])
    return float(lo), float(hi)


def compare(a, b, metric="pass", same_seeds=True):
    gains = paired_gains(a, b, metric, same_seeds)
    lo, hi = interval(gains)
    return {"mean": float(gains.mean()), "lo": lo, "hi": hi, "questions": len(gains), "metric": metric}


def summarise(rows):
    n = len(rows)
    return {"answers": n, "pass": sum(r["passed"] for r in rows) / n, "cap_hit": sum(r["cap_hit"] for r in rows) / n,
            "no_answer": sum(r["no_answer"] for r in rows) / n, "grade_errors": sum(r["grade_error"] for r in rows),
            "mean_tokens": sum(r["tokens"] for r in rows) / n}


def best_by(runs, metric="pass"):
    """Name of the run with the highest mean score on the metric. runs maps name to score rows."""
    return max(sorted(runs), key=lambda k: float(np.mean(list(question_scores(runs[k], metric).values()))))


def verdict(vs_original, vs_random, vs_settings, damage):
    """The whole interval must clear zero for each gain. Damage is a whole interval below zero on any other set."""
    checks = {"gain": vs_original["lo"] > 0, "beats_random": vs_random["lo"] > 0, "beats_settings": vs_settings["lo"] > 0,
              "no_damage": not any(d["hi"] < 0 for d in damage.values())}
    return {"checks": checks, "finding": all(checks.values()),
            "damaged": sorted(k for k, d in damage.items() if d["hi"] < 0)}
