"""The only way to get questions, and the lock on the untouched halves (SPEC.md section 3).

exploring_questions() is open. untouched_questions() needs a ticket, and the only source of a ticket is open_attempt(),
which writes the attempt to CONFIRM_ATTEMPTS.log first. Each claim id gets one attempt, so a crash means a new claim id,
visible in the log. No other module reads an untouched split file.
"""
import json

import pandas as pd

from common import MATH_SUFFIX, ML, RUNS, SPLITS, append_line, manifest, read_lines, utcnow

ATTEMPTS = RUNS / "CONFIRM_ATTEMPTS.log"
_KEY = object()


class UntouchedLocked(Exception):
    pass


class Ticket:
    def __init__(self, key, claim_id):
        if key is not _KEY:
            raise UntouchedLocked("tickets come only from open_attempt()")
        self.claim_id = claim_id


def _rows(name):
    if name == "math500":
        return {r["unique_id"]: {"id": r["unique_id"], "prompt": f"{r['problem']}\n{MATH_SUFFIX}", "gold": r["answer"]}
                for r in read_lines(ML / "data/math500/test.jsonl")}
    if name == "gsm8k":
        df = pd.read_parquet(ML / "data/gsm8k/main/test-00000-of-00001.parquet")
        return {f"gsm8k-test-{i:04d}": {"id": f"gsm8k-test-{i:04d}", "prompt": f"{q}\n{MATH_SUFFIX}",
                                        "gold": a.split("####")[-1].strip().replace(",", "")}
                for i, (q, a) in enumerate(zip(df["question"], df["answer"]))}
    if name == "ifeval":
        return {str(r["key"]): {"id": str(r["key"]), "prompt": r["prompt"], "gold": r}
                for r in read_lines(ML / "data/ifeval/ifeval_input_data.jsonl")}
    raise ValueError(name)


def _ids(name, part):
    return json.loads((SPLITS / manifest()["sets"][name][part]["file"]).read_text(encoding="utf-8"))


def exploring_ids(name):
    return set(_ids(name, "exploring"))


def exploring_questions(name):
    rows = _rows(name)
    return [rows[i] for i in sorted(_ids(name, "exploring"))]


def read_attempts():
    return read_lines(ATTEMPTS)


def open_attempt(claim_id, note_sha, script_sha):
    if any(a["claim"] == claim_id for a in read_attempts()):
        raise UntouchedLocked(f"{claim_id} already has an attempt on record. A new try needs a new claim id.")
    append_line(ATTEMPTS, {"event": "opened", "claim": claim_id, "note_sha256": note_sha, "script_sha256": script_sha,
                           "time": utcnow()})
    return Ticket(_KEY, claim_id)


def close_attempt(ticket, result):
    append_line(ATTEMPTS, {"event": "finished", "claim": ticket.claim_id, "time": utcnow(), "result": result})


def untouched_questions(name, ticket):
    if not isinstance(ticket, Ticket):
        raise UntouchedLocked("untouched questions need a ticket from open_attempt()")
    if not any(a["event"] == "opened" and a["claim"] == ticket.claim_id for a in read_attempts()):
        raise UntouchedLocked(f"no attempt on record for {ticket.claim_id}")
    rows = _rows(name)
    return [rows[i] for i in sorted(_ids(name, "untouched"))]
