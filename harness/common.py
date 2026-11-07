"""Paths, frozen settings and small helpers for the harness (SPEC.md sections 3 to 6).

HARNESS_ROOT points the harness at a stand-in study folder, for tests only. The real study is ~/ml.
"""
import functools
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ML = Path(os.environ.get("HARNESS_ROOT") or Path.home() / "ml")
REAL_ROOT = "HARNESS_ROOT" not in os.environ
RUNS = ML / "runs"
SPLITS = ML / "splits"
MODEL_DIR = ML / "models" / "Qwen3.5-0.8B"
HARNESS = Path(__file__).resolve().parent

SEED = 20261103
SETS = ("math500", "gsm8k", "ifeval")
SAMPLES = 4
SETTINGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "max_tokens": 32768}
GRID = [(p, r) for p in (1.5, 2.0) for r in (1.0, 1.05, 1.1)]
MAX_PROMPT_TOKENS = 2048
MATH_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
SEED_SCHEME = "first 4 bytes of sha256('id|sample|offset'), masked to 31 bits"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(qid, sample, offset):
    digest = hashlib.sha256(f"{qid}|{sample}|{offset}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_line(path, record):
    """Append one JSON line and force it to disk, so a crash can't lose it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def harness_hashes():
    return {p.name: sha256(p) for p in sorted(HARNESS.glob("*.py"))}


@functools.cache
def manifest():
    """Load the manifest and refuse to go on if a split file changed or the halves overlap."""
    m = json.loads((ML / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in SETS:
        halves = []
        for part in ("exploring", "untouched"):
            entry = m["sets"][name][part]
            path = SPLITS / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise SystemExit(f"{path.name} does not match MANIFEST.json. Stopping.")
            halves.append(set(json.loads(path.read_text(encoding="utf-8"))))
        if halves[0] & halves[1]:
            raise SystemExit(f"{name}: exploring and untouched overlap. Stopping.")
    return m
