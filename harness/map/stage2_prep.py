"""Loop map, stage 2 prep (LOOP_MAP.md): exact token positions for each pair, on the CPU. Stage 1's start is a window edge, so it runs early.

A token counts as repeated if the 8 tokens ending at it occurred in the 512 tokens before. The loop is the stretch of repeated tokens
that runs to the end of the answer. For each pair the finished answer is the longest one for that question, so it reaches as far as it can.
"""
import collections
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer

from common import MODEL_DIR
from questions import exploring_questions
from worker import render

RUN = Path.home() / "ml" / "runs" / "noise_20261106-013258-796095_original"
OUT = Path.home() / "ml" / "map"
GRAM, REACH, WINDOW, RATE, EDGE = 8, 512, 128, 0.7, 512


def repeated(ids):
    seen, flags, gaps = {}, [0] * len(ids), [0] * len(ids)
    for t in range(GRAM - 1, len(ids)):
        gram = tuple(ids[t - GRAM + 1:t + 1])
        last = seen.get(gram)
        if last is not None and t - last <= REACH:
            flags[t], gaps[t] = 1, t - last
        seen[gram] = t
    return flags, gaps


def loop_start(ids):
    """Returns (start, period), or None if the answer doesn't end in a repeating stretch."""
    flags, gaps = repeated(ids)
    total = sum(flags[-WINDOW:])
    if total < RATE * WINDOW:
        return None
    t, misses = len(ids) - 1, 0
    while t >= WINDOW:
        total += flags[t - WINDOW] - flags[t]
        t -= 1
        if total < RATE * WINDOW:
            misses += 1
            if misses > 64:
                break
        else:
            misses = 0
    start = t
    while start < len(ids) - 32 and sum(flags[start:start + 32]) < 0.8 * 32:
        start += 1
    tail = [g for g in gaps[start:] if g]
    return start, int(statistics.median(tail)) if tail else None


def had_gold(gold, text):
    return re.search(rf"(?<![\d.]){re.escape(str(gold).strip())}(?![\d.])", text) is not None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    pairs = json.loads((OUT / "pairs.json").read_text())
    rows, questions = {}, {}
    for name, cell in (("math500", "math500-o0"), ("gsm8k", "gsm8k-o0")):
        questions[name] = {q["id"]: q for q in exploring_questions(name)}
        by_id = collections.defaultdict(dict)
        for line in gzip.open(RUN / f"outputs-{cell}.jsonl.gz", "rt"):
            r = json.loads(line)
            by_id[r["id"]][r["sample"]] = r
        rows[name] = by_id
    kept, dropped, gold_seen = [], collections.Counter(), 0
    for p in pairs:
        q = questions[p["set"]][p["id"]]
        samples = rows[p["set"]][p["id"]]
        text = samples[p["loop"]["sample"]]["text"]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        found = loop_start(ids)
        if found is None or found[1] is None:
            dropped["no repeating end"] += 1
            continue
        start, period = found
        done = max((r for r in samples.values() if r["finish_reason"] == "stop"), key=lambda r: r["tokens"])
        done_ids = tok(done["text"], add_special_tokens=False)["input_ids"]
        if start < EDGE:
            dropped["loop starts too early"] += 1
            continue
        if len(done_ids) < start:
            dropped["finished answer too short"] += 1
            continue
        before = tok.decode(ids[max(0, start - 1000):start])
        gold_seen += "gold" in q and p["set"] != "ifeval" and had_gold(q["gold"], before)
        prompt = tok(render(tok, q["prompt"]), add_special_tokens=False)["input_ids"]
        kept.append({"set": p["set"], "id": p["id"], "held_back": p["held_back"], "prompt_tokens": len(prompt), "start": start, "period": period,
                     "loop_sample": p["loop"]["sample"], "loop_tokens": len(ids), "done_sample": done["sample"], "done_tokens": len(done_ids)})
    starts = sorted(k["start"] for k in kept)
    print("kept", len(kept), "dropped", dict(dropped))
    print("held back", sum(k["held_back"] for k in kept), "by set", collections.Counter(k["set"] for k in kept))
    print("start tokens p10", starts[len(starts) // 10], "median", statistics.median(starts), "p90", starts[9 * len(starts) // 10])
    print("period median", statistics.median(k["period"] for k in kept))
    print("right value in the last 1,000 tokens before the loop", gold_seen, "of", len(kept))
    print("tokens to read", sum(2 * k["start"] + 1024 for k in kept))
    (OUT / "pairs_tokens.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    main()
