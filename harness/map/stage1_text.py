"""Loop map, stage 1 (LOOP_MAP.md): where loops start in saved exploring answers. Text only, no GPU, reads no untouched file.

A window is 8,000 characters, stepped by 1,000. Its score is its zlib size over its plain size, so lower means more repetitive.
The threshold is the 10th percentile of window scores in finished answers of the same set. A capped answer has a clean start
if, from some window on, at least 90% of windows sit under the threshold and they carry on to the end of the text.
"""
import collections
import gzip
import hashlib
import json
import random
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grade
from questions import exploring_questions

RUN = Path.home() / "ml" / "runs" / "noise_20261106-013258-796095_original"
OUT = Path.home() / "ml" / "map"
SIZE, STEP, HOLD_SHARE, SEED = 8000, 1000, 3, 20261103
BOXED = re.compile(r"\\boxed\{")


def scores(text):
    return [(i, len(zlib.compress(text[i:i + SIZE].encode(), 6)) / max(1, len(text[i:i + SIZE].encode())))
            for i in range(0, max(1, len(text) - SIZE + 1), STEP)]


def percentile(values, share):
    values = sorted(values)
    return values[min(len(values) - 1, int(share * len(values)))]


def onset(windows, limit):
    """First window from which at least 90% of the remaining windows are under the limit, and the last one is too."""
    if not windows or windows[-1][1] >= limit:
        return None
    under = 0
    tail = []
    for _, s in reversed(windows):
        under += s < limit
        tail.append(under)
    tail.reverse()
    for k, (pos, s) in enumerate(windows):
        if s < limit and tail[k] >= 0.9 * (len(windows) - k):
            return pos
    return None


def period(text, needle_size=60):
    """Distance between the last two exact copies of the text's closing stretch. None if it has no exact earlier copy."""
    needle = text[-needle_size - 200:-200]
    first = text.rfind(needle, 0, len(text) - needle_size - 200)
    return len(text) - needle_size - 200 - first if first >= 0 else None


def boxed_values(text):
    found = []
    for m in BOXED.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth and i - m.end() < 200:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if not depth:
            found.append(text[m.end():i - 1])
    return found


def reached_answer(gold, text):
    """Whether any box before the loop already holds the right answer, by the study's own math check."""
    return any(safe_grade(gold, v) for v in boxed_values(text)[-12:])


def safe_grade(gold, value):
    try:
        return grade.grade_math(gold, f"\\boxed{{{value}}}")
    except Exception:
        return False


def is_held_back(qid):
    return int(hashlib.sha256(f"{qid}|{SEED}".encode()).hexdigest(), 16) % HOLD_SHARE == 0


def run_set(name, cell):
    rows = [json.loads(l) for l in gzip.open(RUN / f"outputs-{cell}.jsonl.gz", "rt")]
    gold = {q["id"]: q["gold"] for q in exploring_questions(name)}
    long_finished = [r for r in rows if r["finish_reason"] == "stop" and len(r["text"]) >= 2 * SIZE]
    limit = percentile([s for r in long_finished for _, s in scores(r["text"])], 0.10)
    report = {"set": name, "limit": round(limit, 3), "answers": len(rows), "capped": 0, "clean": 0, "no_exact_repeat": 0,
              "reached_answer_before_loop": 0, "reached_checked": 0}
    onsets, periods, opening = [], [], collections.Counter()
    looped = collections.defaultdict(list)
    for r in rows:
        if r["finish_reason"] != "length":
            continue
        report["capped"] += 1
        text = r["text"]
        start = onset(scores(text), limit)
        if start is None:
            continue
        report["clean"] += 1
        token_start = int(start * r["tokens"] / len(text))
        onsets.append(token_start)
        p = period(text)
        if p is None:
            report["no_exact_repeat"] += 1
        else:
            periods.append(int(p * r["tokens"] / len(text)))
        report["reached_checked"] += 1
        report["reached_answer_before_loop"] += reached_answer(gold[r["id"]], text[:start])
        opening.update(" ".join(w) for w in zip(*[text[start:start + 1500].split()[i:] for i in range(3)]))
        looped[r["id"]].append({"sample": r["sample"], "char_start": start, "token_start": token_start, "period_tokens": periods[-1] if p else None})
    onsets.sort()
    periods.sort()
    report["onset_tokens"] = {"p10": percentile(onsets, 0.1), "median": percentile(onsets, 0.5), "p90": percentile(onsets, 0.9)} if onsets else None
    report["period_tokens"] = {"p10": percentile(periods, 0.1), "median": percentile(periods, 0.5), "p90": percentile(periods, 0.9)} if periods else None
    report["common_phrases"] = opening.most_common(8)

    pairs = []
    finished = collections.defaultdict(list)
    for r in rows:
        if r["finish_reason"] == "stop":
            finished[r["id"]].append(r)
    rng = random.Random(SEED)
    for qid in sorted(set(looped) & set(finished)):
        loop = rng.choice(looped[qid])
        done = rng.choice(finished[qid])
        pairs.append({"set": name, "cell": cell, "id": qid, "held_back": is_held_back(qid), "loop": loop,
                      "finished": {"sample": done["sample"], "tokens": done["tokens"]}})
    report["paired_questions"] = len(pairs)
    report["paired_held_back"] = sum(p["held_back"] for p in pairs)
    return report, pairs


def main():
    OUT.mkdir(exist_ok=True)
    all_pairs, reports = [], []
    for name, cell in (("math500", "math500-o0"), ("gsm8k", "gsm8k-o0")):
        report, pairs = run_set(name, cell)
        reports.append(report)
        all_pairs += pairs
        print(json.dumps(report), flush=True)
    (OUT / "stage1_report.json").write_text(json.dumps(reports, indent=1))
    (OUT / "pairs.json").write_text(json.dumps(all_pairs, indent=1))
    print("pairs", len(all_pairs))


if __name__ == "__main__":
    main()
