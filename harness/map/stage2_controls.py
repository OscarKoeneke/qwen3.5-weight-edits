"""Loop map, stage 2 controls (LOOP_MAP.md). Stage 2's direction scored about 90% at every layer, layer 0 included, so it may only be reading the text.

Here the same held-back test runs on plain token counts, with no model at all, and on the direction at earlier windows. It also reports how much
of each window already repeats (4 tokens seen in the 512 before), and the direction test at the windows saved with --offset.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer

from common import MODEL_DIR

RUN = Path.home() / "ml" / "runs" / "noise_20261106-013258-796095_original"
OUT = Path.home() / "ml" / "map"
SPAN = 256


def repeat_share(ids, a, b, gram=4, reach=512):
    hits = 0
    for t in range(a, b):
        piece = ids[t - gram + 1:t + 1]
        back = ids[max(0, t - reach):t]
        hits += any(back[i:i + gram] == piece for i in range(len(back) - gram + 1))
    return hits / (b - a)


def counts(ids, a, b):
    return collections.Counter(ids[a:b])


def token_test(pairs):
    """Difference of mean token counts on the fit pairs, tried on the held-back pairs."""
    total = collections.Counter()
    for p in pairs:
        if not p["held_back"]:
            total.update({k: v for k, v in p["diff"].items()})
    hits = n = 0
    for p in pairs:
        if p["held_back"]:
            n += 1
            hits += sum(total[k] * v for k, v in p["diff"].items()) > 0
    return hits, n


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    pairs = json.loads((OUT / "pairs_tokens.json").read_text())
    texts = {}
    for name, cell in (("math500", "math500-o0"), ("gsm8k", "gsm8k-o0")):
        for line in gzip.open(RUN / f"outputs-{cell}.jsonl.gz", "rt"):
            r = json.loads(line)
            texts[name, r["id"], r["sample"]] = r["text"]
    for offset in (256, 1024, 4096):
        rows, loop_rep, done_rep = [], [], []
        for p in pairs:
            if p["start"] < offset:
                continue
            loop = tok(texts[p["set"], p["id"], p["loop_sample"]], add_special_tokens=False)["input_ids"]
            done = tok(texts[p["set"], p["id"], p["done_sample"]], add_special_tokens=False)["input_ids"]
            a = p["start"] - offset
            b = a + SPAN
            diff = counts(loop, a, b)
            diff.subtract(counts(done, a, b))
            rows.append({"held_back": p["held_back"], "diff": diff})
            loop_rep.append(repeat_share(loop, a, b))
            done_rep.append(repeat_share(done, a, b))
        hits, n = token_test(rows)
        print(f"offset {offset}: pairs {len(rows)}, plain token counts pick the looper {hits} of {n} ({hits / n:.2f}), "
              f"repeating share looped {sum(loop_rep) / len(rows):.3f}, finished {sum(done_rep) / len(rows):.3f}", flush=True)
        suffix = "" if offset == SPAN else f"_{offset}"
        if (OUT / f"stage2_vectors{suffix}.pt").exists():
            meta = {r["id"]: r for r in json.loads((OUT / f"stage2_rows{suffix}.json").read_text())}
            vectors = torch.load(OUT / f"stage2_vectors{suffix}.pt")
            diffs = {k: v["loop"] - v["done"] for k, v in vectors.items()}
            fit = torch.stack([d for k, d in diffs.items() if not meta[k]["held_back"]])
            check = torch.stack([d for k, d in diffs.items() if meta[k]["held_back"]])
            shares = [float(((check[:, i] @ fit[:, i].mean(0)) > 0).float().mean()) for i in range(fit.shape[1])]
            print(f"  direction, fit {len(fit)} held back {len(check)}:", " ".join(f"{i}:{s:.2f}" for i, s in enumerate(shares)), flush=True)


if __name__ == "__main__":
    main()
