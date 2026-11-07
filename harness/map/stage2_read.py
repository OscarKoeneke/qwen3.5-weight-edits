"""Loop map, stage 2 (LOOP_MAP.md): read saved answers through the model, no generation. Each pair is a looped and a finished answer to one question.

Per layer, at matching token positions, it keeps:
- the mean residual just before the loop starts, so a direction fitted on some pairs can be tried on the held-back ones;
- the log-probability of the actual next token under the logit lens, to see which layer already knows what comes next;
- how much each Gated DeltaNet layer's running state changes per segment.

Usage: stage2_read.py --check (short CPU run of every code path), --one (time one pair), or a full run with --budget <seconds>.
"""
import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import core
from common import MODEL_DIR
from questions import exploring_questions
from worker import render

RUN = Path.home() / "ml" / "runs" / "noise_20261106-013258-796095_original"
OUT = Path.home() / "ml" / "map"
SPAN, GAP, SEGMENT, SEED = 256, 128, 256, 20261103


def text_model(model):
    base = model.model
    return getattr(base, "language_model", base)


def forward_in_segments(text, ids, on_segment, stop):
    cache = None
    for lo in range(0, stop, SEGMENT):
        hi = min(lo + SEGMENT, stop)
        out = text(input_ids=ids[None, lo:hi], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        on_segment(lo, hi, cache)
    return out


@torch.inference_mode()
def read(model, ids, windows):
    """windows maps a name to (lo, hi) token positions. Returns per-window means, next-token log-probs and state changes."""
    text = text_model(model)
    layers = list(text.layers)
    latest = [None] * len(layers)
    hooks = [layer.register_forward_hook(lambda _, __, out, i=i: latest.__setitem__(i, out[0] if isinstance(out, tuple) else out))
             for i, layer in enumerate(layers)]
    linear = [i for i, layer in enumerate(layers) if hasattr(layer, "linear_attn")]
    kept = {name: [[] for _ in layers] for name in windows}
    previous, changes = {}, []

    def on_segment(lo, hi, cache):
        for name, (a, b) in windows.items():
            x, y = max(a, lo), min(b, hi)
            if x < y:
                for i in range(len(layers)):
                    kept[name][i].append(latest[i][0, x - lo:y - lo])
        row = []
        for i in linear:
            state = cache.layers[i].recurrent_states[0].float()
            row.append(float((state - previous[i]).norm() / previous[i].norm()) if i in previous else float("nan"))
            previous[i] = state.clone()
        changes.append((hi, row))

    try:
        forward_in_segments(text, ids, on_segment, max(b for _, b in windows.values()) + 1)
    finally:
        for h in hooks:
            h.remove()
    result = {}
    for name, (a, b) in windows.items():
        means, logp = [], []
        target = ids[a + 1:b + 1]
        for i in range(len(layers)):
            h = torch.cat(kept[name][i])
            means.append(h.float().mean(0).cpu())
            probs = torch.log_softmax(model.lm_head(text.norm(h)).float(), -1)
            logp.append(float(probs[torch.arange(len(target), device=target.device), target].mean()))
        rows = [r for end, r in changes if a < end <= b + SEGMENT]
        state = [sum(r[k] for r in rows) / len(rows) for k in range(len(linear))] if rows else []
        result[name] = {"mean": torch.stack(means), "logp": logp, "state": state}
    return result


def answers():
    found = {}
    for name, cell in (("math500", "math500-o0"), ("gsm8k", "gsm8k-o0")):
        for line in gzip.open(RUN / f"outputs-{cell}.jsonl.gz", "rt"):
            r = json.loads(line)
            found[name, r["id"], r["sample"]] = r["text"]
    return found


def tokens(tok, prompt, text, device):
    head = tok(render(tok, prompt), add_special_tokens=False)["input_ids"]
    return torch.tensor(head + tok(text, add_special_tokens=False)["input_ids"], device=device), len(head)


def check(model, tok, pair, texts, questions):
    ids, _ = tokens(tok, questions[pair["set"]][pair["id"]]["prompt"], texts[pair["set"], pair["id"], pair["loop_sample"]], "cpu")
    ids = ids[:640]
    text = text_model(model)
    with torch.inference_mode():
        whole = text(input_ids=ids[None]).last_hidden_state[0, -1].float()
        parts = forward_in_segments(text, ids, lambda *a: None, len(ids)).last_hidden_state[0, -1].float()
    print("whole against segments, relative difference", float((whole - parts).norm() / whole.norm()))
    out = read(model, ids, {"pre": (128, 256), "post": (384, 512)})
    print("mean", tuple(out["pre"]["mean"].shape), "logp last layer", out["post"]["logp"][-1], "state values", len(out["post"]["state"]), out["post"]["state"][:3])


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--check", action="store_true")
    args.add_argument("--one", action="store_true")
    args.add_argument("--budget", type=float, default=3300)
    args.add_argument("--offset", type=int, default=SPAN, help="how far before the loop's start the first window begins")
    a = args.parse_args()
    name = "" if a.offset == SPAN else f"_{a.offset}"
    device = "cpu" if a.check else "cuda"
    if a.check:
        torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    pairs = json.loads((OUT / "pairs_tokens.json").read_text())
    random.Random(SEED).shuffle(pairs)
    texts = answers()
    questions = {s: {q["id"]: q for q in exploring_questions(s)} for s in ("math500", "gsm8k")}
    if a.check:
        check(model, tok, pairs[0], texts, questions)
        return
    think_end = tok.convert_tokens_to_ids("</think>")
    began, rows, vectors = time.time(), [], {}
    with core.gpu_lock():
        for n, p in enumerate(pairs[:1] if a.one else pairs):
            if time.time() - began > a.budget:
                print("budget reached after", n, "pairs", flush=True)
                break
            t = time.time()
            prompt = questions[p["set"]][p["id"]]["prompt"]
            loop, head = tokens(tok, prompt, texts[p["set"], p["id"], p["loop_sample"]], device)
            done, _ = tokens(tok, prompt, texts[p["set"], p["id"], p["done_sample"]], device)
            s = head + p["start"]
            if p["start"] < a.offset:
                continue
            pre, post = (s - a.offset, s - a.offset + SPAN), (s + GAP, s + GAP + SPAN)
            looped = read(model, loop, {"pre": pre, "post": post} if a.offset == SPAN else {"pre": pre})
            spots = {"pre": pre}
            if a.offset == SPAN and len(done) > post[1] + 1:
                spots["post"] = post
            finished = read(model, done, spots)
            ends = (done == think_end).nonzero()
            still_thinking = bool(len(ends)) and int(ends[-1]) > post[1]
            vectors[p["id"]] = {"loop": looped["pre"]["mean"], "done": finished["pre"]["mean"]}
            rows.append({**p, "still_thinking": still_thinking, "seconds": round(time.time() - t, 1),
                         "loop": {k: {"logp": v["logp"], "state": v["state"]} for k, v in looped.items()},
                         "done": {k: {"logp": v["logp"], "state": v["state"]} for k, v in finished.items()}})
            print(n + 1, p["set"], "tokens", s + GAP + SPAN, "seconds", rows[-1]["seconds"], flush=True)
            if not a.one and n % 10 == 9:
                (OUT / f"stage2_rows{name}.json").write_text(json.dumps(rows))
                torch.save(vectors, OUT / f"stage2_vectors{name}.pt")
    if not a.one:
        (OUT / f"stage2_rows{name}.json").write_text(json.dumps(rows))
        torch.save(vectors, OUT / f"stage2_vectors{name}.pt")
    print("pairs read", len(rows), "seconds", round(time.time() - began), "peak GB", round(torch.cuda.max_memory_allocated() / 2**30, 1))


if __name__ == "__main__":
    main()
