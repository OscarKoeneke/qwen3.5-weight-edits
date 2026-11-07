"""One process per weights folder: load vLLM once, then answer each cell in the job, saving as each cell finishes.

Run by core.run_worker, never by hand. The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the loaded
weights can be read back directly. Before any answer is generated, the worker checks that the weights vLLM actually holds
differ from the original in exactly the expected places.
"""
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import edits
from common import MAX_PROMPT_TOKENS, SETTINGS, append_line


def param_table(model):
    import torch
    table = {}
    for name, p in model.named_parameters():
        data = p.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        table[name] = {"dtype": str(p.dtype), "shape": list(p.shape), "sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    return table


def render(tokenizer, prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
                                         enable_thinking=True)
    if not text.endswith("<think>\n"):
        raise SystemExit("thinking is not on in the rendered prompt. Stopping.")
    return text


def write_outputs(path, rows):
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def run_cell(llm, tokenizer, cell, run_dir):
    from vllm import SamplingParams
    out_path = run_dir / f"outputs-{cell['name']}.jsonl.gz"
    if out_path.exists():
        return
    prompts = [render(tokenizer, r["prompt"]) for r in cell["requests"]]
    longest = max(len(tokenizer.encode(p)) for p in prompts)
    if longest > MAX_PROMPT_TOKENS:
        raise SystemExit(f"{cell['name']}: a prompt has {longest} tokens, over the {MAX_PROMPT_TOKENS} limit. Stopping.")
    params = [SamplingParams(seed=r["seed"], skip_special_tokens=False, **cell["settings"]) for r in cell["requests"]]
    start = time.time()
    results = llm.generate(prompts, params, use_tqdm=False)
    wall = time.time() - start
    rows = [{"id": r["id"], "sample": r["sample"], "seed": r["seed"], "text": o.outputs[0].text,
             "finish_reason": o.outputs[0].finish_reason, "tokens": len(o.outputs[0].token_ids),
             "prompt_tokens": len(o.prompt_token_ids)} for r, o in zip(cell["requests"], results)]
    write_outputs(out_path, rows)
    tokens = sum(r["tokens"] for r in rows)
    append_line(run_dir / "cells.log", {"cell": cell["name"], "answers": len(rows), "wall_s": round(wall, 1),
                                        "tokens": tokens, "tokens_per_s": round(tokens / wall, 1),
                                        "longest_prompt": longest})


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    run_dir = Path(job["run_dir"])
    from vllm import LLM
    start = time.time()
    llm = LLM(model=job["model"], language_model_only=True, max_model_len=SETTINGS["max_tokens"] + MAX_PROMPT_TOKENS,
              gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
    table = llm.apply_model(param_table)[0]
    load_s = round(time.time() - start, 1)
    (run_dir / "params.json").write_text(json.dumps(table), encoding="utf-8")

    original = Path(job["original_params"])
    if job["is_original"] and not original.exists():
        original.write_text(json.dumps(table), encoding="utf-8")
    problem = edits.readback_problem(json.loads(original.read_text(encoding="utf-8")), table, job["expected_groups"])
    append_line(run_dir / "cells.log", {"load_s": load_s, "readback_ok": problem is None, "problem": problem})
    if problem:
        raise SystemExit(f"weights in vLLM don't match the edit: {problem}")

    tokenizer = llm.get_tokenizer()
    for cell in job["cells"]:
        run_cell(llm, tokenizer, cell, run_dir)


if __name__ == "__main__":
    main()
