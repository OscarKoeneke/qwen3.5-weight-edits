"""What do per-request seeds cost, and what does running the engine in-process cost? Exploring GSM8K prompts, no grading.

Run on the tower, once per engine mode, with VLLM_ENABLE_V1_MULTIPROCESSING set to 0 or 1.
Not a test pass: nothing is scored and nothing is used as evidence.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import MAX_PROMPT_TOKENS, MODEL_DIR, SETTINGS, seed_for  # noqa: E402
from questions import exploring_questions  # noqa: E402

REQUESTS = 256
MAX_TOKENS = 1024


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(MODEL_DIR), language_model_only=True, max_model_len=SETTINGS["max_tokens"] + MAX_PROMPT_TOKENS,
              gpu_memory_utilization=0.9, seed=0, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    questions = exploring_questions("gsm8k")[:REQUESTS]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=True) for q in questions]
    base = dict(SETTINGS, max_tokens=MAX_TOKENS, ignore_eos=True)
    mode = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    for label, seeded in (("warmup", False), ("no seeds", False), ("seeded", True), ("no seeds", False), ("seeded", True)):
        params = [SamplingParams(seed=seed_for(q["id"], 0, 0) if seeded else None, **base) for q in questions]
        start = time.time()
        out = llm.generate(prompts, params, use_tqdm=False)
        wall = time.time() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in out)
        print(f"BENCH mp={mode} {label}: {tokens} tokens in {wall:.1f}s = {tokens / wall:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
