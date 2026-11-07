"""A made-up study folder for the unit tests. No real question is read. Import this before any harness module."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="harness-test-"))
os.environ["HARNESS_ROOT"] = str(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import common  # noqa: E402

N = 8
WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
MLP = "model.language_model.layers.2.mlp.down_proj.weight"


def build():
    (ROOT / "data/math500").mkdir(parents=True)
    (ROOT / "data/gsm8k/main").mkdir(parents=True)
    (ROOT / "data/ifeval").mkdir(parents=True)
    (ROOT / "splits").mkdir()
    with open(ROOT / "data/math500/test.jsonl", "w") as f:
        for i in range(N):
            f.write(json.dumps({"unique_id": f"m{i}", "problem": f"What is {i} plus 1?", "answer": str(i + 1)}) + "\n")
    pd.DataFrame({"question": [f"How many is {i} and 2?" for i in range(N)],
                  "answer": [f"Work.\n#### {i + 2},000" for i in range(N)]}).to_parquet(
        ROOT / "data/gsm8k/main/test-00000-of-00001.parquet")
    with open(ROOT / "data/ifeval/ifeval_input_data.jsonl", "w") as f:
        for i in range(N):
            f.write(json.dumps({"key": 1000 + i, "prompt": f"Say {i}.", "instruction_id_list": [], "kwargs": []}) + "\n")
    ids = {"math500": [f"m{i}" for i in range(N)], "gsm8k": [f"gsm8k-test-{i:04d}" for i in range(N)],
           "ifeval": [str(1000 + i) for i in range(N)]}
    sets = {}
    for name, lst in ids.items():
        entry = {}
        for part, half in (("exploring", lst[:N // 2]), ("untouched", lst[N // 2:])):
            path = ROOT / "splits" / f"{name}_{part}.json"
            path.write_text(json.dumps(sorted(half), indent=0) + "\n")
            entry[part] = {"count": len(half), "file": path.name, "sha256": common.sha256(path)}
        sets[name] = entry
    model_dir = ROOT / "models/Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    torch.manual_seed(0)
    save_file({MLP: (torch.randn(64, 64) * 0.02).to(torch.bfloat16)}, model_dir / WEIGHTS_FILE)
    manifest = {"seed": common.SEED, "sets": sets, "model": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "abc123",
                                                             "files": {WEIGHTS_FILE: common.sha256(model_dir / WEIGHTS_FILE)}}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest))


build()
UNTOUCHED = {"math500": {f"m{i}" for i in range(N // 2, N)}, "gsm8k": {f"gsm8k-test-{i:04d}" for i in range(N // 2, N)},
             "ifeval": {str(1000 + i) for i in range(N // 2, N)}}
