"""Download the model and the three test sets, record their revisions and hashes, and split each set once.

Run on the tower: ~/ml/eval-venv/bin/python -I ~/ml/harness/fetch_and_split.py
Writes ~/ml/models/, ~/ml/data/, ~/ml/splits/ and ~/ml/MANIFEST.json. Prints counts and hashes only, never
question text. Refuses to overwrite an existing split, so the split is made exactly once (SPEC.md section 3).
"""
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

ML = Path.home() / "ml"
SEED = 20261103
MODEL = "Qwen/Qwen3.5-0.8B"
SETS = {
    "math500": ("HuggingFaceH4/MATH-500", ["test.jsonl"]),
    "gsm8k": ("openai/gsm8k", ["main/test-*.parquet"]),
    "ifeval": ("google/IFEval", ["*.jsonl"]),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ids(name, files):
    if name == "math500":
        rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
        return [r["unique_id"] for r in rows]
    if name == "gsm8k":
        df = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
        return [f"gsm8k-test-{i:04d}" for i in range(len(df))]
    if name == "ifeval":
        rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
        return [str(r["key"]) for r in rows]
    raise ValueError(name)


def main():
    api = HfApi()
    splits_dir = ML / "splits"
    if splits_dir.exists() and any(splits_dir.iterdir()):
        raise SystemExit("splits/ already exists; the split is made once. Refusing to overwrite.")
    manifest = {"written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seed": SEED,
                "model": {}, "sets": {}}

    rev = api.model_info(MODEL).sha
    mdir = snapshot_download(MODEL, revision=rev, local_dir=ML / "models" / "Qwen3.5-0.8B")
    manifest["model"] = {"repo": MODEL, "revision": rev,
                         "files": {p.name: sha256(p) for p in sorted(Path(mdir).glob("*.safetensors"))}}
    print("model", MODEL, rev, len(manifest["model"]["files"]), "weight files")

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, (repo, patterns) in SETS.items():
        drev = api.dataset_info(repo).sha
        ddir = Path(snapshot_download(repo, repo_type="dataset", revision=drev, allow_patterns=patterns,
                                      local_dir=ML / "data" / name))
        files = sorted(p for pat in patterns for p in ddir.glob(pat))
        ids = load_ids(name, files)
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        order = sorted(ids)
        random.Random(SEED).shuffle(order)
        k = math.ceil(len(order) / 2)
        parts = {"exploring": sorted(order[:k]), "untouched": sorted(order[k:])}
        entry = {"repo": repo, "revision": drev, "items": len(ids),
                 "files": {str(p.relative_to(ddir)): sha256(p) for p in files}}
        for part, lst in parts.items():
            out = splits_dir / f"{name}_{part}.json"
            out.write_text(json.dumps(lst, indent=0) + "\n", encoding="utf-8")
            entry[part] = {"count": len(lst), "file": out.name, "sha256": sha256(out)}
        manifest["sets"][name] = entry
        print(name, repo, drev, len(ids), "items:", entry["exploring"]["count"], "exploring,",
              entry["untouched"]["count"], "untouched")

    (ML / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("manifest", sha256(ML / "MANIFEST.json"))


if __name__ == "__main__":
    main()
