"""Build a stand-in study folder from real exploring questions only, so confirm can run end to end on the GPU without
the real untouched halves ever being read. Each set's exploring IDs are split again into a pretend exploring half and a
pretend untouched half. Only *_exploring.json files are opened here.

Run on the tower: ~/ml/eval-venv/bin/python ~/ml/harness/tests/make_standin.py ~/ml/standin
"""
import hashlib
import json
import sys
from pathlib import Path

REAL = Path.home() / "ml"


def main():
    root = Path(sys.argv[1])
    if root.exists():
        raise SystemExit(f"{root} already exists")
    (root / "splits").mkdir(parents=True)
    (root / "models").symlink_to(REAL / "models")
    (root / "data").symlink_to(REAL / "data")
    manifest = json.loads((REAL / "MANIFEST.json").read_text())
    sets = {}
    for name, entry in manifest["sets"].items():
        ids = json.loads((REAL / "splits" / entry["exploring"]["file"]).read_text())
        halves = {"exploring": ids[0::2], "untouched": ids[1::2]}
        sets[name] = {}
        for part, half in halves.items():
            path = root / "splits" / f"{name}_{part}.json"
            path.write_text(json.dumps(sorted(half), indent=0) + "\n")
            sets[name][part] = {"count": len(half), "file": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / "MANIFEST.json").write_text(json.dumps({"seed": manifest["seed"], "model": manifest["model"], "sets": sets}))
    print("stand-in ready at", root, {k: v["exploring"]["count"] + v["untouched"]["count"] for k, v in sets.items()})


if __name__ == "__main__":
    main()
