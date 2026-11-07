"""Wiring test on the real GPU and real model, in a stand-in folder: noise, grid, explore, claimcheck, confirm.

Run on the tower: HARNESS_ROOT=~/ml/standin ~/ml/eval-venv/bin/python ~/ml/harness/tests/wire_flow.py [ifeval]
Without the ifeval argument it covers MATH-500 and GSM8K only. A small cap means these numbers are not results.
"""
import json
import os
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not os.environ.get("HARNESS_ROOT"):
    raise SystemExit("wiring tests run only in a stand-in folder")

import core  # noqa: E402
from claims import SUCCESS  # noqa: E402
from common import ML, SETS, sha256  # noqa: E402

SMOKE = core.Smoke(8, 3072)
SETS_HERE = SETS if "ifeval" in sys.argv else ("math500", "gsm8k")
core.CONFIRM_SETS = SETS_HERE


def show(step, value, start):
    print(f"\n=== {step} ({time.time() - start:.0f}s)\n" + json.dumps(value, indent=1, default=str)[:2500], flush=True)


def main():
    t = time.time()
    show("noise", core.noise(2, SETS_HERE, SMOKE), t)
    t = time.time()
    show("grid", core.grid("math500", SMOKE), t)

    edit = ML / "edits" / "E-000-wiring.py"
    edit.parent.mkdir(exist_ok=True)
    edit.write_text(textwrap.dedent("""
        # Wiring test only: touches an MLP, a fused attention input and a norm, so the readback check sees each kind.
        L = "model.language_model.layers."
        def edit(t):
            changed = {}
            for name, scale in ((L + "10.mlp.down_proj.weight", 1.25), (L + "0.linear_attn.in_proj_qkv.weight", 1.1),
                                (L + "3.input_layernorm.weight", 1.05)):
                w = t[name]
                w[:8] = w[:8] * scale
                changed[name] = w
            return changed
    """))
    t = time.time()
    show("explore", core.explore(edit, SETS_HERE, SMOKE), t)

    note = ML / "claims" / "C-90.md"
    note.parent.mkdir(exist_ok=True)
    manifest = json.loads((ML / "MANIFEST.json").read_text())
    note.write_text(f"claim id  C-90\nmodel  {manifest['model']['repo']} @ {manifest['model']['revision']}\n"
                    f"change  edits/{edit.name} sha256:{sha256(edit)}\nclaim  Wiring test, not a real claim.\n"
                    f"primary  math500 pass\nsuccess  {SUCCESS}\nwritten  2026-11-05T00:00:00Z\n")
    t = time.time()
    show("claimcheck", core.claimcheck(note), t)
    t = time.time()
    show("confirm", core.confirm(note, SMOKE), t)
    show("status", core.status(), t)


if __name__ == "__main__":
    main()
