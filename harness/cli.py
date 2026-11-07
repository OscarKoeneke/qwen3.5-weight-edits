"""The harness commands. Run on the tower: ~/ml/eval-venv/bin/python ~/ml/harness/cli.py <command>

  explore --edit edits/E-014.py [--sets math500,gsm8k]   score an edit on exploring questions, logged
  noise                                                  the unchanged model, 5 repeats on the exploring halves
  grid [--set math500]                                   the settings grid on exploring questions
  claimcheck claims/C-01.md                              every check on a note, without opening anything
  confirm claims/C-01.md                                 records the note, then opens the untouched halves once
  status                                                 what has run

--smoke QUESTIONS,MAX_TOKENS runs a small labelled test that never counts as evidence in the real study folder.
"""
import argparse
import json

import core
from common import SETS


def smoke_arg(text):
    questions, max_tokens = text.split(",")
    return core.Smoke(int(questions), int(max_tokens))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("explore")
    p.add_argument("--edit", required=True)
    p.add_argument("--sets", default="math500")
    p.add_argument("--smoke", type=smoke_arg)
    p.add_argument("--keep-weights", action="store_true")
    p = sub.add_parser("noise")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--smoke", type=smoke_arg)
    p.add_argument("--resume")
    p = sub.add_parser("grid")
    p.add_argument("--set", default="math500")
    p.add_argument("--smoke", type=smoke_arg)
    p.add_argument("--resume")
    p = sub.add_parser("claimcheck")
    p.add_argument("claim")
    p = sub.add_parser("confirm")
    p.add_argument("claim")
    p.add_argument("--smoke", type=smoke_arg)
    sub.add_parser("status")
    a = parser.parse_args()

    if a.command == "explore":
        out = core.explore(a.edit, a.sets.split(","), a.smoke, keep_weights=a.keep_weights)
    elif a.command == "noise":
        out = core.noise(a.repeats, a.sets.split(","), a.smoke, a.resume)
    elif a.command == "grid":
        out = core.grid(a.set, a.smoke, resume=a.resume)
    elif a.command == "claimcheck":
        out = core.claimcheck(a.claim)
    elif a.command == "confirm":
        out = core.confirm(a.claim, a.smoke)
    else:
        out = core.status()
    print(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
