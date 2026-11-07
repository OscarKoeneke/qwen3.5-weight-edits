"""Claim notes (SPEC.md section 9). A note is parsed and checked in full before anything untouched can open."""
import re
from datetime import datetime

from common import ML, SETS, manifest, sha256
from stats import METRICS

FIELDS = ("claim id", "model", "change", "claim", "primary", "success", "written")
SUCCESS = "the section 5 rule on the primary metric, beats both controls, no damage"


class BadClaim(Exception):
    pass


def parse_claim(path):
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for name in sorted(FIELDS, key=len, reverse=True):
            if line.startswith(name + " "):
                fields[name] = line[len(name):].strip()
                break
    missing = [f for f in FIELDS if not fields.get(f)]
    if missing:
        raise BadClaim(f"claim note is missing: {', '.join(missing)}")

    if not re.fullmatch(r"C-\d{2,}", fields["claim id"]):
        raise BadClaim("claim id must look like C-01")

    model = manifest()["model"]
    if fields["model"] != f"{model['repo']} @ {model['revision']}":
        raise BadClaim("model must be the manifest's repo and revision, as 'repo @ revision'")

    change = re.fullmatch(r"(\S+) sha256:([0-9a-f]{64})", fields["change"])
    if not change:
        raise BadClaim("change must be 'path sha256:<hash>'")
    script = ML / change.group(1)
    if not script.is_file() or sha256(script) != change.group(2):
        raise BadClaim("the edit script is missing or doesn't match the hash in the note")

    primary = fields["primary"].split()
    if len(primary) != 2 or primary[0] not in SETS or primary[1] not in METRICS:
        raise BadClaim(f"primary must be one of {SETS} and one of {sorted(METRICS)}")

    if fields["success"] != SUCCESS:
        raise BadClaim("success must state the one rule for all claims, word for word")

    try:
        datetime.strptime(fields["written"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise BadClaim("written must be a UTC time like 2026-11-20T12:00:00Z")

    return {"id": fields["claim id"], "script": script, "script_sha": change.group(2), "claim": fields["claim"],
            "primary_set": primary[0], "metric": primary[1], "note_sha": sha256(path)}
