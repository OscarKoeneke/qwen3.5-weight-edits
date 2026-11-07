"""Loop map, stage 2 report (LOOP_MAP.md). Reads what stage2_read.py saved and prints one table per signal. Correlation only, no claims.

The held-back test: a direction per layer is fitted on the fit pairs (looped minus finished, just before the loop starts). On the held-back
pairs it counts how often that direction picks the answer that went on to loop. Chance is 50%, and each pair is judged on its own.
"""
import json
import math
from pathlib import Path

import torch

OUT = Path.home() / "ml" / "map"


def wilson(hits, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = hits / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def direction_test(rows, vectors):
    diffs = {r["id"]: vectors[r["id"]]["loop"] - vectors[r["id"]]["done"] for r in rows}
    fit = torch.stack([diffs[r["id"]] for r in rows if not r["held_back"]])
    check = torch.stack([diffs[r["id"]] for r in rows if r["held_back"]])
    table = []
    for layer in range(fit.shape[1]):
        d = fit[:, layer].mean(0)
        hits = int(((check[:, layer] @ d) > 0).sum())
        lo, hi = wilson(hits, len(check))
        table.append({"layer": layer, "hits": hits, "n": len(check), "share": round(hits / len(check), 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    return {"fit_pairs": len(fit), "held_back_pairs": len(check), "layers": table}


def mean_rows(rows, side, window, key):
    values = [r[side][window][key] for r in rows if window in r[side]]
    return [round(sum(v[i] for v in values) / len(values), 3) for i in range(len(values[0]))], len(values)


def main():
    rows = json.loads((OUT / "stage2_rows.json").read_text())
    vectors = torch.load(OUT / "stage2_vectors.pt")
    thinking = [r for r in rows if r["still_thinking"]]
    report = {"pairs": len(rows), "still_thinking": len(thinking), "seconds": round(sum(r["seconds"] for r in rows)),
              "direction_all": direction_test(rows, vectors), "direction_still_thinking": direction_test(thinking, vectors)}
    for key in ("logp", "state"):
        report[key] = {f"{side}_{window}": mean_rows(thinking, side, window, key) for side in ("loop", "done") for window in ("pre", "post")}
    (OUT / "stage2_report.json").write_text(json.dumps(report, indent=1))
    print("pairs", report["pairs"], "still thinking at the same spot", report["still_thinking"], "seconds", report["seconds"])
    for name in ("direction_all", "direction_still_thinking"):
        t = report[name]
        print(name, "fit", t["fit_pairs"], "held back", t["held_back_pairs"])
        print(" ".join(f"{x['layer']}:{x['share']:.2f}" for x in t["layers"]))
    for key in ("logp", "state"):
        for name, (values, n) in report[key].items():
            print(key, name, "n", n, " ".join(f"{v:.2f}" for v in values))


if __name__ == "__main__":
    main()
