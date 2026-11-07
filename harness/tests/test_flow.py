"""The whole flow with a fake GPU: what reaches the generator, what gets logged, and what a claim has to get past."""
import json
import shutil
import textwrap
import unittest
from unittest import mock

import fixture
import claims
import common
import core
import grade
import questions as q
import worker

GPU = {"name": "Test GPU", "uuid": "GPU-1", "driver": "1"}
MLP = fixture.MLP


class FakeGPU:
    """Answers every request. good(label, cell) decides whether a model gets its answers right."""

    def __init__(self, good=lambda label, cell: False, on_call=None):
        self.good, self.on_call, self.calls = good, on_call, []

    def __call__(self, weights, cells, run_dir):
        if self.on_call:
            self.on_call(weights, cells)
        self.calls.append({"label": weights.label, "dir": weights.dir, "cells": cells})
        for cell in cells:
            ok = self.good(weights.label, cell["name"])
            rows = []
            for r in cell["requests"]:
                gold = cell["gold"][r["id"]]
                answer = gold if isinstance(gold, str) else "x"
                text = f"hm</think>\\boxed{{{answer if ok else 999}}}" + (" GOOD" if ok else "")
                rows.append({"id": r["id"], "sample": r["sample"], "seed": r["seed"], "text": text,
                             "finish_reason": "stop", "tokens": 7, "prompt_tokens": 3})
            worker.write_outputs(run_dir / f"outputs-{cell['name']}.jsonl.gz", rows)

    def ids(self):
        return {r["id"] for c in self.calls for cell in c["cells"] for r in cell["requests"]}


class Base(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(common.RUNS, ignore_errors=True)
        common.RUNS.mkdir()
        (common.ML / "edits").mkdir(exist_ok=True)
        (common.ML / "claims").mkdir(exist_ok=True)
        for patch in (mock.patch.object(core, "gpu_info", return_value=GPU),
                      mock.patch.object(core, "versions", return_value={}),
                      mock.patch.object(core, "GRADE_WORKERS", 0),
                      mock.patch.object(grade, "require_punkt"),
                      mock.patch.object(grade, "grade_ifeval", side_effect=lambda doc, a: "GOOD" in a)):
            patch.start()
            self.addCleanup(patch.stop)

    def script(self, name="E-001.py", delta=0.5):
        path = common.ML / "edits" / name
        path.write_text(textwrap.dedent(f"""
            def edit(t):
                w = t["{MLP}"]
                w[0, :8] += {delta}
                return {{"{MLP}": w}}
        """))
        return path

    def note(self, cid="C-01", script=None, **over):
        script = script or self.script()
        fields = {"claim id": cid, "model": "Qwen/Qwen3.5-0.8B @ abc123",
                  "change": f"{script.relative_to(common.ML)} sha256:{common.sha256(script)}",
                  "claim": "It loops less.", "primary": "math500 pass", "success": claims.SUCCESS,
                  "written": "2026-11-20T12:00:00Z", **over}
        path = common.ML / "claims" / f"{cid}.md"
        path.write_text("".join(f"{k}  {v}\n" for k, v in fields.items()))
        return path

    def ready(self):
        """A baseline and a grid, as the noise run and settings grid would leave them."""
        core.noise(repeats=2, generate=FakeGPU())
        core.grid(generate=FakeGPU())


class Exploring(Base):
    def test_only_exploring_ids_reach_the_gpu(self):
        gpu = FakeGPU()
        core.explore(self.script(), list(common.SETS), generate=gpu)
        core.noise(repeats=2, generate=gpu)
        core.grid(generate=gpu)
        self.assertTrue(gpu.ids())
        for name in common.SETS:
            self.assertFalse(gpu.ids() & fixture.UNTOUCHED[name])

    def test_a_cell_refuses_an_untouched_question_without_a_ticket(self):
        with self.assertRaises(q.UntouchedLocked):
            core.make_cell("x", "math500", [{"id": "m7", "prompt": "p", "gold": "1"}], 0, dict(common.SETTINGS))

    def test_every_pass_is_logged_with_its_edit_hash(self):
        script = self.script()
        core.explore(script, generate=FakeGPU())
        core.explore(script, generate=FakeGPU())
        passes = common.read_lines(core.PASSES)
        self.assertEqual([p["kind"] for p in passes], ["explore", "explore"])
        self.assertEqual(passes[0]["script_sha256"], common.sha256(script))

    def test_the_run_record_is_complete_and_holds_no_question_text(self):
        out = core.explore(self.script(), generate=FakeGPU())
        run = common.RUNS / out["run"]
        config = json.loads((run / "config.json").read_text())
        for key in ("weights", "cells", "splits", "seed_scheme", "env", "gpu", "versions", "harness"):
            self.assertIn(key, config)
        self.assertEqual(config["env"]["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertIn("core.py", config["harness"])
        self.assertEqual(config["cells"][0]["settings"], common.SETTINGS)
        for name in ("config.json", "summary.json", "scores-math500.json"):
            self.assertNotIn("What is", (run / name).read_text())
        self.assertNotIn("What is", json.dumps(out))

    def test_edited_weights_are_cleaned_up(self):
        gpu = FakeGPU()
        core.explore(self.script(), generate=gpu)
        self.assertFalse(gpu.calls[0]["dir"].exists())

    def test_a_score_against_baseline_uses_the_same_seeds(self):
        core.noise(repeats=2, generate=FakeGPU())
        out = core.explore(self.script(), generate=FakeGPU(lambda label, cell: True))
        self.assertEqual(out["sets"]["math500"]["vs_original"]["mean"], 1.0)

    def test_a_baseline_from_another_card_is_refused(self):
        core.noise(repeats=2, generate=FakeGPU())
        with mock.patch.object(core, "gpu_info", return_value=dict(GPU, uuid="GPU-2")):
            out = core.explore(self.script(), generate=FakeGPU())
        self.assertIn("different GPU", out["sets"]["math500"]["vs_original"])


class Noise(Base):
    def test_repeats_use_different_seeds_and_the_report_says_so(self):
        gpu = FakeGPU(lambda label, cell: cell.endswith("o1"))
        report = core.noise(repeats=2, sets=["math500"], generate=gpu)
        seeds = [[r["seed"] for r in c["requests"]] for c in gpu.calls[0]["cells"]]
        self.assertFalse(set(seeds[0]) & set(seeds[1]))
        self.assertEqual(report["math500"]["pass_by_repeat"], [0.0, 1.0])
        self.assertEqual(report["math500"]["null_pairs_clearing_zero"], 1)

    def test_one_repeat_gives_a_baseline_and_its_folder_resumes_to_more(self):
        first = core.noise(repeats=1, sets=["math500"], generate=FakeGPU())
        self.assertIsNone(first["math500"]["pass_sd"])
        run = core.read_json(core.BASELINE)["run"]
        second = core.noise(repeats=3, sets=["math500"], resume=common.RUNS / run, generate=FakeGPU())
        self.assertEqual(len(second["math500"]["pass_by_repeat"]), 3)
        self.assertEqual(core.read_json(core.BASELINE)["run"], run)

    def test_a_smoke_run_never_becomes_the_real_baseline(self):
        with mock.patch.object(core, "REAL_ROOT", True):
            core.noise(repeats=2, smoke=core.Smoke(1, 64), generate=FakeGPU())
        self.assertFalse(core.BASELINE.exists())

    def test_the_grid_runs_five_points_and_reuses_the_default(self):
        core.noise(repeats=2, generate=FakeGPU())
        gpu = FakeGPU(lambda label, cell: "p2.0-r1.1" in cell)
        report = core.grid(generate=gpu)
        self.assertEqual(len(gpu.calls[0]["cells"]), 5)
        self.assertEqual(len(report["points"]), 6)
        self.assertEqual(report["best"]["pass"], {"presence_penalty": 2.0, "repetition_penalty": 1.1})

    def test_the_grid_resumes_in_its_own_folder(self):
        core.noise(repeats=2, generate=FakeGPU())
        run = core.grid(generate=FakeGPU())["run"]
        self.assertEqual(core.grid(resume=common.RUNS / run, generate=FakeGPU())["run"], run)

    def test_the_grid_needs_the_noise_run_first(self):
        with self.assertRaises(SystemExit):
            core.grid(generate=FakeGPU())


class ClaimNotes(Base):
    def test_a_good_note_parses(self):
        self.assertEqual(claims.parse_claim(self.note())["primary_set"], "math500")

    def test_bad_notes_are_refused(self):
        bad = ({"claim id": "one"}, {"model": "other @ abc123"}, {"primary": "mmlu pass"}, {"primary": "math500 vibes"},
               {"success": "any improvement"}, {"written": "yesterday"}, {"change": "edits/E-001.py sha256:" + "0" * 64})
        for over in bad:
            with self.assertRaises(claims.BadClaim, msg=str(over)):
                claims.parse_claim(self.note(**over))

    def test_a_missing_field_is_refused(self):
        path = self.note()
        path.write_text("".join(l + "\n" for l in path.read_text().splitlines() if not l.startswith("primary")))
        with self.assertRaises(claims.BadClaim):
            claims.parse_claim(path)

    def test_claimcheck_opens_nothing(self):
        self.ready()
        out = core.claimcheck(self.note())
        self.assertEqual(len(out["random_controls"]), 3)
        self.assertEqual(q.read_attempts(), [])


class Confirm(Base):
    def good(self, label, cell):
        return label.endswith("-edited")

    def test_the_attempt_is_on_disk_before_the_first_generate(self):
        self.ready()
        seen = []
        gpu = FakeGPU(self.good, on_call=lambda w, c: seen.append(len(q.read_attempts())))
        note = self.note()
        core.confirm(note, generate=gpu)
        self.assertTrue(seen)
        self.assertGreaterEqual(min(seen), 1)
        self.assertEqual(q.read_attempts()[0]["note_sha256"], common.sha256(note))

    def test_refused_claims_leave_no_attempt_and_touch_no_gpu(self):
        gpu = FakeGPU()
        with self.assertRaises(claims.BadClaim):
            core.confirm(self.note(), generate=gpu)
        core.noise(repeats=2, generate=FakeGPU())
        with self.assertRaises(claims.BadClaim):
            core.confirm(self.note(), generate=gpu)
        self.assertEqual(q.read_attempts(), [])
        self.assertEqual(gpu.calls, [])

    def test_a_changed_script_is_refused_before_opening(self):
        self.ready()
        script = self.script()
        note = self.note(script=script)
        script.write_text(script.read_text() + "\n# changed\n")
        with self.assertRaises(claims.BadClaim):
            core.confirm(note, generate=FakeGPU())
        self.assertEqual(q.read_attempts(), [])

    def test_a_noise_run_from_another_card_is_refused(self):
        self.ready()
        with mock.patch.object(core, "gpu_info", return_value=dict(GPU, uuid="GPU-2")):
            with self.assertRaises(claims.BadClaim):
                core.confirm(self.note(), generate=FakeGPU())

    def test_a_short_noise_run_is_refused_in_the_real_folder(self):
        self.ready()
        gpu = FakeGPU()
        with mock.patch.object(core, "REAL_ROOT", True):
            with self.assertRaises(claims.BadClaim):
                core.confirm(self.note(), generate=gpu)
        self.assertEqual(q.read_attempts(), [])
        self.assertEqual(gpu.calls, [])

    def test_all_five_repeats_clear_the_real_folder(self):
        core.noise(repeats=5, generate=FakeGPU())
        core.grid(generate=FakeGPU())
        with mock.patch.object(core, "REAL_ROOT", True):
            self.assertIn("edit", core.claimcheck(self.note()))

    def test_it_runs_the_plan_and_only_untouched_questions(self):
        self.ready()
        gpu = FakeGPU(self.good)
        core.confirm(self.note(), generate=gpu)
        ran = {c["label"]: [cell["name"] for cell in c["cells"]] for c in gpu.calls}
        self.assertEqual(sorted(ran["original"]), ["original-gsm8k", "original-ifeval", "original-math500"])
        self.assertEqual(sorted(ran["C-01-edited"]), ["edited-gsm8k", "edited-ifeval", "edited-math500"])
        for seed in (1, 2, 3):
            self.assertEqual(ran[f"C-01-random{seed}"], [f"random{seed}-math500"])
        self.assertEqual(gpu.ids(), set().union(*fixture.UNTOUCHED.values()))

    def test_every_arm_shares_the_same_seeds(self):
        self.ready()
        gpu = FakeGPU(self.good)
        core.confirm(self.note(), generate=gpu)
        arms = [cell for c in gpu.calls for cell in c["cells"] if cell["name"].endswith("math500")]
        self.assertEqual(len({tuple(r["seed"] for r in a["requests"]) for a in arms}), 1)

    def test_a_clean_win_is_a_finding(self):
        self.ready()
        report = core.confirm(self.note(), generate=FakeGPU(self.good))
        self.assertTrue(report["finding"])
        self.assertEqual(q.read_attempts()[-1]["result"]["finding"], True)

    def test_a_random_edit_doing_as_well_is_not_a_finding(self):
        self.ready()
        gpu = FakeGPU(lambda label, cell: label.endswith("-edited") or label.endswith("-random2"))
        report = core.confirm(self.note(), generate=gpu)
        self.assertFalse(report["finding"])
        self.assertFalse(report["checks"]["beats_random"])
        self.assertEqual(report["best_random_seed"], 2)

    def test_a_setting_doing_as_well_is_not_a_finding(self):
        core.noise(repeats=2, generate=FakeGPU())
        core.grid(generate=FakeGPU(lambda label, cell: "p2.0-r1.1" in cell))
        gpu = FakeGPU(lambda label, cell: label.endswith("-edited") or cell.startswith("settings"))
        report = core.confirm(self.note(), generate=gpu)
        self.assertFalse(report["checks"]["beats_settings"])
        self.assertTrue(report["checks"]["gain"])

    def test_damage_on_another_set_stops_a_finding_and_is_named(self):
        self.ready()
        gpu = FakeGPU(lambda label, cell: (label.endswith("-edited") and cell.endswith("math500"))
                      or (label == "original" and cell.endswith("gsm8k")))
        report = core.confirm(self.note(), generate=gpu)
        self.assertFalse(report["finding"])
        self.assertEqual(report["damaged"], ["gsm8k"])

    def test_a_claim_id_cannot_run_twice(self):
        self.ready()
        core.confirm(self.note(), generate=FakeGPU(self.good))
        with self.assertRaises(q.UntouchedLocked):
            core.confirm(self.note(), generate=FakeGPU(self.good))

    def test_a_crash_stays_on_record_and_blocks_a_quiet_retry(self):
        self.ready()

        def boom(weights, cells):
            raise RuntimeError("power cut")

        with self.assertRaises(RuntimeError):
            core.confirm(self.note(), generate=FakeGPU(on_call=boom))
        self.assertEqual([a["event"] for a in q.read_attempts()], ["opened"])
        with self.assertRaises(q.UntouchedLocked):
            core.confirm(self.note(), generate=FakeGPU())

    def test_a_later_claim_reuses_the_original_untouched_runs(self):
        self.ready()
        core.confirm(self.note("C-01"), generate=FakeGPU(self.good))
        gpu = FakeGPU(self.good)
        core.confirm(self.note("C-02", self.script("E-002.py", 0.7)), generate=gpu)
        self.assertNotIn("original", [c["label"] for c in gpu.calls])

    def test_weights_are_cleaned_up(self):
        self.ready()
        gpu = FakeGPU(self.good)
        core.confirm(self.note(), generate=gpu)
        self.assertEqual([c["dir"].exists() for c in gpu.calls if c["label"] != "original"], [False] * 4)

    def test_a_smoke_confirmation_is_refused_in_the_real_folder(self):
        self.ready()
        with mock.patch.object(core, "REAL_ROOT", True):
            with self.assertRaises(SystemExit):
                core.confirm(self.note(), smoke=core.Smoke(1, 64), generate=FakeGPU())
        self.assertEqual(q.read_attempts(), [])

    def test_the_report_holds_no_question_text_or_ids(self):
        self.ready()
        report = json.dumps(core.confirm(self.note(), generate=FakeGPU(self.good)))
        self.assertNotIn("What is", report)
        self.assertNotIn('"m7"', report)


class Readiness(Base):
    def test_a_run_needing_ifeval_stops_before_the_gpu_when_its_data_is_missing(self):
        gpu = FakeGPU()
        with mock.patch.object(grade, "require_punkt", side_effect=grade.IFEvalNotReady("x")):
            with self.assertRaises(grade.IFEvalNotReady):
                core.noise(repeats=2, generate=gpu)
        self.assertEqual(gpu.calls, [])


if __name__ == "__main__":
    unittest.main()
