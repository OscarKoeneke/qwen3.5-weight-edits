"""Edits: validation, the rounding refusal, random controls, and the readback check."""
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

import fixture  # noqa: F401
import edits

MLP = "model.language_model.layers.2.mlp.down_proj.weight"
NORM = "model.language_model.layers.0.linear_attn.norm.weight"
ALOG = "model.language_model.layers.0.linear_attn.A_log"


def original():
    torch.manual_seed(0)
    return {MLP: (torch.randn(64, 64) * 0.02).to(torch.bfloat16), NORM: torch.ones(128), ALOG: torch.randn(16),
            "model.visual.blocks.0.attn.proj.weight": torch.randn(8, 8).to(torch.bfloat16),
            "mtp.fc.weight": torch.randn(8, 8).to(torch.bfloat16)}


class Footprint(unittest.TestCase):
    def test_counts_and_norm(self):
        o = original()
        new = o[MLP].clone()
        new[0, :4] += 0.5
        fp = edits.footprint(o, {MLP: new})[MLP]
        self.assertEqual(fp["changed"], 4)
        self.assertAlmostEqual(fp["l2"], 1.0, delta=0.02)

    def test_an_empty_edit_is_refused(self):
        with self.assertRaises(edits.BadEdit):
            edits.footprint(original(), {})

    def test_shape_dtype_nan_and_unknown_names_are_refused(self):
        o = original()
        bad = [{MLP: o[MLP][:2]}, {MLP: o[MLP].float()}, {"model.language_model.nope": o[MLP]},
               {MLP: o[MLP] * float("nan")}]
        for changed in bad:
            with self.assertRaises(edits.BadEdit):
                edits.footprint(o, changed)

    def test_tensors_vllm_never_loads_are_refused(self):
        o = original()
        for name in ("model.visual.blocks.0.attn.proj.weight", "mtp.fc.weight"):
            with self.assertRaises(edits.BadEdit):
                edits.footprint(o, {name: o[name] + 1})

    def test_an_edit_that_rounds_away_is_refused(self):
        o = original()
        self.assertEqual(edits.held_dtype(NORM), torch.bfloat16)
        with self.assertRaisesRegex(edits.BadEdit, "rounds away"):
            edits.footprint(o, {NORM: o[NORM] + 1e-5})

    def test_a_log_keeps_full_precision_in_vllm(self):
        o = original()
        self.assertEqual(edits.held_dtype(ALOG), torch.float32)
        self.assertEqual(edits.footprint(o, {ALOG: o[ALOG] + 1e-5})[ALOG]["changed"], 16)

    def test_a_change_big_enough_for_bfloat16_survives(self):
        o = original()
        new = o[NORM].clone()
        new[0] = 1.5
        self.assertEqual(edits.footprint(o, {NORM: new})[NORM]["changed"], 1)


class Scripts(unittest.TestCase):
    def script(self, body):
        d = Path(tempfile.mkdtemp())
        (d / "e.py").write_text(textwrap.dedent(body))
        return d / "e.py"

    def test_a_script_returns_only_what_it_changed(self):
        o = original()
        s = self.script(f"""
            def edit(t):
                w = t["{MLP}"]
                w[0, 0] = 1.0
                return {{"{MLP}": w}}
        """)
        self.assertEqual(list(edits.run_script(s, o)), [MLP])

    def test_a_script_cannot_touch_the_original(self):
        o = original()
        before = o[MLP].clone()
        s = self.script(f"""
            def edit(t):
                t["{MLP}"].zero_()
                return {{}}
        """)
        edits.run_script(s, o)
        self.assertTrue(torch.equal(o[MLP], before))

    def test_scripts_are_seeded(self):
        o = original()
        s = self.script(f"""
            import torch
            def edit(t):
                return {{"{MLP}": t["{MLP}"] + torch.randn(64, 64).to(torch.bfloat16)}}
        """)
        self.assertTrue(torch.equal(edits.run_script(s, o)[MLP], edits.run_script(s, o)[MLP]))


class Control(unittest.TestCase):
    def setUp(self):
        self.o = original()
        new = self.o[MLP].clone()
        new[3, :200 % 64] += 0.1
        new[5] -= 0.2
        self.fp = edits.footprint(self.o, {MLP: new})

    def test_same_size_same_tensors(self):
        changed, made = edits.random_control(self.o, self.fp, 1)
        self.assertEqual(list(changed), [MLP])
        self.assertAlmostEqual(made[MLP]["l2"], self.fp[MLP]["l2"], delta=0.02 * self.fp[MLP]["l2"])
        self.assertAlmostEqual(made[MLP]["changed"], self.fp[MLP]["changed"], delta=0.02 * self.fp[MLP]["changed"])

    def test_positions_differ_from_the_edit_and_by_seed(self):
        a, _ = edits.random_control(self.o, self.fp, 1)
        b, _ = edits.random_control(self.o, self.fp, 2)
        self.assertFalse(torch.equal(a[MLP], b[MLP]))

    def test_same_seed_same_control(self):
        a, _ = edits.random_control(self.o, self.fp, 3)
        b, _ = edits.random_control(self.o, self.fp, 3)
        self.assertTrue(torch.equal(a[MLP], b[MLP]))

    def test_a_size_that_rounding_cant_match_is_refused(self):
        tiny = {MLP: dict(self.fp[MLP], l2=self.fp[MLP]["l2"] * 1e-4)}
        with self.assertRaises(edits.BadEdit):
            edits.random_control(self.o, tiny, 1)


class Readback(unittest.TestCase):
    base = {"language_model.model.layers.2.mlp.down_proj.weight": {"sha256": "a"},
            "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight": {"sha256": "b"},
            "language_model.model.embed_tokens.weight": {"sha256": "c"},
            "language_model.model.norm.weight": {"sha256": "d"}}

    def table(self, **changes):
        t = {k: dict(v) for k, v in self.base.items()}
        for key, value in changes.items():
            name = next(n for n in t if key in n)
            t[name]["sha256"] = value
        return t

    def test_group_names_match_across_checkpoint_and_fused_names(self):
        self.assertEqual(edits.group_key("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
                         edits.group_key("language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"))
        self.assertEqual(edits.group_key("model.language_model.embed_tokens.weight"), "embed_tokens")
        self.assertEqual(edits.group_key("model.language_model.norm.weight"), "final_norm")

    def test_the_original_changes_nothing(self):
        self.assertIsNone(edits.readback_problem(self.base, self.table(), []))

    def test_an_edit_that_landed_passes(self):
        self.assertIsNone(edits.readback_problem(self.base, self.table(down_proj="z"), ["2.mlp"]))

    def test_an_edit_that_never_reached_vllm_is_caught(self):
        self.assertIn("didn't reach", edits.readback_problem(self.base, self.table(), ["2.mlp"]))

    def test_a_change_in_the_wrong_block_is_caught(self):
        table = self.table(down_proj="z", in_proj_qkvz="z")
        self.assertIn("shouldn't", edits.readback_problem(self.base, table, ["2.mlp"]))


if __name__ == "__main__":
    unittest.main()
