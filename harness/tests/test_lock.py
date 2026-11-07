"""The lock on the untouched halves has to hold before any real data flows."""
import json
import unittest

import fixture
import common
import questions as q


class Lock(unittest.TestCase):
    def setUp(self):
        q.ATTEMPTS.unlink(missing_ok=True)

    def test_exploring_never_contains_untouched(self):
        for name in common.SETS:
            got = {r["id"] for r in q.exploring_questions(name)}
            self.assertEqual(len(got), fixture.N // 2)
            self.assertFalse(got & fixture.UNTOUCHED[name])

    def test_no_ticket_no_questions(self):
        for bad in (None, "C-01", object()):
            with self.assertRaises(q.UntouchedLocked):
                q.untouched_questions("math500", bad)

    def test_ticket_cannot_be_made_by_hand(self):
        with self.assertRaises(q.UntouchedLocked):
            q.Ticket(object(), "C-01")

    def test_ticket_with_no_logged_attempt_is_refused(self):
        ticket = q.open_attempt("C-01", "n", "s")
        q.ATTEMPTS.unlink()
        with self.assertRaises(q.UntouchedLocked):
            q.untouched_questions("math500", ticket)

    def test_attempt_is_on_disk_before_the_ids_open(self):
        ticket = q.open_attempt("C-02", "notehash", "scripthash")
        line = json.loads(q.ATTEMPTS.read_text().splitlines()[0])
        self.assertEqual((line["claim"], line["note_sha256"], line["script_sha256"]), ("C-02", "notehash", "scripthash"))
        got = {r["id"] for r in q.untouched_questions("math500", ticket)}
        self.assertEqual(got, fixture.UNTOUCHED["math500"])

    def test_a_claim_id_gets_one_attempt_even_after_a_crash(self):
        q.open_attempt("C-03", "n", "s")
        with self.assertRaises(q.UntouchedLocked):
            q.open_attempt("C-03", "n", "s")
        self.assertEqual(len(q.read_attempts()), 1)

    def test_every_attempt_stays_listed(self):
        first = q.open_attempt("C-04", "n", "s")
        q.close_attempt(first, {"finding": False})
        q.open_attempt("C-05", "n", "s")
        self.assertEqual([(a["claim"], a["event"]) for a in q.read_attempts()],
                         [("C-04", "opened"), ("C-04", "finished"), ("C-05", "opened")])


class Splits(unittest.TestCase):
    def tearDown(self):
        common.manifest.cache_clear()

    def edited_split(self, new):
        path = common.SPLITS / "math500_exploring.json"
        old = path.read_text()
        path.write_text(new)
        self.addCleanup(path.write_text, old)

    def test_a_changed_split_file_stops_everything(self):
        common.manifest.cache_clear()
        self.edited_split('["m0"]\n')
        with self.assertRaises(SystemExit):
            common.manifest()

    def test_overlapping_halves_stop_everything(self):
        common.manifest.cache_clear()
        path = common.SPLITS / "math500_exploring.json"
        old = path.read_text()
        leak = json.dumps(sorted(json.loads(old) + ["m7"]), indent=0) + "\n"
        self.edited_split(leak)
        m = json.loads((common.ML / "MANIFEST.json").read_text())
        m["sets"]["math500"]["exploring"]["sha256"] = common.sha256(path)
        before = (common.ML / "MANIFEST.json").read_text()
        (common.ML / "MANIFEST.json").write_text(json.dumps(m))
        self.addCleanup((common.ML / "MANIFEST.json").write_text, before)
        with self.assertRaises(SystemExit):
            common.manifest()


class Helpers(unittest.TestCase):
    def test_seeds_are_stable_and_differ(self):
        self.assertEqual(common.seed_for("m0", 0, 0), common.seed_for("m0", 0, 0))
        seeds = {common.seed_for(i, s, o) for i in ("m0", "m1") for s in range(4) for o in range(5)}
        self.assertEqual(len(seeds), 40)
        self.assertTrue(all(0 <= s < 2 ** 31 for s in seeds))

    def test_gsm8k_gold_loses_commas(self):
        self.assertEqual(q._rows("gsm8k")["gsm8k-test-0001"]["gold"], "3000")

    def test_math_prompt_carries_the_card_instruction(self):
        self.assertTrue(q.exploring_questions("math500")[0]["prompt"].endswith(common.MATH_SUFFIX))

    def test_ifeval_prompt_is_as_is(self):
        self.assertEqual(q.exploring_questions("ifeval")[0]["prompt"], "Say 0.")


if __name__ == "__main__":
    unittest.main()
