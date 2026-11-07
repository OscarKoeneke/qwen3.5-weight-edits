"""The 99% rule, on distributions with known answers."""
import unittest

import numpy as np

import fixture  # noqa: F401
import stats


def rows(scores, seed=0):
    """scores maps question id to a list of pass flags, one per sample."""
    return [{"id": q, "sample": i, "seed": seed + i, "passed": bool(p), "cap_hit": not p, "no_answer": not p,
             "grade_error": False, "tokens": 10} for q, flags in scores.items() for i, p in enumerate(flags)]


class Interval(unittest.TestCase):
    def test_same_input_same_interval(self):
        g = np.random.default_rng(1).normal(0.1, 0.3, 250)
        self.assertEqual(stats.interval(g), stats.interval(g))

    def test_matches_the_known_spread_of_a_mean(self):
        g = np.random.default_rng(2).normal(0.0, 1.0, 2000)
        lo, hi = stats.interval(g)
        self.assertAlmostEqual((hi - lo) / 2, 2.576 / 2000 ** 0.5, delta=0.008)

    def test_clear_gain_clears_zero_and_null_does_not(self):
        rng = np.random.default_rng(3)
        self.assertGreater(stats.interval(rng.normal(0.3, 0.3, 250))[0], 0)
        lo, hi = stats.interval(rng.normal(0.0, 0.3, 250))
        self.assertTrue(lo < 0 < hi)

    def test_uses_the_spec_resample_count(self):
        self.assertEqual(stats.RESAMPLES, 10000)


class Pairing(unittest.TestCase):
    def test_gain_is_per_question_over_samples(self):
        a = rows({"q1": [1, 1, 1, 1], "q2": [1, 0, 0, 0]})
        b = rows({"q1": [1, 1, 0, 0], "q2": [1, 0, 0, 0]})
        self.assertEqual(list(stats.paired_gains(a, b)), [0.5, 0.0])

    def test_unpaired_seeds_are_refused(self):
        with self.assertRaises(ValueError):
            stats.paired_gains(rows({"q1": [1]}, seed=0), rows({"q1": [1]}, seed=9))

    def test_different_questions_are_refused(self):
        with self.assertRaises(ValueError):
            stats.paired_gains(rows({"q1": [1]}), rows({"q2": [1]}))

    def test_no_loop_metric_rises_when_cap_hits_fall(self):
        a = rows({"q1": [1, 1]})
        b = rows({"q1": [0, 0]})
        self.assertEqual(list(stats.paired_gains(a, b, "no_loop")), [1.0])

    def test_best_by_picks_highest(self):
        self.assertEqual(stats.best_by({"r1": rows({"q": [0, 0]}), "r2": rows({"q": [1, 0]})}), "r2")


class Verdict(unittest.TestCase):
    up = {"lo": 0.01, "hi": 0.2}
    flat = {"lo": -0.01, "hi": 0.2}
    down = {"lo": -0.2, "hi": -0.01}

    def test_all_clear_is_a_finding(self):
        self.assertTrue(stats.verdict(self.up, self.up, self.up, {"gsm8k": self.flat})["finding"])

    def test_each_check_can_fail_it_alone(self):
        for args in ((self.flat, self.up, self.up, {}), (self.up, self.flat, self.up, {}),
                     (self.up, self.up, self.flat, {}), (self.up, self.up, self.up, {"gsm8k": self.down})):
            self.assertFalse(stats.verdict(*args)["finding"])

    def test_a_drop_that_could_be_noise_is_not_damage(self):
        v = stats.verdict(self.up, self.up, self.up, {"ifeval": self.flat})
        self.assertEqual(v["damaged"], [])

    def test_damage_names_what_broke(self):
        v = stats.verdict(self.up, self.up, self.up, {"ifeval": self.down, "gsm8k": self.flat})
        self.assertEqual(v["damaged"], ["ifeval"])


if __name__ == "__main__":
    unittest.main()
