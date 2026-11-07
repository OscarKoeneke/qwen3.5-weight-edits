"""Grading behaviour, pinned on made-up answers."""
import unittest
from unittest import mock

import fixture  # noqa: F401
import grade


def out(text, finish="stop"):
    return {"id": "x", "sample": 0, "seed": 1, "tokens": 5, "text": text, "finish_reason": finish}


class Answers(unittest.TestCase):
    def test_answer_is_after_the_last_close(self):
        self.assertEqual(grade.answer_text("a</think>b</think> c ", "stop"), "c")

    def test_cap_hit_has_no_answer_even_with_a_close(self):
        self.assertIsNone(grade.answer_text("a</think>b", "length"))

    def test_unclosed_thinking_has_no_answer(self):
        self.assertIsNone(grade.answer_text("thinking only \\boxed{4}", "stop"))


class Math(unittest.TestCase):
    def check(self, gold, answer, want):
        self.assertEqual(grade.grade_math(gold, answer), want, (gold, answer))

    def test_equal_forms_pass(self):
        self.check("\\frac{1}{2}", "So the answer is \\boxed{0.5}.", True)
        self.check("\\frac{1}{2}", "The answer is \\boxed{\\frac{2}{4}}", True)
        self.check("4", "Thus \\boxed{4}", True)
        self.check("1000", "Total: \\boxed{1,000}", True)

    def test_the_last_box_counts_across_lines(self):
        self.check("7", "\\boxed{3}\n\nActually \\boxed{7}.", True)

    def test_two_boxes_in_one_sentence_are_read_as_a_set(self):
        self.check("7", "First \\boxed{3}, but corrected: \\boxed{7}", False)

    def test_wrong_answers_fail(self):
        self.check("4", "\\boxed{5}", False)
        self.check("\\frac{1}{2}", "\\boxed{2}", False)

    def test_text_answers(self):
        self.check("\\text{Evelyn}", "\\boxed{\\text{Evelyn}}", True)
        self.check("\\text{Evelyn}", "\\boxed{\\text{Adam}}", False)


class Rows(unittest.TestCase):
    def test_a_loop_is_a_cap_hit_and_a_fail_and_is_never_parsed(self):
        with mock.patch.object(grade, "grade_math", side_effect=AssertionError("parsed a loop")):
            row = grade.grade_one(("math500", "4", out("\\boxed{4}" * 3, "length")))
        self.assertEqual((row["passed"], row["cap_hit"], row["no_answer"]), (False, True, True))

    def test_unclosed_thinking_is_a_fail_but_not_a_cap_hit(self):
        row = grade.grade_one(("gsm8k", "4", out("\\boxed{4}")))
        self.assertEqual((row["passed"], row["cap_hit"], row["no_answer"]), (False, False, True))

    def test_a_right_answer_passes(self):
        row = grade.grade_one(("gsm8k", "4", out("hm</think>\n\\boxed{4}")))
        self.assertEqual((row["passed"], row["no_answer"], row["grade_error"]), (True, False, False))

    def test_score_rows_hold_no_answer_text(self):
        self.assertNotIn("text", grade.grade_one(("gsm8k", "4", out("hm</think>\\boxed{4}"))))

    def test_a_grader_crash_is_counted_not_hidden(self):
        with mock.patch.object(grade, "grade_math", side_effect=RuntimeError("boom")):
            row = grade.grade_one(("math500", "4", out("x</think>\\boxed{4}")))
        self.assertEqual((row["passed"], row["grade_error"]), (False, True))

    def test_pool_grades_in_order(self):
        rows = grade.grade_outputs("gsm8k", {"x": "4"}, [out("a</think>\\boxed{4}"), out("a</think>\\boxed{5}")], workers=2)
        self.assertEqual([r["passed"] for r in rows], [True, False])


class IFEvalGate(unittest.TestCase):
    def test_missing_sentence_data_stops_before_lm_eval_is_imported(self):
        import sys
        with mock.patch("nltk.data.find", side_effect=LookupError):
            with self.assertRaises(grade.IFEvalNotReady):
                grade.grade_one(("ifeval", {"key": 1, "instruction_id_list": [], "prompt": "p", "kwargs": []},
                                 out("a</think>b")))
        self.assertNotIn("lm_eval.tasks.ifeval.instructions_util", sys.modules)


if __name__ == "__main__":
    unittest.main()
