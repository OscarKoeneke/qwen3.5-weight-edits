"""Grading (SPEC.md section 4). An answer is the text after the last </think>.

An output that hit the cap or never closed its thinking has no answer and fails, without being parsed.
lm-eval's IFEval code downloads NLTK data when imported, so it is imported only after require_punkt() has passed.
"""
import multiprocessing

SENTENCE_DATA = "tokenizers/punkt_tab"


class IFEvalNotReady(Exception):
    pass


def require_punkt():
    import nltk
    try:
        nltk.data.find(SENTENCE_DATA)
    except LookupError:
        raise IFEvalNotReady("IFEval grading needs NLTK punkt_tab, which isn't on this machine. It needs its own go.")


def answer_text(text, finish_reason):
    if finish_reason == "length" or "</think>" not in text:
        return None
    return text.rsplit("</think>", 1)[1].strip()


def grade_math(gold, answer):
    from math_verify import parse, verify
    return bool(verify(parse(f"\\boxed{{{gold}}}"), parse(answer)))


def grade_ifeval(doc, answer):
    require_punkt()
    from langdetect import DetectorFactory
    from lm_eval.tasks.ifeval import utils
    DetectorFactory.seed = 0
    inp = utils.InputExample(key=doc["key"], instruction_id_list=doc["instruction_id_list"], prompt=doc["prompt"],
                             kwargs=doc["kwargs"])
    return bool(utils.test_instruction_following_strict(inp, answer).follow_all_instructions)


def grade_one(item):
    """item is (set name, gold, output row). Returns the score row, with no answer text in it."""
    name, gold, out = item
    answer = answer_text(out["text"], out["finish_reason"])
    row = {"id": out["id"], "sample": out["sample"], "seed": out["seed"], "tokens": out["tokens"], "passed": False,
           "cap_hit": out["finish_reason"] == "length", "no_answer": answer is None, "grade_error": False}
    if answer is None:
        return row
    try:
        row["passed"] = grade_ifeval(gold, answer) if name == "ifeval" else grade_math(gold, answer)
    except IFEvalNotReady:
        raise
    except Exception:
        row["grade_error"] = True
    return row


def grade_outputs(name, gold_by_id, outputs, workers=8):
    """math_verify times out with signal.alarm, which needs a main thread, so grading uses processes and not threads."""
    items = [(name, gold_by_id[o["id"]], o) for o in outputs]
    if not workers:
        return [grade_one(i) for i in items]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return pool.map(grade_one, items, chunksize=16)
