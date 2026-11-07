# Loop map: plan

Written 6 Nov 2026, before any of it ran, as a cheap side study. What it found is in `README.md` ("Loop map") and `TIMELINE.md` (6 Nov).

## The question

Where in Qwen3.5-0.8B does a thinking loop show up, and does aiming an edit there beat guessing?

This counts as exploring under `SPEC.md` section 8, so it makes no claims. It uses saved noise-run answers from the exploring halves only and never reads an untouched file. Its result is recorded either way, including a null.

## Why it's cheap

Nothing in stages 1 and 2 generates new text. Stage 1 reads saved text on the CPU. Stage 2 reads saved answers back through the model once, and that should be far faster than the 3,256 tokens a second of generation, though it's unmeasured.

## Which answers

Compare answers to the same question, one that looped and one that finished. Comparing all looped answers with all finished ones would mostly measure length and difficulty, since capped answers are always 32,768 tokens and finished ones have a median of 13,000 to 16,000.

Saved so far, 331 questions have both kinds: 69 on MATH-500 and 262 on GSM8K. A first pass uses about 100 of them. A third of the questions are held back to check that what the map finds still holds on questions it wasn't built from.

## Stages

| Stage | What it does | GPU time | When |
|---|---|---|---|
| 1. Text | Find where each loop starts and how long its repeat is. Report how many answers have a clean start, since this is a heuristic. Also count how many looping answers had already reached the right answer before the loop. | None | On the tower's CPU during IFEval |
| 2. Reading pass | Feed saved answers through the model and compare each layer's activity before and after the loop starts, against the matched finished answer. The earliest layer that changes at the start is the lead. | Measured first, see below. Capped at 1 hour | After IFEval, never alongside it |
| 3. Test | Turn the 2 or 3 lead layers into edits and run them through `explore`. | About 2.5 hours per edit, the same edits we'd run anyway | Each on its own go |

**Stage 2 cost gate.** Load the model in Hugging Face transformers and time one full 32,768-token answer, then project the cost as that time multiplied by the number of answers. 18 of the 24 layers are Gated DeltaNet, and they may fall back to slow code if `flash-linear-attention` isn't there. If the projection is over 1 hour, I shrink the sample or stop and ask. Installing that package would need its own go.

**What stage 2 looks at first.** Gated DeltaNet layers carry a running state, and a loop may be that state ceasing to change. If transformers exposes it, the size of its change at each step is the first signal to check, ahead of layer outputs. Results are reduced on the GPU per answer, with no full activations saved.

**Cheap pre-check for stage 3, if it proves worth building.** Restart a saved answer from just before its loop and generate 2,000 tokens with the edit in place. That tests an edit in minutes instead of 2.5 hours. It would be a new small script, with every pass logged as section 8 requires.

## Stop rules

- If stage 1 finds few clean loop starts, stop and record that.
- If stage 2 finds no layer that changes at the start, or the signal fades on the held-back questions, stop and record a null. No edits follow from it.
- GPU time for the map itself is capped at 1 hour without a new go.

## How it joins the experiment

The study already compares each confirmed edit with a random edit of the same size. So the map gets a fair test for free: if map-aimed edits don't beat random ones, the map didn't help, and we report that. A map-aimed edit still has to pass all five checks.

## Limits

- Stage 2 is correlation. It shows where a loop is visible, not where it starts. Only stage 3 tests cause.
- Transformers and vLLM differ slightly in their numbers. That's fine for a map, which is why any edit is then tested in vLLM.
- Saved text is re-tokenized, because the outputs keep text and not token ids, so tokens may differ slightly from what was generated.
- The presence penalty of 1.5 shaped these answers in the sampler, outside the model. The map sees sampler-shaped text.

## Practical rules

- Scripts live in `harness/map/`, with hashes in `TIMELINE.md`.
- A map script takes the harness's GPU lock, or runs only when no harness run is active.
- Each named run needs its own go from the study's owner.
