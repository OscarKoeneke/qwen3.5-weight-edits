# Can weight edits stop Qwen3.5-0.8B's thinking loops?

A small study, run 3 to 7 November 2026 on one RTX 4090. It edits the weights of Qwen3.5-0.8B directly, with no training, to see whether that stops the model's thinking loops. The plan is in `SPEC.md`, the side study of where loops show up is in `LOOP_MAP.md`, and every step is dated in `TIMELINE.md`. Code, edit scripts, exploring splits and per-answer scores are in this repo (see "What's in this repo" at the end).

## The answer

The question was whether editing the weights of Qwen3.5-0.8B directly, with no training, can stop its thinking loops. **The study found no confirmed finding.**

- One edit, E-002, made the Gated DeltaNet memory in layers 4 to 22 forget twice as fast. It cut loops and raised the pass rate on exploring MATH-500 questions from 18.2% to 28.7%, a paired gain of +10.5 points [+6.6, +14.5].
- But a free sampler setting (repetition penalty 1.1) does about as well, 26.9%, and E-002 only ties it: +1.8 [-2.1, +5.6].
- No random edit of the same size was run, so it isn't shown that the direction of E-002's change matters.
- The two other edits made the model worse.

No edit earned the confirmation run, so no untouched question was ever sent to the model.

## What was set up

- **Model:** `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, text only, in thinking mode with the model card's sampling settings and a 32,768-token cap. Engine: vLLM 0.33.0 on one RTX 4090.
- **Sets:** MATH-500, GSM8K and IFEval, each split once at random (seed 20261103) into an exploring half and a locked untouched half, before any model output existed. The hashes are in `MANIFEST.json` and `TIMELINE.md` (3 Nov).
- **A finding needs five checks** (`SPEC.md`):
  1. confirmed once on untouched questions;
  2. bigger than run-to-run noise;
  3. beats a random edit of the same size and the best settings fix;
  4. no damage on the other sets;
  5. released with everything needed to rerun it.
- **Statistics:** every gap below is paired by question, with a 99% bootstrap interval (10,000 resamples of questions, seed 20261103). All of them use one seed offset, so they carry no run-to-run noise (see "What the study can't say").

## Baseline: how often the model loops

Unedited model, default settings, exploring halves, 4 answers per question:

| Set | Pass | Hit the cap |
|---|---|---|
| MATH-500 (250 questions) | 18.2% | 81.3% |
| GSM8K (660 questions) | 33.6% | 64.0% |
| IFEval (271 prompts) | 57.1% | 15.6% |

Loops are the main failure. On MATH-500, 182 of the 187 answers that finished were right (97.3%), so the pass rate is close to the finish rate. 793 of the 813 capped answers were still thinking at the cap, and most of them repeat the same few doubts.

## Settings grid: the cheap fix

Five sampler settings on MATH-500 exploring questions, one run each, against the default (presence 1.5, repetition 1.0):

| Presence, repetition | Pass | Hit the cap | Gap in pass to the default |
|---|---|---|---|
| 1.5, 1.0 (default) | 18.2% | 81.3% | n/a |
| 1.5, 1.05 | 22.6% | 76.5% | +4.4 [+1.7, +7.2] |
| **1.5, 1.1** | **26.9%** | 62.0% | **+8.7 [+5.1, +12.4]** |
| 2.0, 1.0 | 20.3% | 78.8% | +2.1 [-0.5, +4.8] |
| 2.0, 1.05 | 24.5% | 71.6% | +6.3 [+3.2, +9.5] |
| 2.0, 1.1 | 25.4% | **41.1%** | +7.2 [+3.3, +11.2] |

- The repetition penalty does most of the work. Presence alone does little.
- (1.5, 1.1) is best on pass and (2.0, 1.1) on loops. The top three on pass can't be told apart.
- (2.0, 1.1) halves the loops but breaks answers. On the 78 questions the default passed at least once, its finished answers are right only 75.0% of the time.

## Loop map: where loops show up inside the model

Saved answers were read back through the model to compare matched pairs: the same question answered once with a loop and once without. Exploring answers only, 22 minutes of GPU time.

- **No clean lead layer.** A direction fitted just before a loop picks the looping answer in 88 to 95% of held-back pairs at every layer, layer 0 included. Plain token counts, with no model at all, do 86%. So most of the signal is in the words, not in a particular layer.
- **Loops build gradually.** At 256 tokens before a loop starts, 68% of the recent text already repeats, against 44% in finished answers at the same point.
- **Two weak leads:**
  - Inside a loop the model is far more certain of its next token, and that extra certainty appears only in layers 22 and 23.
  - The Gated DeltaNet running state changes about 35 to 55% less inside a loop in layers 4 to 22.

  Both are correlation. Repeated input alone could explain the second.
- By the plan's stop rule, no edit follows from the map. The three edits below are ordinary exploring edits guided by the two weak leads.

## Three exploring edits

Each edit is a short Python script that changes named weights, checked by reading the loaded weights back before any answer is generated. Each ran once on the 250 MATH-500 exploring questions (1,000 answers). All runs used the default settings.

| Edit | What it changes | Pass | Hit the cap | Finished answers right | Gap to the default | Gap to (1.5, 1.1) |
|---|---|---|---|---|---|---|
| Default | nothing | 18.2% | 81.3% | 97.3% | n/a | -8.7 [-12.4, -5.1] |
| E-001 | layers 22 and 23 write 10% less into the residual stream | 12.6% | 87.3% | 99.2% | -5.6 [-8.4, -3.0] | -14.3 [-19.1, -9.7] |
| **E-002** | **DeltaNet memory in layers 4 to 22 forgets twice as fast** | **28.7%** | 67.4% | 88.0% | **+10.5 [+6.6, +14.5]** | **+1.8 [-2.1, +5.6]** |
| E-003 | the same, four times as fast | 13.7% | 80.1% | 68.8% | -4.5 [-8.0, -1.0] | -13.2 [-17.6, -9.1] |

How E-002 and E-003 work: each Gated DeltaNet head fades its running memory by a fixed factor every token. Adding ln 2 to the `linear_attn.A_log` weights squares that factor for every head, whatever the input, so each head's memory half-life halves. Adding ln 4 quarters it. That's 16 numbers per layer in 15 layers, 240 numbers in a model of about 0.8 billion.

### Easy and hard questions

The grid's best setting passes more by letting harder questions finish, and some of those finish wrong. To compare like with like, the questions are split by whether the default passed them at least once (78 "easy") or never (172 "hard"):

| Run | Easy: pass | Easy: finished right | Hard: pass | Hard: finished right |
|---|---|---|---|---|
| Default | 58.3% | 99.5% | 0.0% | 0.0% (4 finished) |
| (1.5, 1.1) | 71.2% | 95.3% | 6.8% | 32.0% |
| E-001 | 40.1% | 100.0% | 0.1% | 50.0% (2 finished) |
| **E-002** | **70.5%** | **98.7%** | **9.7%** | **65.0%** |
| E-003 | 38.1% | 87.5% | 2.6% | 28.6% |

The groups are picked from the default's own results, which favours the other runs on the hard questions and works against them on the easy ones.

### What the edits suggest (exploring evidence, not claims)

- **Faster forgetting in the DeltaNet memory cuts loops, up to a point.** Twice as fast helped. Four times as fast loops as often as the default and breaks finished answers. So the strength of the change matters, with a sweet spot near twice.
- **E-002 helps in a different way from the setting.** It ties on pass, but it keeps easy answers right (98.7% against 95.3%), and its finished answers on hard questions are right twice as often (65.0% against 32.0%). This is one run each, on a split chosen after the fact.
- **Turning down layers 22 and 23 made loops worse**, not better. That is a real effect on these questions, but in the wrong direction.
- E-002 is the best of three edits picked on the same questions, which flatters it a little.

## The five checks for E-002

| Check | Status |
|---|---|
| 1. Untouched questions | Not run. No claim note was written and no confirmation attempt exists, so no untouched question was sent to the model or scored. |
| 2. Noise | Can't be judged. Repeats 2 to 5 of the noise run never ran, so run-to-run noise is unmeasured. |
| 3. Controls | Settings fix: a tie on exploring questions. Random edit of the same size: not run. |
| 4. Damage | Not run. E-002 was never run on GSM8K or IFEval. |
| 5. Release | Doesn't apply. Nothing was confirmed. |

## What the study can't say

- **Nothing here is confirmed.** Every number above comes from exploring questions, which were used to choose the settings and the edits.
- **Run-to-run noise was never measured.** The plan's noise rule rests on 5 repeats of the unedited model. Only the first ran, because the rest (about 36 hours) were to run only if an edit earned a confirmation. Every interval resamples questions at one set of seeds. Running the same model again with other seeds would move each score by an amount that is unknown.
- **Only MATH-500 was tested** for the grid and the edits. The baselines on GSM8K and IFEval exist, but nothing was compared on them.
- **No random control ran,** so an edit of E-002's size in a random direction might do as well.
- **One model, one size, one engine.**

## Untested next steps

Each would need its own go. None is planned.

- A random edit of E-002's size, about 2.5 tower hours. It would show whether the direction of the change matters.
- E-002 on GSM8K and IFEval, as a first damage check.
- E-002 together with the (1.5, 1.1) setting. This needs a small harness change, because explore runs at the default settings only.
- Strengths between twice and four times, such as 1.5 or 3 times.
- The restart screener from `LOOP_MAP.md`: restart a saved answer just before its loop and generate 2,000 tokens with the edit in place, as a cheap filter for edits.

## Cost

Tower time, measured as generation time per run. It leaves out model loading, grading, smoke tests and stand-in wiring tests.

| Step | Tower time |
|---|---|
| Baseline, all three sets | 8.9 h (32,005 s) |
| Settings grid, 5 points | 11.2 h (40,397 s) |
| Loop map, stages 1 and 2 | 0.4 h (22 min) |
| Three edits | 7.3 h (9,548 + 8,078 + 8,569 s) |
| **Total** | **about 27.8 h** |

An exploring edit costs about 2.2 to 2.7 hours, at 3,213 to 3,322 tokens a second, because most answers run long. Claude usage per edit wasn't measured.

## What's in this repo

| Path | What it is |
|---|---|
| `SPEC.md` | The plan, written before any model output existed, with the details fixed by the harness build in section 11 |
| `LOOP_MAP.md` | The plan for the loop map, written before it ran |
| `TIMELINE.md` | Dated record of every run, check and correction |
| `MANIFEST.json` | Model and data revisions, file hashes, split hashes |
| `splits/` | The exploring halves (question IDs) |
| `harness/` | The harness, its tests (`unittest`, with a fake GPU), the loop map scripts in `harness/map/` and the edit scripts in `harness/edits/` |
| `results/` | For each run: `config.json` (settings, versions, harness hashes, the edit's weight footprint), `summary.json`, `cells.log` and per-answer `scores-*.json`; plus `baseline.json`, `grid_math500.json` and `PASSES.log` |

Edit scripts and their SHA-256:
- `E-001-late-layers-0.9.py`: `adcb41e2d9460469f0fe819e7a5e18bcf626e488bba8d9edfebf6f16403a78e6`
- `E-002-deltanet-forget-x2.py`: `17c0b2d6d920bfb6268ff0d70e597ce7410f286b31519f27522484e633c3a201`
- `E-003-deltanet-forget-x4.py`: `2c4d4f185e4855f7b27014f698a4ffb61510d7bc323f72e62fe54d5858a5e6af`

Runs in `results/`:
- `noise_20261106-013258-796095_original`: the baseline, all three sets
- `grid_20261106-131418-868511_original`: the settings grid
- `explore_20261107-124323-442503_E-001-late-layers-0.9-adcb41e2`
- `explore_20261107-152413-020897_E-002-deltanet-forget-x2-17c0b2d6`
- `explore_20261107-174715-148775_E-003-deltanet-forget-x4-2c4d4f18`

The harness files here match the hashes recorded in the grid and edit runs' `config.json`. The baseline ran on earlier versions of `core.py` and `cli.py`: the later changes added the 5-repeat rule for claims and `grid --resume`, and don't touch how answers are generated or graded (see `TIMELINE.md`, 6 Nov).

A score row has `id`, `sample`, `seed`, `passed`, `cap_hit`, `no_answer`, `grade_error` and `tokens`. The paired gaps above come from `harness/stats.py` on these rows.

**Not included:**
- The untouched halves. They were never sent to the model, and they stay unpublished so they can still serve as a clean test. `harness/fetch_and_split.py` is the script that made the split. It takes each repo's latest revision, so to rebuild the same halves, pin it to the revisions in `MANIFEST.json` and compare the hashes there.
- The answer texts (26 to 31 million tokens per MATH-500 run) and the job files, which repeat the questions' text.
- The loop map's outputs.
- The GPU's serial number, replaced with `GPU-redacted` in `results/`.

## Rerun

The harness expects a CUDA machine with vLLM 0.33.0, torch 2.14.0, transformers 5.20.2, math-verify 0.9.0 and lm-eval 0.4.13 (IFEval also needs NLTK's `punkt_tab`), with the model, data and splits under `~/ml` (`HARNESS_ROOT` points the harness at a stand-in folder, for tests). `harness/fetch_and_split.py` downloads into that layout; see the note on revisions above. Then, from the repo root, for example:

```
python harness/cli.py explore --edit harness/edits/E-002-deltanet-forget-x2.py
```

vLLM isn't bit-exact across batches, so a rerun gives the same seeds but not identical text. Expect scores close to these, not equal.

## License

Code: MIT (`LICENSE`). Qwen3.5-0.8B is Apache-2.0. MATH-500, GSM8K and IFEval are under their own licenses and aren't redistributed here beyond question IDs.
