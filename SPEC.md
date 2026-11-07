# Weight-edit study: harness spec

Draft of 3 November 2026, written before any model output existed. Section 11 was added on 5 Nov, before the first real run, with one dated note after that. This public copy changes only this line and the log's name: `TIMELINE.md` is a public summary of the private study log.

## What the study is for

Oscar's call of 3 Nov 2026 is that findings come first. Claude edits the weights of small open models directly, with no training, on the RTX 4090 tower. An edit counts as a finding only if it passes all five checks:

1. **Untouched questions.** It is confirmed once, on questions that were locked away before exploring started.
2. **Noise.** The gain is bigger than the model's measured run-to-run noise.
3. **Controls.** It beats a random edit of the same size and the best easy settings fix.
4. **Damage.** No other measured ability gets worse.
5. **Release.** It is published with the weights, the exact change, the code and the settings, so anyone can rerun it.

Claude's predictions are not scored. Exploring is free and unlimited, apart from the usage budget. Only confirmation runs make claims.

## 1. Model

**`Qwen/Qwen3.5-0.8B`**, text only (`--language-model-only`).
- Apache-2.0, not gated.
- 24 layers and hidden size 1024. The layout is 6 blocks, each made of 3 Gated DeltaNet layers and 1 gated attention layer, with an FFN after each.

**Why this model:**
- It is the model in the 2 Nov weight-space programming report, so the first finding can be set beside theirs.
- Its own model card warns that, in thinking mode, it "is more prone to thinking loops" than the other Qwen3.5 sizes. That is a known, measurable failure.

Larger models or other families come later. Each one is chosen after the first confirmed finding, with its own download go.

The download revision (commit hash) is recorded in `MANIFEST.json`.

## 2. First failure to target

**Thinking loops:** answers that run until they hit the output cap.

This is measured, not assumed. The noise run (section 5) reports the share of answers that hit the cap under our settings. If that share is small, the first target is chosen from the noise run's other failures, and the choice is logged before exploring starts.

## 3. Test sets and the split

| Set | Source | Items | Exploring | Untouched | Role |
|---|---|---|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500`, test | 500 | 250 | 250 | Primary target (math) |
| GSM8K | `openai/gsm8k`, main, test | 1,319 | 660 | 659 | Second math set; damage check for non-math targets |
| IFEval | `google/IFEval`, train | 541 | 271 | 270 | Damage check (instruction following) |

**How the split is made:**
- Each set is split at random with a fixed seed. Sorted item IDs are shuffled with `random.Random(20261103)`; the first half is exploring and the rest is untouched.
- The split happens once, after download and before any model output exists on these questions.
- The ID lists and their SHA-256 hashes are written to `splits/` and to `TIMELINE.md`.

**How the untouched halves are protected:**
- The harness's exploring commands accept exploring IDs only.
- The confirm command needs the path of a claim note. It records that note's SHA-256 before it opens the untouched IDs, and it logs every confirmation attempt.
- Every attempt is reported, including the ones that fail.

**Later addition:** a general-knowledge damage set (an MMLU-style sample). It needs its own download go.

## 4. Settings, frozen for every run

- **Mode:** thinking. The sampling values are the model card's values for text in thinking mode:

  | Setting | Value |
  |---|---|
  | temperature | 1.0 |
  | top_p | 0.95 |
  | top_k | 20 |
  | min_p | 0.0 |
  | presence_penalty | 1.5 |
  | repetition_penalty | 1.0 |

- **Output cap:** 32,768 tokens, the card's suggested length.
- **Samples:** 4 per question, with a fixed seed per question and per sample. Original and edited models get the same seeds, so they can be compared question by question.
- **Grading:**
  - MATH-500 and GSM8K: answers are checked with `math-verify`.
  - IFEval: the strict prompt-level score, from the `lm-eval` IFEval checker.
- **Engine:** vLLM. The versions are recorded in `MANIFEST.json`.

Our settings differ from the 2 Nov report, which used temperature 0.6, top_p 0.95, top_k 20 and presence penalty 1.5. So their numbers are context, not a baseline. Our baseline is the original model under our settings.

## 5. Noise

**How it is measured:**
- The unchanged model runs over the exploring halves 5 times, with 5 different seed offsets.
- For each metric, the noise is the standard deviation of the pass score across the 5 repeats.
- The same run also gives:
  - the paired run-to-run difference distribution, which the bootstrap rule below uses;
  - the cap-hit rate.

**The rule for "bigger than the noise":** on the untouched questions, the paired gain (edited minus original, over questions, averaged across samples) has a 99% bootstrap interval that lies entirely above zero. The bootstrap uses 10,000 resamples of questions, seed 20261103.

There is one rule for all claims. The bar is 99% rather than 95% because the study makes several claims. Every confirmation attempt is listed, so readers can count them.

## 6. Controls

Each confirmed edit is compared on the same untouched questions with two controls.

1. **Random edit of the same size.** It touches the same layers and the same number of weights, with the same norm of change, but in a random direction. Three random seeds are used, and the edit must beat the best of the three by the rule in section 5.
2. **Easy settings fix.** The best of a small grid, tuned on the exploring halves only. The grid is presence_penalty in {1.5, 2.0} and repetition_penalty in {1.0, 1.05, 1.1}. The edit must beat the best grid point by the same rule.

## 7. Damage check

For each confirmed edit, every other set's untouched half is run, edited against original.

**Damage** means the paired change has a 99% bootstrap interval lying entirely below zero. An edit with damage is not a finding. It is reported as a failure, with what it broke.

## 8. Claude's exploring sandbox

- Claude explores from the Mac over `ssh tower`.
- **Tools.** Python on the tower with the model weights, plus a test-pass command that runs on exploring IDs only. Every test pass is counted and logged.
- **Budget.** The first exploring session measures how much Claude usage and tower time one edit costs. Later sessions are sized to Oscar's weekly usage.
- **No sudo.** Downloads beyond this spec need Oscar's go.

## 9. Claim note

Each note is written before its confirmation run. Its SHA-256 is recorded by the confirm command. Fields:

```
claim id      C-NN
model         repo and revision
change        the exact edit, with a script path and its SHA-256
claim         what gets better, in one sentence
primary       set and metric
success       the section 5 rule on the primary metric, beats both controls, no damage
written       UTC time
```

## 10. Release, per finding, on Oscar's go

- the edited weights (Hugging Face);
- the edit script, which turns the original revision into the edited model;
- the harness at the run's commit;
- the split files;
- every number (noise, controls, damage, confirmation), with all confirmation attempts including failures.

## Files

- On the tower, under `~/ml/`:
  - `eval-venv/`
  - `models/`
  - `data/`
  - `splits/`
  - `harness/`
  - `runs/`
- On the Mac, in this folder:
  - `SPEC.md`
  - `TIMELINE.md`
  - copies of `splits/` and `MANIFEST.json`
  - `harness/`, the source that is copied to the tower

## 11. Details fixed by the harness build

Added 5 Nov 2026. Sections 1 to 10 left these open. They were fixed in the code before the first tower run and written up the same day. Any of them can still change before the noise run. After that, a change needs its own dated log entry.

**Prompts.** No system prompt. MATH-500 and GSM8K questions get the model card's line on a new line after the question: "Please reason step by step, and put your final answer within \boxed{}." IFEval prompts go in as they are. The rendered prompt must end in `<think>\n`, or the run stops, because this model is non-thinking by default.

**What counts as an answer.** The text after the last `</think>`. An output that hits the cap, or that stops without closing its thinking, has no answer and fails. Only an output that hits the cap counts as a loop. The other kind is counted separately, as `no_answer`.

**How `math-verify` reads boxes.** The harness uses it unchanged. It takes the last box across separate lines, and it reads two boxes in one sentence as a set, so "\boxed{3}, but corrected: \boxed{7}" does not match 7. A test pins this behaviour.

**Seeds.** One request per sample, each with its own seed: the first 4 bytes of SHA-256 of `id|sample|offset`, masked to 31 bits. Confirmation and exploring use offset 0. Noise repeats use offsets 0 to 4. Same seeds do not mean identical text, because vLLM isn't bit-exact across batches, so comparisons rest on the paired interval and not on exact matches.

**Metrics.** A claim's primary metric is `pass` (pass rate) or `no_loop` (share of answers that didn't hit the cap). Damage always uses `pass`.

**What vLLM holds.** It holds `linear_attn.A_log` in float32 and everything else in bfloat16, including the float32 `linear_attn.norm.weight` tensors in the checkpoint. Edit sizes are measured on those held values, and an edit that rounds away there is refused. Before any answer is generated, the worker reads the loaded weights back and stops unless they differ from the original in exactly the expected blocks.

**Edits.** An edit is a Python file with `edit(tensors)`, returning only the tensors it changed. It may touch language model tensors only, since vLLM doesn't load the vision tower or the MTP head here. Confirmation rebuilds the weights from the script, so what is confirmed is what the script makes.

**Random control.** Per tensor: the same tensors, the same number of changed entries at random positions, and Gaussian values scaled to the same norm of change, within 2%, measured on the values vLLM holds. Seeds 1, 2 and 3. The "best of the three" is the one with the highest mean on the claim's primary metric, on the untouched questions.

**Settings grid.** The default point reuses the noise run's offset 0 pass, so 5 points are run. The best point is picked on the exploring half, per metric. If the best point is the default, the settings control is the original model's own run.

**Bootstrap.** numpy's `default_rng` with the study seed. The interval is the 0.5th to 99.5th percentile of the resampled means.

**One attempt per claim id.** There is no resume for confirmation. A crash leaves an open attempt in `CONFIRM_ATTEMPTS.log`, and a new try needs a new claim id, so readers can count it. A note can be checked in full first with `claimcheck`, which opens nothing.

**Engine.** vLLM runs in the worker's own process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the loaded weights can be read back, with prefix caching off, `gpu_memory_utilization` 0.9 and a 2,048-token prompt limit on top of the output cap.

**Records.** Each pass writes `runs/<kind>_<time>_<label>/` with its full configuration, a hash of every harness file, its outputs and its scores, and adds a line to `runs/PASSES.log`. The folder isn't a git repository, so those hashes stand in for "the harness at the run's commit" in section 10.

**Order of the noise repeats.** Added 6 Nov 2026 on Oscar's decision, after the noise run started (see `TIMELINE.md`). Exploring and the settings grid compare against offset 0 only, so they can start once offset 0 is finished on all three sets. All 5 repeats of section 5 must be on record before any confirmation, and the harness refuses a claim in the real folder until they are.
