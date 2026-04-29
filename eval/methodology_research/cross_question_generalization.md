# Cross-question generalization — design

The three experiments to date were all run on Q* (a single
contested question). The headline finding from `findings_multipool.md`
— that pool draws are asymmetric around gold — could be either
**a general property of contested questions** or **Q*-specific**.

This experiment tests which.

## Hypothesis

The §3 orthogonal decomposer's "average draw" is biased toward
whichever reasoning framework is more common in the model's
"expert's first instinct" prior for the question's topical area.
On most contested questions, this bias direction is mixed —
sometimes toward gold, sometimes away. Q*'s pool asymmetry may be
a strong instance of an effect that exists in milder form across
many questions, **or** Q* may be unusual.

## Predicted outcomes

- **If pool asymmetry is general**: most contested questions tested
  show ≥3 of 5 pools landing in the same framework direction. Cross-
  pool ensembling fails reliably; baseline-pool single-pool results
  are unreliable.
- **If pool asymmetry is Q*-specific**: most contested questions
  show roughly random pool direction (≈50/50 split across pools).
  Single-pool results are noisy but unbiased; cross-pool ensembling
  rescues accuracy as expected.

## Experimental design

### Question selection

Sample from the unused GPQA Diamond pool. Use
`csv_to_questions.py --n 5 --seed 99 --update-ledger` (or similar
unused seed) to get 5 questions. Filter to **contested** questions:

- Run baseline §3 round 0 + round 1 at N=4 on each (cheap: 9 calls
  per question).
- Identify questions where at least one persona disagrees at round 0
  AND consensus does not converge to unanimity by round 1. These are
  contested.
- Discard non-contested questions (unanimous round-0 consensus is
  the trivial case where pool draws don't matter).

Goal: 2–3 contested questions for the multi-pool generalization
sweep.

### Multi-pool generalization sweep

For each of the 2–3 contested questions:

- Generate 4 independent decomposer pools (brainstorm-20, §3
  orthogonal, identical to multipool experiment).
- Per pool: round-0 evals at N=4 only (smallest N where pool effects
  are visible).
- Plus the baseline pool (already exists from question selection
  step).

Total: 4 new pools × 4 personas × N=1 = 16 calls per pool ×
4 pools = 64 calls per question. 2–3 questions = 128–192 calls.

### Variants per question

For at least one contested question, also run:

- **v_judge** (baseline-J aggregator only) on the existing pools.
  Compares to baseline-J 9/10 result on Q*. ~16 calls per cell.
- **v5** (probability output) on the baseline pool. ~16 calls per
  cell.

These are diagnostic: do the v_judge and v5 effects observed on Q*
replicate on a different question?

## Question selection bias-leak

Critical: the question-selection step (running baseline §3 to
identify contested questions) reveals which questions are contested
in the *baseline pool*. That selection is itself a form of
conditioning on baseline pool's behavior — it might bias us toward
questions where baseline pool happens to be representative.

Mitigation: also include 1 randomly-selected (non-contested-filtered)
question as control. If baseline-pool unanimity at round 0 holds
across all 4 fresh pools too, that's a "non-contested" check; if
fresh pools disagree, we've discovered pool-induced contestation.

## Total cost estimate

- Question selection (5 questions × 9 calls): 45
- Multi-pool generalization (2–3 contested questions × 64): 128–192
- v_judge replication (1 question × ~16): 16
- v5 replication (1 question × ~16): 16
- Plus opus grader at end (1 call)

**Total: ~205–270 sub-agent calls.**

## Analysis

For each contested question, compute the same per-pool modal table
as in `findings_multipool.md` and compare distributions across
questions:

| Question | Pool 1 modal | Pool 2 modal | Pool 3 modal | Pool 4 modal | Pool 5 (baseline) | Direction count (gold vs not) |
|---|---|---|---|---|---|---|
| Q* (already known) | β | β | β | β | β | 4β / 1α; gold = α |
| Q1 | ... | ... | ... | ... | ... | ... |
| Q2 | ... | ... | ... | ... | ... | ... |

If most contested questions show roughly balanced pool directions
(~3 toward gold / ~2 against, or random), then Q*'s 5/6 wrong-
direction pattern is a tail event. The methodology is OK at the
typical-case level but vulnerable to specific question types.

If most contested questions show ≥3 wrong-direction pools, then
**pool asymmetry is general** and the methodology paper's headline
finding generalizes.

## Why N=4 only

Multi-pool experiment showed that pool variance is observable
across all N values, but N=4 has the smallest persona count =
fastest to dispatch and most sensitive to per-persona framework
assignment. Cheaper to test 2–3 questions at N=4 only than to
test 1 question at all 5 N values.

If a question shows ambiguous results at N=4, optionally extend to
N=7 (the case where Q* showed dramatic pool flip — most likely
to surface variance).

## Bias-leak hygiene

- Use `csv_to_questions.py` with `--update-ledger` so the next
  iteration doesn't re-sample these questions.
- Don't read `pilot_questions.jsonl` in main session; use the
  blinded splitter pattern (`PROTOCOL_v2.md` §9.10 / iter-3 §4c).
- Don't include question text in any committed file.
- Per-question modal-distribution tables should use generic
  framework labels (α/β) per question, not topic-specific names.

## Connection to decomposer interventions

This experiment is independent of `decomposer_interventions.md` but
the results inform priority:

- If pool asymmetry is general → decomposer interventions are
  high-priority methodology fixes.
- If pool asymmetry is Q*-specific → decomposer interventions are
  Q*-specific patches; the methodology is OK for typical contested
  questions and only fails on adversarial ones.

Either way, the methodology paper benefits from this data: it tells
us whether the v_judge / multipool / v5 findings generalize or are
question-pathology-specific.
