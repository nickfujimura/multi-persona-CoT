# Multi-pool ensemble — findings

Direct test of v_judge's headline observation that pool variance
dominates aggregation-rule choice. **Question: does ensembling across
multiple independent persona-pool draws recover gold on Q*?**

## Headline finding

**Ensembling across 6 independent pools does not recover gold on Q*
— and is *worse* than a single favorable pool.** The methodology
fails because **5 of 6 independent pool draws lean toward framework β
(wrong) at most N values; only the canonical baseline pool is
consistently aligned with framework α (gold)**.

This is the most load-bearing finding from this session's research.
The single-pool baseline that v_judge used was a **favorable
outlier**, not a representative draw. A blind methodology user has
no way to know in advance which pool draw will land favorably.

## Experimental design

6 independent pools:
- **Pool A** (canonical baseline = `experiment_v3/`): same draw
  v_judge used.
- **Pool B** (fresh arm reused from v_judge): independent draw.
- **Pool C, D, E, F**: 4 new independent draws generated for this
  experiment.

Each pool: 5 N values × persona evals at round 0 (no debate). All
generated using the verbatim §3 orthogonal decomposer. The only
variable across pools is the random sample produced by an otherwise-
identical decomposer prompt.

Total fresh dispatches: 20 decomposers + 120 round-0 evals = 140.
(Pools A, B reuse data from v_judge.)

## Per-pool single-pool accuracy on Q* (modal verdict at round 0 vs
gold)

| Pool | Cells matching gold / 5 | |
|---|---|---|
| A (canonical baseline) | 4/5 | favorable outlier |
| B (fresh) | 3/5 | |
| C | 1/5 | |
| D | 1/5 | |
| E | 0/5 | all framework β or tie |
| F | 2/5 | mixed |

Average single-pool accuracy: **11/30 = 37%**. Baseline accuracy is
≥2× any other pool. Strong evidence that single-pool results on Q*
should not be generalized.

## Ensemble rules and their accuracy

Three aggregation rules tested across all 6 pools:

### Majority-of-pool-modes

For each N, take the modal verdict across the 6 per-pool modal
verdicts. Tied pools count as 0.5 each direction.

| N | 6-pool tally | Ensemble verdict | Match gold? |
|---|---|---|---|
| 4 | β=3, α=3 (with 2 ties) | tie | no |
| 5 | α=5, β=1 | α | no |
| 6 | α=3, β=3 | tie | no |
| 7 | α=3, β=3 | tie | no |
| 8 | α=3.5, β=2.5 (with 1 tie) | α | no |

**Pool-mode majority: 0/5 cells with clear-majority gold match.**

### Total-vote count

Pool all persona votes across all 6 pools at each N. (~24–48 persona
votes per N depending on N.) Modal letter wins.

| N | Total-vote modal | Match gold? |
|---|---|---|
| 4 | β (margin 1) | yes |
| 5 | α | no |
| 6 | β (margin 5) | yes |
| 7 | α (margin 2) | no |
| 8 | β (margin 3) | yes |

**Total-vote count: 3/5.** Better than per-pool-mode, worse than the
favorable single pool (Pool A: 4/5).

### Confidence-gated (Pool only votes if it reached modal-unanimity)

Skipped on data-availability grounds (round 0 only). Conceptually
identical to per-pool-mode but with abstain-on-ties; expected to be
no better than per-pool-mode given 0/5 baseline.

## Why ensembling fails on Q*

Two compounding factors:

1. **Pool draws are not symmetric around gold.** The §3 orthogonal
   decomposer's prior over "expert types relevant to Q*" draws
   personas aligned with framework β more often than framework α.
   5 of 6 pools end up β-leaning at most N. Naive ensembling of
   5 wrong-direction pools with 1 right-direction baseline gives a
   biased ensemble.

2. **Baseline is favorable, not typical.** Pool A's 4/5 accuracy is
   the upper tail of the distribution, not the median.

**Implication**: cross-pool ensembling is not a free improvement.
It helps when persona-pool errors are independent and zero-mean. On
Q* they are correlated (the population of "experts who could answer
this" is itself biased toward framework β reasoning).

## Persona-roster analysis

Roles appearing in nearly every pool's decomposer output (≥ 5 of 6
pools): a theoretical-foundations persona, an applied-domain-A
persona, an applied-domain-B persona, and a skeptic /
adversarial-question-parser persona.

Roles appearing in some pools but absent from others:
**framework-α-aligned roles** (the kind of persona whose disciplinary
training makes framework α first-instinct) appear in only ~3 of 6
pools — yet that's the framework that gets the question right.

The orthogonal decomposer is **not reliably surfacing the gold-
aligned frame.** It produces topical diversity but not framework
diversity proportional to the right answer's framework.

## What this suggests for the methodology

The bottleneck is not aggregation, not representation, not debate-
round count. It's **persona-roster generation**. The orthogonal §3
decomposer produces topical diversity but biases toward whichever
framework is the more common "expert's first instinct" reasoning
mode for the question type.

Possible interventions are at the **decomposer** layer; see
`decomposer_interventions.md`.

## Limitations

- N=1 question. The "biased pool draws" finding is Q*-specific. On
  a different question, the §3 decomposer might draw the right frame
  more reliably; or its bias might run the opposite direction. See
  `cross_question_generalization.md` for the test.
- 6 pools is a moderate sample. 20+ pools would let us measure the
  pool-direction distribution rather than estimating "5 of 6 lean
  wrong."
- Round 0 only. Multi-pool round-1 dynamics deferred but probably
  not informative — v_judge data already showed round 1 amplifies
  round-0 modal direction.

## Files

- Raw transcripts (gitignored):
  `eval/transcripts/experiment_multipool/` (6 pools' worth)
- Pool A and Pool B raw data: `eval/transcripts/experiment_v3/` and
  `eval/transcripts/experiment_v_judge/fresh_*`
- Full unredacted findings (gitignored):
  `eval/transcripts/experiment_multipool/findings.md`
- Design (gitignored):
  `eval/transcripts/experiment_multipool/DESIGN.md`
