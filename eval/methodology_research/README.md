# Methodology research — overview

This directory contains methodology-level research on the
multi-persona-CoT debate protocol (`PROTOCOL_v2.md`). Distinct from
`summary.md` (per-iteration accuracy results) — this is structural
research on the methodology itself.

All research below was conducted on a single contested question
(referred to as Q* — the iteration-2 contested case, gpqa_007).
Cross-question generalization is one of the explicitly-deferred next
steps; see `cross_question_generalization.md`.

## Reading order

1. `README.md` (this file) — orientation
2. `findings_v_judge.md` — variant test of judge-as-aggregator
3. `findings_multipool.md` — multi-pool ensemble test (the most
   load-bearing finding)
4. `findings_v5.md` — variant test of probabilistic persona output
5. `HANDOFF.md` — what the next session should do, in priority order
6. `wager_arm_spec.md` — deferred v5_v9 wager-framing arm
7. `decomposer_interventions.md` — designs for decomposer-level
   experiments (the most-promising next direction)
8. `cross_question_generalization.md` — testing the "pool draws
   asymmetric around gold" finding across multiple questions

## Headline finding (across all three experiments)

**Persona-pool variance is a first-order effect on contested
questions.** Aggregation rule (v_judge), persona output format (v5),
debate-round count, judge prompt sensitivity, deadlock-only synthesis
— all secondary. The §3 orthogonal decomposer's "average draw" on Q*
is biased toward one of the two reasoning frameworks (the one that
produces the wrong answer), and ensembling across 6 independent
pool draws does not recover gold.

This reframes the priority of next experiments toward
**decomposer-level interventions** rather than aggregation-rule
tweaks. The methodology bottleneck is upstream of the aggregation
step, not at it.

## What is referred to here as "Q*"

Q* is the iteration-2 contested case (gpqa_007 in
`eval/pilot_questions.jsonl`, gitignored). All three experiments use
this single question. Per `PROTOCOL_v2.md` §9.7, question text and
choice text are NOT reproduced in any committed document. Persona-
role names are abstracted to generic descriptors. Numerical
quantities specific to the question are abstracted to "the stated
quantity."

The two reasoning frameworks for Q* are abstracted as
**framework α** (the gold-aligned framework) and **framework β**
(the framework that produces a wrong answer). Personas drawn from
the §3 orthogonal decomposer for Q* divide imperfectly between
these two frameworks.

## Status

| Experiment | Status | Sub-agent calls (approx) |
|---|---|---|
| v_judge | done | ~390 |
| multipool | done | ~140 |
| v5 (no wager) | done | ~65 |
| v5_v9 wager arm | deferred | ~60 estimated |
| Cross-question generalization | not started | ~200–500 estimated |
| Decomposer-level interventions | not started | ~150–400 estimated per variant |

Raw experiment data lives in `eval/transcripts/experiment_*` (each
gitignored). The findings here are leak-clean summaries.
