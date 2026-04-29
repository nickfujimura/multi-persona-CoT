# v5_v9 wager arm — deferred experiment spec

The v5 (probabilistic, no wager) experiment was run; see
`findings_v5.md`. The wager-framing arm (the "v9" component of
v5_v9, per `methodology_experiments.md`) was deferred to manage
session context budget.

## What deferred

The wager-arm prompt template (per `methodology_experiments.md`
v9 spec):

```
You are wagering $100 on each letter at the probabilities you
will report. State only the probabilities and final answer
that reflect a real wager you'd make.
```

This is appended to the v5 prompt template. Otherwise identical.

## Why it matters (after seeing v5 results)

The v5 findings showed two relevant patterns:

1. **Asymmetric per-persona confidence by direction** — when a
   persona's modal is framework α, p_α typically lands 0.55–0.78
   (moderate). When modal is framework β, p_β typically 0.70–0.88
   (higher). Asking for explicit probabilities forces personas to
   articulate this asymmetry.

2. **Probability output sometimes flips letter-modal verdicts**
   in non-uniform directions, depending on which framework the
   persona starts in.

The wager framing's hypothesized mechanism: forcing personas to
treat their probabilities as wagers they'd actually make may shrink
the asymmetric-confidence pattern (toward more conservative
distributions across the board) or amplify it (toward more
extreme/peaked distributions, since "wager" implies commitment).
Either outcome is informative; v5 alone can't distinguish.

## Recommended scope (smaller than original DESIGN.md plan)

The original v5_v9 DESIGN.md spec'd full 2-pool × 2-condition × all
N × all rounds × tiered k = ~300 calls. Given v5 results showed
modest effects, recommend smaller scope:

- **1 pool only** (Pool A reuses v_judge fresh decomposers): direct
  comparison to v5 Pool A baseline.
- **N=5 and N=7 only** — the cells where v5 showed prompt-format
  shifts. The hypothesis is most testable where v5 already produces
  an effect.
- **k=1 per persona, round 0**.
- Total: (5 + 7) × 1 = 12 wager-arm calls. Cheap.

If wager arm shows meaningful difference from v5 at these cells:
extend to all 5 N values for full picture. ~30 more calls.

If wager arm shows no meaningful difference from v5: deprioritize
v9 in the variant zoo. The 12-call diagnostic settles it.

## Comparison metric

For each cell, compare:

| Metric | v5 (no wager) value | v5_v9 (wager) value | Δ |
|---|---|---|---|
| Modal letter | from `findings_v5.md` | new | match? |
| Mean p_α across personas | computed | computed | shift |
| Mean p_β across personas | computed | computed | shift |
| Per-persona confidence asymmetry (A-modal personas vs B-modal personas) | observed in v5 | recomputed | reduced? amplified? |
| Log-pool verdict | from `findings_v5.md` | computed | match? |

If wager framing **reduces** the asymmetry: it's a useful
calibration intervention. If it **amplifies** asymmetry: framework
β personas commit harder to their wrong answer; v9 is harmful.
If no significant change: v9 is neutral and skippable.

## Bias-leak audit on the wager prompt

The wager sentence specifies a **content-neutral behavioral frame**
(monetary commitment), not any specific letter or framework. No
hypothesis-encoding leak risk per `PROTOCOL_v2.md` §3 bias-leak
warning.

The framing is from the superforecasting literature (canonical
calibration intervention) and is not specific to this methodology
or this question. Low-risk to include verbatim.

## Files needed at run time

- `eval/transcripts/experiment_v5_v9/poolA_v5v9_n{5,7}_round0_p{i}.txt`
  (12 new transcripts)
- Reuse Pool A persona descriptions from
  `eval/transcripts/experiment_v_judge/fresh_n{5,7}_decomposer.txt`
- Reuse v5 Pool A r0 transcripts from
  `eval/transcripts/experiment_v5_v9/poolA_v5_n{5,7}_round0_p{i}.txt`
  for direct comparison.

## Why this should be lower-priority than decomposer interventions

`findings_multipool.md` showed the bottleneck on Q* is at the
decomposer layer, not at the persona-output-format layer. v9 is
still in the persona-output layer. Even if v9 reduces the
asymmetric-confidence pattern, that probably doesn't recover gold
on Q* — the persona pool is biased toward the wrong framework
regardless of how its individual personas express probability.

Run v9 if you want a complete picture of v5_v9 (the variant zoo
slot is open and the experiment is cheap). Skip if you only have
budget for one more experiment — pick a decomposer intervention
instead.
