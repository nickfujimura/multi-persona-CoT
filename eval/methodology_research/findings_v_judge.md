# v_judge — findings

Variant test of the v_judge methodology (`methodology_experiments.md`):
**replace count-aggregation with a separate judge sub-agent that reads
all persona transcripts and issues a verdict.** Run on Q* (the
iteration-2 contested case).

## Headline finding

**Aggregation rule choice (count vs reasoning-based judge) is a small
effect on Q*. Pool variance dominates.** The judge sometimes deviates
from majority count, and the deviation is occasionally toward gold,
but the effect size is small relative to the cross-pool variance later
characterized in `findings_multipool.md`.

## Experimental design

Two prompt versions:
- **Prompt M**: minimal "render verdict" instruction. No explicit
  guidance about counting or substance evaluation.
- **Prompt J**: M plus the standard judgment instruction *"evaluate
  the substance of each persona's reasoning — do not let majority
  count alone determine your verdict."*

Decision to run both M and J was driven by user feedback: an initial
draft removed the "don't decide by count alone" line on bias-leak
grounds. The user pointed out that "tyranny of the majority" / its
avoidance is a **standard general instruction in judgment contexts**,
not a hypothesis-specific leak. M-vs-J became a 2-way prompt
sensitivity test rather than the bias-clean choice. See
`HANDOFF.md` for the bias-leak distinction this clarified.

Two designs:
- **Design A — judge-as-aggregator**: judge runs at round 0 and round
  1 separately on each (pool, N).
- **Design B — judge-as-deadlock-breaker**: judge reads the full
  debate transcript (rounds 0–3) on N values that didn't reach
  unanimity by round 1.

Two arms:
- **Baseline** (Pool A): judge over canonical baseline persona
  transcripts.
- **Fresh** (Pool B): judge over a fresh, independently-decomposed
  persona pool.

Sweep: N ∈ {4, 5, 6, 7, 8}, k=4 trials per cell.

Total dispatches: ~390 sub-agent calls.

## Headline accuracy by condition (modal verdict matching gold)

| Condition | Cells matching gold / total | |
|---|---|---|
| Baseline aggregator M | 7/10 | |
| Baseline aggregator J | **9/10** | best aggregator condition |
| Baseline deadlock M | 3/3 | |
| Baseline deadlock J | 2/3 | |
| Baseline total | **21/26 (81%)** | |
| Fresh aggregator M | 6/10 | |
| Fresh aggregator J | 5/10 | |
| Fresh deadlock M | 2/3 | |
| Fresh deadlock J | 2/3 | |
| Fresh total | **15/26 (58%)** | |

The arm difference (baseline vs fresh = 81% vs 58%) is much larger
than the within-arm prompt difference (M vs J ≈ ±2 cells). The
arm-difference effect is exactly what `findings_multipool.md`
investigates further; it is structural, not noise.

## Within-pool prompt sensitivity (M vs J)

Modal-verdict differences between M and J prompts on the same Pool A
transcripts were observed at exactly **one cell out of 10**. At that
cell, the underlying personas marginally favored framework β (which
gives the wrong answer); the M judges followed that majority, while
the J judges deviated against it toward framework α (which gives gold).

The "evaluate substance / don't decide by count alone" instruction
in J does meaningful work — but its value depends on a specific
persona-distribution configuration that can't be predicted in
advance. On 9 of 10 baseline cells, M and J converge.

## Notable behaviors

1. **Judges occasionally pick a letter that no persona voted for**
   (2 cells out of 208 trials). Existence proof that judges reason
   independently, not just adjudicate between persona votes. Rare,
   but real.

2. **Judges follow confidently-wrong personas.** On cells where the
   persona pool confidently converges on the wrong framework
   (framework β unanimity in the fresh arm), judges (both M and J)
   follow them. The judge is not a defense against wrong-frame
   persona convergence.

3. **Deadlock-judging behaves differently than round-by-round
   judging at the same N.** At one N value, the J judge at round 0
   (snapshot) was unanimous for one letter, while the J judge over
   full debate transcripts (deadlock condition) split. Reading the
   full debate gives weight to persistent dissenting voices that
   a round snapshot ignores.

## Bias-leak audit

The judge prompts as committed (see `eval/transcripts/experiment_v_judge/PROMPT_DESIGN.md`):
- ❌ no enumeration of evaluation criteria
- ❌ no "watch out for X" guidance
- ❌ no specific failure modes named
- ✅ minimal role framing ("evaluating a debate")
- ✅ standard judgment meta-instruction in J only
  ("don't decide by count alone")
- ✅ §9.2 self-save pattern, no-tools restriction

Opus auditor (post-hoc) confirmed clean run: substantive reasoning
across both prompts; no tool-misuse signals; pool divergence at the
high-variance N value driven by persona-role differences across
pools, not by personas making different conclusions on a shared
frame.

## Why v_judge doesn't solve Q*

The methodology fails because:

1. The persona pool sometimes anchors on the wrong reasoning frame.
2. Once a wrong frame dominates persona round-0 outputs, debate
   amplifies rather than corrects (a single contested N value's
   round-0 was 4 for framework α / 2 for β / 1 for an irrelevant
   letter; round-1 went to unanimous α, *amplifying away from gold*).
3. Judges, being readers of the debate, inherit the dominant frame
   regardless of whether it's correct.

Aggregation choice doesn't escape that failure mode.

## What v_judge does tell us

- Aggregation rule design space is real but small.
- "Don't decide by count alone" is a meaningful instruction
  occasionally — but contingent on specific persona-distribution
  configurations.
- The "judge picks letters no persona voted for" behavior shows the
  judge can in principle outperform count-aggregation. Whether this
  is reliably useful is unclear from this experiment alone.

## Limitations

- N=1 question. Cross-question generalization is the obvious next
  step; see `cross_question_generalization.md`.
- One persona-pool sample per arm. The "baseline >> fresh" effect
  could partly be unlucky fresh draw — see `findings_multipool.md`
  for the multi-pool follow-up.
- Pre-experiment inference about gold was wrong (predicted one letter,
  actual was the other). Future sessions inheriting this work should
  verify gold via blinded grader rather than infer it from secondary
  documentation.

## Files

- Raw transcripts (gitignored): `eval/transcripts/experiment_v_judge/`
- Full unredacted findings (gitignored):
  `eval/transcripts/experiment_v_judge/findings.md`
- Prompt design (gitignored):
  `eval/transcripts/experiment_v_judge/PROMPT_DESIGN.md`
