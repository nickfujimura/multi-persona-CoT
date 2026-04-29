# Methodology research — overview

This directory contains methodology-level research on the
multi-persona-CoT debate protocol. It is **distinct from** the
per-iteration accuracy results in `eval/summary.md`. Iterations run
the methodology on a fresh 10-question sample and append numbers;
methodology research designs targeted experiments to understand
*why* the methodology behaves the way it does and which levers
matter.

**If you're a new Claude Code session arriving here**: read
`HANDOFF.md` first — it's the entry point with a project-orientation
section, distinguishes this thread from the iteration thread, and
lays out priority order for next experiments. This README is just a
map across the findings docs.

## Project context (one paragraph)

The repo tests whether multi-persona CoT debate (decomposer →
parallel persona answers → multi-round debate → aggregated verdict)
beats plain zero-shot CoT on GPQA Diamond (graduate-level science
MCQ). The eval substrate is Sonnet 4.6, which scores ~90% on this
set zero-shot — so the methodology has at most 10pp of headroom,
concentrated on contested questions. The architecture is in
`eval/PROTOCOL_v2.md`.

## Reading order

For new sessions, read in this order:

1. `HANDOFF.md` — entry point: project orientation + priority order
   + operational essentials.
2. `eval/PROTOCOL_v2.md` (NOT in this directory) — the methodology
   architecture itself. §1, §3, §9 are mandatory.
3. `uncertainty_gated_evaluation.md` — the highest-priority next
   experiment spec. Read this before designing anything new.
4. `findings_decomposer_audit.md` — most recent findings. Reframes
   the multipool interpretation; read this before reading multipool.
5. `findings_multipool.md` — second-most-recent before the audit
   reframed it. Most load-bearing of the older findings.
6. `findings_v_judge.md`, `findings_v5.md` — older single-question
   findings.
7. `decomposer_interventions.md` — interventions, of which #1 is
   done; #2–#4 are downgraded.
8. `cross_question_generalization.md` — superseded by
   `uncertainty_gated_evaluation.md`; skim only.
9. `wager_arm_spec.md` — deferred indefinitely.

## Current state in one paragraph

Four single-question experiments have run on Q* (the iter-2
contested case, gpqa_007): v_judge, multipool, v5, and the
decomposer audit. The audit's key finding overturned the prior
narrative: **roster-level framework coverage on Q* is uniformly
high** (every framework appears in every pool), so the bottleneck
is *within-framework* reasoning-path heterogeneity, not roster
generation. Aggregation interventions (v_judge, v5) produce small
mixed-direction effects. The next experiment must test whether
these patterns generalize to other questions, and at the 90%
baseline that requires a population-scale design with proper
test/holdout discipline — which is what
`uncertainty_gated_evaluation.md` specs.

## Headline finding (across four experiments)

**Pool-level vote variance is real, but its source is NOT
roster-level framework absence.** From the audit:

- All 5 plausible reasoning frameworks for Q* are present in all 7
  pools. Coverage is uniformly high.
- The decomposer is highly stereotyped (same archetypal lineup in
  every pool).
- Pool-to-pool vote variance is driven by *within-framework*
  reasoning-path heterogeneity — specifically within F1 (the
  numerically dominant framework, 28% of personas), which
  empirically splits into a gold-voting subpath and a non-gold-voting
  subpath without any role-level signal distinguishing them.
- Generic "Other" personas (declined to classify) vote gold 85% of
  the time — higher than any specific framework. Specific framework
  commitments may pull personas *away* from the model's gold-aligned
  base rate on this question.

Decomposer-level interventions (brainstorm-50, framework-spotter,
frame-explicit) all target roster-level coverage, which the audit
shows is not the bottleneck. The next-experiment priority shifts to
**uncertainty-gated population-scale evaluation** — see
`uncertainty_gated_evaluation.md` and `HANDOFF.md`.

Aggregation rule (v_judge), persona output format (v5),
debate-round count, judge prompt sensitivity remain secondary
effects.

## What is referred to here as "Q*"

Q* is the iteration-2 contested case (`gpqa_007` in
`eval/pilot_questions.jsonl`, gitignored). The four
single-question experiments all used this single question. Per
`PROTOCOL_v2 §9.7`, question text and choice text are NOT
reproduced in any committed document; persona role names are
abstracted to generic descriptors; numerical quantities are
abstracted to "the stated quantity."

The audit's framework taxonomy uses neutral labels F1–F5 (defined
in `findings_decomposer_audit.md`). Older docs use a binary
α/β framework labeling — those labels were a simplification of
what is actually a 5-framework structure; the F1–F5 taxonomy
supersedes them.

## Status

| Experiment | Status | Sub-agent calls (approx) |
|---|---|---|
| v_judge | ✅ done | ~390 |
| multipool | ✅ done | ~140 |
| v5 (no wager) | ✅ done | ~65 |
| Decomposer orthogonality audit (intervention #1) | ✅ done | 2 |
| **Uncertainty-gated evaluation** | **proposed (highest priority)** | **~1500 estimated** |
| Audit refinement (F1 sub-classification) | proposed (cheap diagnostic) | 1–2 |
| Persona-execution intervention battery | proposed (after uncertainty-gated) | ~65 per variant |
| v5_v9 wager arm | deferred | ~60 estimated |
| Cross-question generalization | superseded by uncertainty-gated | — |
| Decomposer-level interventions #2–#4 | priority downgraded by audit | ~150–400 per variant |

Raw experiment data lives in `eval/transcripts/experiment_*` (each
gitignored). The findings docs here are leak-clean summaries.
