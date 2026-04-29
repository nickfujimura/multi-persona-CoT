# Uncertainty-gated evaluation — design

This spec supersedes the prior iteration cadence ("run methodology on
10 questions, see what happens") and supersedes
`cross_question_generalization.md` as the highest-priority next
experiment. Its purpose is to test the multi-persona-CoT methodology
at population scale on the subset of questions where it could
plausibly help, with proper test/holdout discipline.

## Why this design

**The 90% baseline problem.** Sonnet 4.6 (the eval substrate model)
achieves ~90% accuracy on GPQA Diamond zero-shot. The methodology has
at most 10 percentage points of headroom, concentrated on the hardest
questions. Iter-2 already showed A=10/10 vs B=9/10 on a 10-question
sample — multi-persona LOST one question (Q*) versus zero-shot. That
result is consistent with either "methodology is net-negative at this
baseline" or "10 questions is too noisy to tell," and we cannot
distinguish from the data we have.

**The iter-N pattern is structurally underpowered.** At a 90% baseline,
detecting a 5pp methodology gain at p<0.05 requires ~200 questions.
Running 10 questions per iteration cannot demonstrate population-level
methodology effects. Per-question debug findings (like the within-F1
heterogeneity in `findings_decomposer_audit.md`) are mechanism-level
useful but cannot establish methodology-level claims.

**Methodology levers we have evidence for.** Three levers have shown
signal on Q* and need population-scale testing:
- Aggregation rule (v_judge J prompt): flipped 1 of 10 cells toward
  gold on Pool A, but `findings_v_judge.md` notes it depends on
  specific persona-distribution configuration.
- Orthogonal decomposer (`PROTOCOL_v2 §3` / §9.11): produced a
  measurable round-0 distribution shift on Q*. Mechanism reframed by
  the audit (it's not "surfacing missing frameworks") but the shift
  is real.
- Persona-execution interventions (proposed by
  `findings_decomposer_audit.md`): not yet tested. The within-F1
  heterogeneity finding suggests this layer is where remaining
  variance lives.

**Uncertainty gating.** A 90% baseline implies most questions are
"already correct, leave them alone." The methodology only has anything
to do on questions where the model is genuinely uncertain. Filtering to
those questions (a) sharpens signal-to-noise for methodology evaluation
and (b) potentially produces a deployable gating rule ("invoke
multi-persona only when zero-shot k=5 disagrees with itself") rather
than a blanket replacement claim.

## Headline experiment structure (4 phases)

1. **Ambiguity mapping**: zero-shot CoT k=5 on the unused GPQA Diamond
   pool (~150 of 178 remaining questions). Categorize each question by
   the consistency of zero-shot responses.
2. **Test/holdout split**: stratified random split of the
   modal-disagreement subset into dev (60%) and holdout (40%).
3. **Sequential lever testing on dev**: test three methodology levers
   as factors with a "keep the winner" rule.
4. **Holdout validation**: run the winning configuration on the
   sealed holdout set; compare to zero-shot k=5 modal verdict on the
   same questions.

## Phase 1 — Ambiguity mapping

### Sample

Use `eval/csv_to_questions.py --csv eval/GPQA/gpqa_diamond.csv
--n 150 --seed <new> --update-ledger` to sample 150 questions not yet
used in iter-1, iter-2, or methodology-research sessions.

(Pool size verification: 198 GPQA Diamond rows total, ~20 used in
prior iterations + ~5 ledger-buffer = ~178 available. Sampling 150
leaves a small buffer for re-runs. The dataset will be substantially
exhausted after Phase 1 — note this for future iterations.)

### Per-question dispatch

For each question, dispatch **5 independent zero-shot CoT sub-agents**.

Use the round-0 prompt from `PROTOCOL_v2 §3` *without* persona
framing — i.e., the Condition A prompt. Apply the §9.2 self-save
pattern: each sub-agent saves full reasoning to disk, returns only
`Answer: <letter>` to main.

Save transcripts to:
`eval/transcripts/experiment_uncertainty_gated/phase1_zero_shot/{id}_zs_t{1..5}.txt`

Total Phase-1 zero-shot calls: **150 × 5 = 750**.

### Categorization (gold-aware grader)

Dispatch one opus grader sub-agent (Read+Write only). It:
- Reads `eval/pilot_questions.jsonl` for gold answers (gold-aware).
- Reads each question's 5 zero-shot transcripts to extract letters.
- Categorizes each question:
  - **Modal-disagreement** (uncertain): ≥2 different letters across
    the 5 votes.
    - Sub-divide: weak (4-1 split), medium (3-2 / 3-1-1), strong
      (2-2-1 / 2-1-1-1 / etc.).
  - **Wrong-confident**: 5/5 same letter, but wrong.
  - **Right-confident**: 5/5 same letter, correct.
- Writes per-question categorization to
  `eval/transcripts/experiment_uncertainty_gated/phase1_classification.md`
  (gitignored — gold-aware).
- Returns to main session: only count summaries (e.g., "120
  right-confident, 18 wrong-confident, 12 modal-disagreement of which
  6 weak, 4 medium, 2 strong"). Does NOT echo per-question
  categorization or gold letters.

This is the §9.5 grader pattern: main session never reads gold.

### Bias-leak hygiene (apply throughout all phases)

- **Implement the §9.10 question splitter pattern** at the start of
  Phase 1. Dispatch a Read+Write splitter sub-agent that reads
  `pilot_questions.jsonl` and writes per-question files like
  `eval/transcripts/experiment_uncertainty_gated/questions/{id}_question.txt`
  containing the question + choices but NOT the gold answer. Main
  session and downstream planners reference these per-question files
  by path; they NEVER read `pilot_questions.jsonl` directly.

  (Iter-2 and the audit session both leaked question text into main
  context by reading `pilot_questions_blinded.jsonl` directly. Iter-3
  proposed the splitter pattern; this experiment finally implements it.)

- Main session sees question IDs only after splitting; question text
  and choices live only in the per-id files and in transcript outputs
  (never echoed back to main).

## Phase 2 — Test/holdout split

### Selection criterion

The dev/holdout pool is the **modal-disagreement subset** from Phase
1. Wrong-confident questions are excluded from the methodology
evaluation (multi-persona inherits the model's wrong belief and is
unlikely to fix it; out of scope for this experiment, may be revisited
later). Right-confident questions are excluded (no headroom).

### Stratification

The split should preserve representation across:
- **Uncertainty subtype** (weak / medium / strong)
- **Domain** (Physics / Chemistry / Biology — read from the JSONL's
  `domain` field per question; this is metadata, not gold)

If the modal-disagreement pool is small (<30 questions), accept
under-representation and document the limitation.

### Operational split

Dispatch one Read+Write sub-agent (gold-blind):
- Reads `phase1_classification.md` (this requires the splitter
  approach — categorization output is gold-aware so the split
  sub-agent should see only ID + uncertainty-subtype + domain, not
  per-question gold). To keep gold off the split sub-agent, the
  grader in Phase 1 should write a *redacted* split-input file
  containing only `{id, subtype, domain}` per question.
- Performs stratified random split with deterministic seed (e.g.,
  seed=99) into 60% dev / 40% holdout.
- Writes
  `eval/transcripts/experiment_uncertainty_gated/phase2_split.json`
  containing only question IDs grouped by dev/holdout. **This file
  is committable** (IDs only, no gold, no question text).

Expected dev set size: ~15-30 questions. Holdout: ~10-20 questions.
Small but actionable.

## Phase 3 — Sequential lever testing on dev

The methodology has multiple independent levers. Cross-factorial
testing (2³ = 8 conditions) blows up budget. Instead: sequential
"keep the winner" with a documented limitation (lever interactions
may be missed; flag in the writeup).

### Lever A — Decomposer choice

Two arms, both at N=4 (cheapest), round 0 only:
- **A1**: orthogonal decomposer (current `PROTOCOL_v2 §3` / §9.11).
- **A2**: flat decomposer — the older "make perspectives genuinely
  divergent" prompt (recoverable from git history pre-commit
  `b4115fd` / `3717c5f`). Use the §3-pre-orthogonal prompt verbatim;
  do NOT rewrite or parameterize, to avoid prompt-engineering
  contamination.

Per dev question per arm: 1 decomposer + 4 round-0 evals = 5 calls.

Apply §9.2 self-save and §9.3 planner-Read patterns.

After all dev questions × both arms run, dispatch a gold-aware grader
to compute per-arm modal-verdict accuracy on the dev set. Returns to
main: "A1 = X/n, A2 = Y/n; winner = A?, effect-size = (X-Y)/n,
seed-confidence interval".

**Lever A budget**: dev × 5 calls × 2 arms ≈ 200-300 calls.

### Lever B — Aggregation rule

Two arms, applied to the **same round-0 outputs from Lever A's
winner** (no new round-0 calls):
- **B1**: modal count aggregation.
- **B2**: judge-J aggregation (§3 v_judge prompt, J variant —
  "evaluate substance, don't decide by count alone").

Per dev question, fire one judge-J call per question (k=1 trial,
since v_judge showed judges are stable at k=1 on contested cases;
k=2 if budget permits for variance estimation).

After Lever B runs, gold-aware grader compares per-arm accuracy on
dev. Returns winner + effect size.

**Lever B budget**: dev × 1 judge call ≈ 15-30 calls.

### Lever C — Persona-execution intervention

This is the new direction motivated by
`findings_decomposer_audit.md`. Two arms, applied to the **same
decomposer output as Lever A's winner** (new round-0 calls with
modified prompt):
- **C1**: standard round-0 prompt (control).
- **C2**: round-0 prompt with v7-style pre-mortem appended:
  *"After your reasoning and final answer, also report:
  WOULD-CHANGE-MY-MIND: 1-3 specific conditions that, if shown to be
  true, would make you change your answer."*

  Use the spec from `methodology_experiments.md` v7 verbatim. Do
  **NOT** modify the v7 prompt to be more elaborate or to enumerate
  failure types — see `PROTOCOL_v2 §3` bias-leak warning. Round-0
  only at this stage; v7's round-1 dynamics deferred.

Aggregate using Lever B's winner. Compare per-arm dev accuracy.

**Lever C budget**: dev × 4 round-0 × 2 arms + dev × 1 judge ≈
130-260 calls.

### Sequential ordering caveat

Sequential lever testing assumes lever effects are additive and
direction-independent. They may not be. **After Phase 3, before
Phase 4, also run the "control configuration" (A1+B1+C1, all
defaults) on dev** as a sanity check. If the winning A+B+C combo is
not strictly better than the control on dev, do NOT validate on
holdout — the sequential procedure has gone wrong and needs
iteration.

**Phase 3 total budget**: ~400-600 calls.

## Phase 4 — Holdout validation

### What to run on holdout

- **Multi-persona winning configuration** from Phase 3
  (winning A + winning B + winning C). One pool draw per question,
  N=4, round 0.
- **Zero-shot k=5 baseline** on holdout (already collected in Phase 1
  — just reference those transcripts).

### What to compare

Per holdout question:
- Methodology modal verdict (after winning aggregation).
- Zero-shot modal verdict (most-common letter across 5 zero-shot
  trials).
- Gold.

Compute:
- Methodology accuracy on holdout.
- Zero-shot k=5 modal accuracy on holdout.
- Effect size (methodology − zero-shot).
- Wilson confidence interval at the small-sample level.
- Stratified by uncertainty subtype + domain.

### What invalidates the result

- Methodology accuracy on holdout is materially lower than on dev:
  overfitting; report negatively.
- Methodology accuracy is no different from zero-shot on holdout:
  null result; report.
- Methodology accuracy is meaningfully higher than zero-shot:
  positive result, with effect-size CI.
- Domain-stratified analysis shows large interaction (helps
  Chemistry, hurts Physics, or vice versa): publishable as a
  conditional finding, not a general one.

**Phase 4 budget**: holdout × 5 calls (decomposer + 4 round-0) +
holdout × 1 judge ≈ 60-120 calls.

## Total budget

| Phase | Calls | Notes |
|---|---|---|
| 1: Ambiguity mapping | 750 (k=5 zero-shot × 150) + ~5 grader/splitter overhead | Bulk of budget |
| 2: Test/holdout split | 1 split sub-agent | Trivial |
| 3: Lever testing on dev | ~400-600 | Bulk of methodology work |
| 4: Holdout validation | ~60-120 | Smallest |
| **Total** | **~1200-1500 sub-agent calls** | |

Plus opus grader/auditor calls at each phase boundary (~10 calls).

This is tractable for one session. Pre-approve `Agent` permission in
`/permissions` before starting.

## Connection to prior findings (and how they inform this experiment)

- **`findings_v_judge.md`**: Lever B tests whether the J-prompt
  effect generalizes from Q* to a population sample.
- **`findings_multipool.md`**: pool variance is a known phenomenon
  on contested questions. This experiment uses single-pool draws per
  question to control budget. Multi-pool ensembling can be added as a
  Phase-5 robustness check if Phase 4 shows positive methodology
  signal.
- **`findings_v5.md`**: probability-output prompt is orthogonal to
  the levers tested here. Can be added as Lever D in a follow-up.
- **`findings_decomposer_audit.md`**: within-F1 heterogeneity is a
  *diagnostic* in this experiment, not a lever. After Phase 3, run a
  miniature audit on Phase-3 dev decomposer outputs and check whether
  per-question within-F1 heterogeneity correlates with methodology
  success (high heterogeneity → less reliable methodology). This
  would inform a future per-question gating rule beyond
  zero-shot-uncertainty.

## Stratification analyses (do these with the data, not as
separate experiments)

After Phase 4:
- Methodology vs zero-shot, stratified by:
  - Uncertainty subtype (weak / medium / strong)
  - Domain (Physics / Chemistry / Biology)
- Methodology vs zero-shot on Phase 1's wrong-confident subset (not
  in dev/holdout): does the methodology accidentally fix any
  wrong-confident questions? If yes, the gating rule may be wrong.
  Cheap follow-up; ~80-100 calls if you want to run methodology on
  the wrong-confident pool.
- Within-F1 heterogeneity (audit pattern) on the dev decomposer
  outputs: does it predict per-question methodology accuracy?

## Bias-leak hygiene summary

| Phase | Main sees | Sub-agents see |
|---|---|---|
| Splitter | nothing | full `pilot_questions.jsonl` (Read+Write only) |
| 1 zero-shot | per-question paths | per-question files (no gold) |
| 1 grader | counts only | full gold |
| 2 split | counts + ID list (no gold) | redacted classification (id+subtype+domain only) |
| 3 lever testing | per-arm winner ID + effect size | per-question paths (no gold), planners Read transcripts |
| 3 lever grader | per-arm accuracy + winner | full gold |
| 4 holdout | final summary stats | full gold |

Main session NEVER reads `pilot_questions.jsonl` or any per-question
file containing question text. All question text flows through
sub-agents that have explicit Read access to specific files.

## Open questions / things future sessions should decide

1. **Sample size adequacy**: if the modal-disagreement subset is
   <20 questions, can we still draw conclusions? Probably only large
   effects (≥10pp). Document this when designing Phase 4 reporting.
2. **k for zero-shot**: 5 is a compromise. k=10 doubles Phase-1 cost
   but produces tighter confidence on what counts as
   modal-disagreement. Worth considering if budget allows.
3. **Pool draws per question in Phase 3**: 1 draw per condition is
   cheapest. Multi-pool would add robustness but ~4× the budget.
   Acceptable trade-off; document.
4. **What happens if winning configuration is "control"**: i.e., all
   defaults. That's a finding in itself: the methodology levers don't
   help. Holdout would still confirm whether multi-persona-CoT in
   any form beats zero-shot. Don't skip Phase 4 in that case.

## Files / paths

```
eval/transcripts/experiment_uncertainty_gated/
├── questions/                          (Phase 1 splitter outputs; gitignored)
│   ├── gpqa_XXX_question.txt
│   └── ...
├── phase1_zero_shot/                   (Phase 1 transcripts; gitignored)
│   ├── gpqa_XXX_zs_t1.txt
│   └── ...
├── phase1_classification.md            (Phase 1 grader output; gitignored)
├── phase1_classification_redacted.json (Phase 2 split input; gitignored)
├── phase2_split.json                   (Phase 2 split; COMMITTABLE — IDs only)
├── phase3_lever_A/                     (Phase 3 transcripts; gitignored)
├── phase3_lever_B/
├── phase3_lever_C/
├── phase3_winners.md                   (Phase 3 grader output; gitignored)
├── phase4_holdout/                     (Phase 4 transcripts; gitignored)
└── phase4_summary.md                   (Phase 4 grader output; gitignored)

eval/methodology_research/
└── findings_uncertainty_gated.md       (post-experiment writeup; leak-clean)
```

## Pre-flight checklist for the session that runs this

```bash
git status --short                                  # clean tree
git branch --show-current                           # likely main; branch off
ls eval/transcripts/experiment_*/                   # prior dirs present
test -f eval/pilot_questions.jsonl                  # gold lives here
test -f eval/used_csv_indices.txt                   # ledger
wc -l eval/used_csv_indices.txt                     # confirm ledger has prior indices
# Pre-approve Agent permission in /permissions before starting
```

## Connection to next-session ergonomics

This experiment exhausts most of the unused GPQA Diamond pool. After
running it, future iterations need either:
- A new question source (different benchmark), or
- A re-use policy ("re-test on the same questions to measure
  stability over methodology versions").

Document this trade-off in the post-experiment writeup so the
next-next session knows the dataset is no longer fresh.
