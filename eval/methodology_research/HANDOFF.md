# Handoff to next methodology research session

You are a fresh Claude Code session picking up methodology research
on the multi-persona-CoT debate protocol. Three experiments have
been run on Q* (the iter-2 contested case, gpqa_007). The findings
are in `findings_v_judge.md`, `findings_multipool.md`, and
`findings_v5.md`.

Read those before designing any new experiment.

## What to internalize

1. **Pool-level persona-frame selection is the dominant effect on
   contested questions** (see `findings_multipool.md`). 5 of 6
   independent decomposer draws on Q* lean toward the wrong
   framework. Aggregation rule, prompt format, debate-round count,
   judge prompt sensitivity — all secondary effects relative to
   this.

2. **The §3 orthogonal decomposer produces topical diversity but
   not framework diversity proportional to the right answer's
   framework.** It surfaces the gold-aligned framework's persona
   archetype in only ~3 of 6 pools.

3. **Single-pool methodology results on contested questions can
   be unrepresentative.** The originating session's "baseline" pool
   was a favorable outlier (4/5 cells correct vs an average of
   1.8/5 across other pools). Don't draw methodology conclusions
   from a single pool draw.

4. **Aggregation interventions don't escape the bottleneck.**
   v_judge tested judge-as-aggregator. v5 tested probability output.
   Both produce small effects (occasional verdict flips, asymmetric
   confidence patterns) but neither materially improves accuracy
   on Q*. They don't fix what they don't address.

## What to NOT do (lessons learned)

1. **Don't infer gold from secondary documentation.** The
   originating session inferred gold = A from `NEXT_ITERATION.md`
   summary references; the grader returned gold = B. Always verify
   via blinded grader before interpreting verdicts.

2. **Don't repeatedly tweak aggregation in the hope of
   improvement.** The bottleneck is upstream of aggregation; try
   `decomposer_interventions.md` ideas first.

3. **Don't assume "tyranny of majority avoidance" is a bias leak.**
   It's a standard general judgment instruction. Bias leaks are
   about *what* to look for (specific axes, named failure modes),
   not *how* to judge (counting vs reasoning). See the bias-leak
   discussion in `findings_v_judge.md`.

4. **Don't forget to check session-context bias.** This session
   spent ~600+ sub-agent calls on Q* and saw extensive persona
   reasoning. The findings docs are designed to be read without
   that context — read them as a peer reviewer would, not as the
   experimenter.

## Priority order for next experiments

In rough decreasing order of expected information value:

### 1. Decomposer-level interventions (most informative)

The bottleneck is at decomposer-level frame selection. Several
intervention designs are spec'd in `decomposer_interventions.md`:

- **Brainstorm-50** (vs current brainstorm-20): does a wider
  candidate pool surface the under-represented framework reliably?
- **Framework-spotter pre-decomposer**: identify candidate frames
  from the question first, then seed decomposer.
- **Frame-explicit decomposer** (caution: bias-leak risk; see doc).
- **Decomposer orthogonality audit**: analyze the existing
  multipool decomposer outputs for actual epistemic diversity vs
  topical diversity.

The audit (option 4) costs almost nothing and should be done first;
it gives empirical grounding for which intervention matters.

### 2. Cross-question generalization (most generalization-relevant)

Test whether "pool draws asymmetric around gold" is Q*-specific or
a general pattern. Spec in `cross_question_generalization.md`.
Sample 2–3 contested questions from the unused GPQA Diamond pool
(via `csv_to_questions.py` with the existing ledger).

For each: run multipool at N=4 only (cheapest), 4-6 pools, round 0
only. Measure pool-direction distribution. If consistently biased
across questions, decomposer intervention is the methodology paper's
headline. If random direction, single-pool is OK and the Q* finding
is a coincidence.

### 3. v5_v9 wager arm (v9 component)

Spec in `wager_arm_spec.md`. Tests whether wager framing reduces the
asymmetric-confidence pattern v5 surfaced. Lower priority than (1)
and (2) because it's still an aggregation-layer test, not a
decomposer-layer one. Worth running if budget allows after (1) and
(2).

### 4. Eventually: full GPQA Diamond benchmark

The natural endpoint is comparing multi-persona-debate (with
whichever decomposer-level intervention works) against the GPQA
Diamond benchmark accuracy on a substantial sample (50+ questions).
**Wait until decomposer interventions are validated** — running
full benchmark with current decomposer would just reproduce the
"sometimes lucky pool, sometimes unlucky" pattern at scale.

## Reading order recap

1. `eval/PROTOCOL_v2.md` — the architecture (read §1, §3, §9)
2. `eval/methodology_experiments.md` — variant zoo with status
3. `eval/methodology_research/README.md` — orientation
4. `eval/methodology_research/findings_multipool.md` — read first;
   most load-bearing finding
5. `eval/methodology_research/findings_v_judge.md`
6. `eval/methodology_research/findings_v5.md`
7. `eval/methodology_research/decomposer_interventions.md`
8. `eval/methodology_research/cross_question_generalization.md`
9. `eval/methodology_research/wager_arm_spec.md`

## Pre-flight checks for next session

```bash
git status --short                              # clean tree
git branch --show-current                       # likely on main; branch off
ls eval/transcripts/experiment_*/findings.md    # full findings exist locally
ls eval/methodology_research/                   # this dir present
test -f eval/pilot_questions.jsonl              # gold lives here, gitignored
test -f eval/used_csv_indices.txt               # ledger for cross-question
```

## What's been committed in this session

- `eval/methodology_research/` (this directory) — leak-clean
  findings + design ideas + handoff
- Updates to `eval/methodology_experiments.md` (status of variants)
- This session's branch: `methodology-v_judge`

NOT committed (gitignored):
- `eval/transcripts/experiment_*/` — raw experiment data, full
  findings, design docs
- `eval/pilot_questions.jsonl`, `eval/pilot_results.jsonl`
- All persona/judge transcripts

## Tool budget

Pre-approve `Agent` permission in `/permissions` before starting —
all three experiments above involve 100+ sub-agent calls each.
