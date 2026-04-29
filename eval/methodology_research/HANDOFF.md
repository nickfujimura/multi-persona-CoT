# Handoff to next methodology research session

You are a fresh Claude Code session picking up a research thread.
This document is the entry point for that thread. Read it end-to-end
before doing anything.

## What this project is, in one paragraph

This repo is testing whether a **multi-persona chain-of-thought
debate methodology** (decomposer generates N expert personas → each
persona answers in parallel → personas debate over multiple rounds →
final verdict via aggregation) produces better answers on hard
multiple-choice questions than plain zero-shot CoT. The test bench is
**GPQA Diamond** (graduate-level science MCQ, 198 questions). The eval
substrate is **Sonnet 4.6**, which scores ~90% accuracy on this set
zero-shot. The methodology architecture lives in
`eval/PROTOCOL_v2.md` (the README at repo root and `eval/README.md`
have older background; PROTOCOL_v2 is current). Top-level project
motivation is in the repo-root README.md (worth skimming for
intuition; not load-bearing for this thread).

## Two threads run in this repo — make sure you're on the right one

**Thread 1 — iteration evaluation.** Per-iteration runs of the
methodology on fresh 10-question samples, with results appended to
`eval/summary.md`. Task brief: `eval/NEXT_ITERATION.md`. Iter-1 and
iter-2 are done; iter-3 is queued.

**Thread 2 — methodology research (this thread).** Targeted
experiments on *specific aspects* of the methodology, motivated by
findings from prior iterations (especially iter-2's contested case,
"Q*" = `gpqa_007`). Task briefs live in
`eval/methodology_research/`. This thread does NOT advance the
iteration counter.

**If you were told to "run iteration 3" / "do a fresh pilot eval" /
"sample 10 new questions" → you're on Thread 1. Stop reading this
file and read `eval/NEXT_ITERATION.md`.**

**If you were told to "pick a methodology experiment" / "investigate
why the methodology fails on Q*" / "test a methodology variant" →
you're on Thread 2.** Continue reading.

## Read first, in this order

1. `eval/PROTOCOL_v2.md` — the architecture (§1 = blinded planner +
   transport pattern; §3 = prompt templates with mandatory bias-leak
   warning; §9 = iter-2 amendments — these are mandatory). The
   single most important section for any new experiment is §3's
   bias-leak warning — read carefully before iterating any prompt.
2. `eval/methodology_research/README.md` — orientation across the
   methodology-research findings.
3. **`eval/methodology_research/uncertainty_gated_evaluation.md`** —
   the highest-priority next experiment spec. If you're going to run
   an experiment, this is the one.
4. `eval/methodology_research/findings_decomposer_audit.md` — the
   most recent findings doc. Substantively revises the older
   multipool interpretation. Read this before reading multipool.
5. `eval/methodology_research/findings_multipool.md`,
   `findings_v_judge.md`, `findings_v5.md` — older single-question
   findings. Read after the audit findings (which reframe them).
6. `eval/methodology_research/decomposer_interventions.md` —
   intervention designs. Note: #1 (orthogonality audit) is done; #2,
   #3, #4 are downgraded by the audit.
7. `eval/methodology_research/cross_question_generalization.md` —
   superseded by `uncertainty_gated_evaluation.md`. Skim only.
8. `eval/methodology_research/wager_arm_spec.md` — deferred.

You do NOT need to read `eval/summary.md`, `eval/README.md`, or
`eval/NEXT_ITERATION.md` for Thread 2 work — those are Thread 1.

## What changed in the most recent session (audit-completion)

Two pieces of context drove a substantial reprioritization:

1. **The decomposer audit refuted the multipool interpretation.** All
   5 plausible reasoning frameworks for Q* are present in all 7
   pools (the "framework α appears in only ~3 of 6 pools" claim from
   `findings_multipool.md` is not supported at the roster level).
   The bottleneck is *within-framework* reasoning-path
   heterogeneity, not roster-level coverage. See
   `findings_decomposer_audit.md`. This downgrades all
   decomposer-layer interventions
   (`decomposer_interventions.md` #2, #3, #4) and the prior
   `cross_question_generalization.md` design.

2. **The 90% baseline matters more than prior iterations
   acknowledged.** Sonnet 4.6 hits ~90% on GPQA Diamond zero-shot.
   The methodology has at most 10pp of headroom, concentrated on the
   hardest questions. Iter-2 already showed A=10/10 vs B=9/10 on a
   10-question sample — the methodology *lost* on Q*. With n=10 we
   can't tell if the methodology is net-zero, slightly positive, or
   slightly negative at population scale. The iter-N pattern (10
   questions per iteration, reactive analysis) is structurally
   underpowered for population-level claims.

Both observations point to the same redesign: gate by zero-shot
uncertainty, test methodology levers as factors on a dev set,
validate on a sealed holdout. That's
`uncertainty_gated_evaluation.md`.

## What to internalize before designing or running anything

1. **Roster-level framework coverage on Q* is uniformly high.** All
   5 plausible frameworks appear in all 7 pools. The decomposer is
   highly stereotyped — same archetypal lineup everywhere. The
   diversity it produces is real.

2. **Pool-to-pool vote variance is real, but the source is
   within-framework reasoning-path heterogeneity.** F1 (the
   numerically dominant framework, 28% of personas) splits into a
   gold-voting subpath and a non-gold-voting subpath without any
   role-level signal distinguishing them. Pool A's F1 = 100% gold;
   Pool E's F1 = 33% gold.

3. **The crowding-of-gold-aligned-personas hypothesis is refuted.**
   Roster composition (F2-fraction) does not predict modal=gold
   (0.176 in modal=gold cells vs 0.179 in modal-not-gold cells).

4. **Specific framework commitments may HURT accuracy on Q*.**
   "Other" personas (declined-to-classify generic role descriptions)
   vote gold 85% of the time, higher than any specific framework.
   Generic personas inherit the model's well-calibrated prior;
   specific framework commitments perturb it, often unhelpfully.

5. **Aggregation interventions don't escape the bottleneck.**
   v_judge produced a 1-cell flip on Pool A (real but small effect).
   v5 produced mixed-direction flips. Neither addresses
   within-framework reasoning heterogeneity.

6. **At a 90% baseline, the productive question is not "does
   multi-persona work" but "on which questions does it help and on
   which does it hurt."** The methodology may be a useful conditional
   tool (gated on zero-shot uncertainty), even if it's a worse
   blanket replacement for zero-shot.

## What NOT to do (lessons learned across sessions)

1. **Don't pursue brainstorm-50 / framework-spotter / frame-explicit
   decomposer as high-priority.** All three target roster-level
   framework coverage, which the audit shows is not the bottleneck.

2. **Don't run methodology evaluation on small samples (n≤30) and
   draw population-level conclusions.** At a 90% baseline you need
   ~200 questions to detect 5pp effects.

3. **Don't write prompts that pre-tell the personas / judge /
   decomposer to "watch out for" specific failure modes.** That's
   how experiments get planted-answer contamination — see
   `PROTOCOL_v2 §3` bias-leak warning. Read that section carefully
   before iterating any prompt.

4. **Don't read `pilot_questions.jsonl` or
   `pilot_questions_blinded.jsonl` directly from main session.** Use
   the `PROTOCOL_v2 §9.10` question-splitter pattern. Both prior
   methodology-research sessions had main-session exposure to
   question text via direct reads of the blinded JSONL — note their
   transparency caveats in the findings docs and don't repeat the
   mistake. The new spec
   (`uncertainty_gated_evaluation.md`) finally implements the
   splitter; don't shortcut it.

5. **Don't assume "tyranny of majority avoidance" is a bias leak.**
   It's a standard general judgment instruction. Bias leaks are
   about *what* to look for (specific axes, named failure modes),
   not *how* to judge.

6. **Don't repeatedly tweak aggregation in the hope of overall
   accuracy improvement.** Aggregation effects are small and
   contingent. Treat as small expected effect, not a methodology
   centerpiece.

7. **Don't commit experiment transcripts.** All
   `eval/transcripts/experiment_*/` directories and
   `eval/pilot_*.jsonl` are gitignored — the GPQA dataset terms
   forbid verbatim question/choice text in public commits. Only
   commit leak-clean findings docs in
   `eval/methodology_research/`, after running an opus leak-checker
   (PROTOCOL_v2 §9.6).

## Operational essentials — how experiments are actually run here

This section is a brief reference to the workflow patterns. Read
`PROTOCOL_v2.md §1, §3, §9` for full detail; this is just the
"what's load-bearing" summary.

- **Main session is dumb transport.** It dispatches sub-agents via
  the `Agent` tool, captures their structured returns, and runs
  graders/auditors at phase boundaries. It does NOT construct
  reasoning prompts itself or read gold answers.

- **Sub-agents do real work.** Use `model="sonnet"` for eval
  sub-agents (decomposer, persona, synthesizer); `opus` for
  auditors / graders / leak-checkers / planners that need to be
  smart. Hold model constant within an experimental arm — mixing
  models invalidates A-vs-B comparisons.

- **Sub-agent self-save pattern (§9.2).** Each eval sub-agent saves
  its full reasoning to disk via Write before returning. Returns
  ONLY a structured single-line summary (e.g.,
  `Answer: <letter>`). This collapses main-session context cost
  from ~thousands of tokens of reasoning per call to dozens of
  tokens of letter. Apply to every eval call.

- **Planner Read pattern (§9.3).** Round-N+1 planners get
  Read-only access to specific transcript paths. They build
  prompts that reference those paths so the next-round eval
  sub-agent reads transcripts itself. Avoids main-session
  paraphrasing → drift.

- **Blinding (§9.5).** Main session NEVER reads
  `pilot_questions.jsonl`. Gold-aware tasks (grading,
  framework-labeling) are dispatched as opus sub-agents that read
  gold themselves and return only count summaries / abstract
  labels.

- **Question-splitter (§9.10).** Mandatory now.
  `uncertainty_gated_evaluation.md` Phase 1 implements it: a
  Read+Write splitter sub-agent reads `pilot_questions.jsonl` and
  writes per-question files containing question + choices but NOT
  the answer field. Main session and downstream sub-agents
  reference per-id files by path, never read the JSONL directly.

- **Opus auditor after each round (§9.4).** Spot-checks transcripts
  for tool-misuse, persona drift, leakage signals, refusal
  patterns. Read-only.

- **Opus leak-checker before commit (§9.6).** Mandatory before any
  commit of `summary.md` or any methodology-research findings doc.
  Cross-checks staged file against `pilot_questions.jsonl` for
  verbatim ≥8-word substrings, unique numerical values, named
  entities, topical vocabulary. Read-only — main session applies
  redactions.

- **Sub-agents cannot recursively call Agent (§9.1).** Only main
  has Agent. Plan dispatches accordingly — don't try to delegate
  "the whole experiment" to a sub-agent.

- **Tool returns come back in completion order, not dispatch
  order.** When mapping persona → letter, always Read the saved
  transcripts to verify rather than relying on the order parallel
  results came back in.

- **Pre-approve `Agent` permission in `/permissions` before
  starting.** A 10-question multi-persona run = 100+ permission
  prompts otherwise. Big experiments (1000+ calls) need this
  approved up front.

## Priority order for next experiments

In rough decreasing order of expected information value. This list
is the canonical "what to work on next" ordering.

### 1. Uncertainty-gated evaluation — the highest-priority next experiment

**Spec**: `uncertainty_gated_evaluation.md`. ~1200-1500 sub-agent
calls across 4 phases (mapping → split → lever testing on dev →
holdout validation). Tests three methodology levers (decomposer
choice, aggregation rule, persona-execution intervention) as factors
on a population-scale sample, with proper test/holdout discipline.

**Why this is highest priority**: it's the only experiment that can
produce a population-level claim about whether the methodology
works. All prior experiments were N=1 (single question) and gave us
mechanism-level findings. This experiment tests whether those
mechanisms generalize.

**Outcomes**, all publishable:
- Methodology beats zero-shot on uncertain questions → write up the
  gating rule + winning lever configuration.
- Methodology = zero-shot on uncertain questions → null result on
  accuracy, stratification analysis still tells you when to use vs
  not use the methodology.
- Methodology < zero-shot on uncertain questions → the specific
  framework commitments hypothesis from the audit holds at scale;
  the methodology is a worse intervention than a well-calibrated
  prior.

### 2. Audit refinement — F1 sub-classification

The decomposer audit's F1 vs F1+finishing-move split was discovered
by the grader at vote-extraction time, not by the auditor. A second
auditor pass that classifies F1 personas using their *full round-0
reasoning text* would directly quantify within-F1 heterogeneity.
**Cost**: ~1-2 opus sub-agent calls. **Best run during Phase 3 of
the uncertainty-gated experiment** as a diagnostic on dev decomposer
outputs, not as a separate experiment.

### 3. Persona-execution intervention battery (after #1 if positive signal)

If `uncertainty_gated_evaluation.md` Lever C (v7 pre-mortem) shows
positive signal, follow up with other clean (non-bias-leaking)
persona-execution interventions: "consider one alternative reading"
prompt, confidence-elicited weighting, etc.

### 4. v5_v9 wager arm — deferred indefinitely

Spec in `wager_arm_spec.md`. Aggregation-layer effect, unlikely
load-bearing at population scale. Run only after #1 and #3 if a
specific aggregation-layer hypothesis emerges.

### 5. Decomposer interventions #2, #3, #4 — downgraded

See `decomposer_interventions.md`. The audit showed roster-level
coverage is not the bottleneck. Run only if
`uncertainty_gated_evaluation.md` Lever A reveals a specific
roster-coverage-related failure mode that the data shows.

### 6. Multi-pool ensembling (Phase 5 add-on)

Only useful if `uncertainty_gated_evaluation.md` shows positive
methodology signal. Don't run as a standalone experiment.

## Repo layout (what's where, gitignored vs committed)

```
multi-persona-CoT/
├── README.md                          (top-level project intro — committed)
├── Multi_Persona_Problem_Solver.py    (original PoC code — committed)
├── eval/
│   ├── README.md                      (iter-1 task brief, historical — committed)
│   ├── NEXT_ITERATION.md              (iter-3 task brief, Thread 1 — committed)
│   ├── PROTOCOL_v2.md                 (current methodology — committed; READ)
│   ├── summary.md                     (iter results — committed)
│   ├── methodology_experiments.md     (variant catalog — committed)
│   ├── csv_to_questions.py            (sampling tool — committed)
│   ├── GPQA/                          (gitignored — dataset)
│   ├── pilot_questions.jsonl          (gitignored — gold lives here)
│   ├── pilot_questions_blinded.jsonl  (gitignored — no answer field)
│   ├── pilot_questions_full.jsonl     (gitignored — full corpus)
│   ├── pilot_results.jsonl            (gitignored — per-iter grading)
│   ├── used_csv_indices.txt           (gitignored — sampling ledger)
│   ├── transcripts/                   (gitignored — all experiment data)
│   │   ├── iteration_1/, iteration_2/ (Thread 1 archives)
│   │   ├── experiment_v_judge/        (Thread 2 — v_judge data)
│   │   ├── experiment_multipool/      (Thread 2 — multipool data)
│   │   ├── experiment_v3/             (Thread 2 — canonical baseline pool)
│   │   ├── experiment_v5_v9/          (Thread 2 — v5 data)
│   │   ├── experiment_decomposer_audit/ (Thread 2 — most recent)
│   │   └── experiment_uncertainty_gated/ (Thread 2 — for the new spec to populate)
│   └── methodology_research/          (Thread 2 — committable findings + specs)
│       ├── README.md                  (orientation)
│       ├── HANDOFF.md                 (this file)
│       ├── uncertainty_gated_evaluation.md  (new highest-priority spec)
│       ├── findings_decomposer_audit.md     (newest findings)
│       ├── findings_multipool.md, findings_v_judge.md, findings_v5.md
│       ├── decomposer_interventions.md      (interventions; #2-4 downgraded)
│       ├── cross_question_generalization.md (superseded)
│       └── wager_arm_spec.md                (deferred)
```

Everything in `eval/transcripts/` and the `pilot_*.jsonl` files is
gitignored to prevent dataset-terms violation. Findings docs in
`eval/methodology_research/` are committable but require leak-check
first (PROTOCOL_v2 §9.6).

## Pre-flight checks for next session

```bash
git status --short                              # clean tree
git branch --show-current                       # likely main; branch off:
                                                # git checkout -b methodology-<your-experiment-name>
ls eval/transcripts/experiment_*/               # prior experiment dirs present
ls eval/methodology_research/                   # this directory present
test -f eval/pilot_questions.jsonl              # gold lives here
test -f eval/pilot_questions_blinded.jsonl      # for blinded reads (will be deprecated when splitter is implemented)
test -f eval/used_csv_indices.txt               # ledger
wc -l eval/used_csv_indices.txt                 # confirm prior indices (expect ≥25 lines)
```

## What to do next

If you're picking up this thread to advance methodology research:

1. Read the files listed in "Read first" above, in order. Budget
   30–45 minutes — these are dense.
2. Form your own interpretation. The findings docs are written for
   peer-review reading, not for you to inherit prior-session
   narratives uncritically.
3. Pick one experiment from the priority order above (almost
   certainly `uncertainty_gated_evaluation.md` — it's both highest
   priority and best-specced).
4. Pre-approve `Agent` in `/permissions`.
5. Pre-flight checks (above).
6. Branch off main: `git checkout -b methodology-<your-experiment-name>`.
7. Execute the spec end-to-end. Save transcripts to gitignored
   `eval/transcripts/experiment_<name>/`.
8. Write a leak-clean findings doc to
   `eval/methodology_research/findings_<name>.md`.
9. Run an opus leak-checker on the findings doc before commit
   (PROTOCOL_v2 §9.6).
10. Update this `HANDOFF.md` for the NEXT session — add a "what
    changed in this session" entry and revise the priority order if
    your findings reorder things.

Don't commit unless the user explicitly asks — committing methodology
findings is durable + visible; ask first.

## Tool budget

Pre-approve `Agent` permission in `/permissions` before starting.
The highest-priority experiment (`uncertainty_gated_evaluation.md`)
involves ~1500 sub-agent calls. Audit refinement (#2) is 1–2 calls
and can run interactively without pre-approval.

## What's been committed in recent sessions

- `eval/methodology_research/` — leak-clean findings + spec docs.
  Most recent first:
  - `uncertainty_gated_evaluation.md` (new spec; highest priority)
  - `findings_decomposer_audit.md` (audit findings)
  - This `HANDOFF.md` (rewritten for newcomer-friendliness)
  - Earlier (prior session): `findings_multipool.md`,
    `findings_v_judge.md`, `findings_v5.md`,
    `decomposer_interventions.md`,
    `cross_question_generalization.md` (now annotated as
    superseded), `wager_arm_spec.md`
- Updates to `eval/methodology_experiments.md` (status block)
- This session's branch: `methodology-decomposer-audit`
- Prior methodology-research session's branch: `methodology-v_judge`

NOT committed (gitignored):
- `eval/transcripts/experiment_*/` — raw experiment data, full
  findings, design docs (including `audit_full.md`,
  `grader_full.md`)
- All `pilot_*.jsonl` files
- All persona / judge / auditor transcripts
