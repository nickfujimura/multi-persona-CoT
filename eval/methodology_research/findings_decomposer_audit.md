# Decomposer orthogonality audit — findings

Empirical audit of the §3 orthogonal decomposer's persona rosters across
the 7 independent pool draws on Q* (the iter-2 contested case), plus a
gold-aware grader that maps roster-level framework coverage to vote
behavior. Designed in `decomposer_interventions.md` as intervention #1
(0–1 sub-agent calls; informs which of #2–#4 to pursue).

## Headline finding

**The "low framework coverage" interpretation in `findings_multipool.md`
is not supported by roster-level data.** All 5 reasoning frameworks the
question admits are present in all 7 pools, in at least one of each
pool's 5 N-rosters. The decomposer is not failing to surface the
gold-aligned framework — it surfaces it everywhere. The variance lives
elsewhere.

**Specifically, the bottleneck is *within-framework reasoning-path
heterogeneity*, not framework-roster generation.** The numerically
dominant framework (F1, 28% of all personas) is internally
heterogeneous — its constituent personas vote gold 100% in the
canonical-baseline pool and 33% in the worst pool, despite sharing the
F1 framework label. Same role-based classification, very different
empirical reasoning behavior.

This reframes the priority of next experiments away from
decomposer-roster interventions (brainstorm-50, framework-spotter)
toward (a) cross-question generalization to test whether
within-framework heterogeneity is Q*-specific, and (b) a new
intervention class operating at the persona-execution layer rather than
the persona-generation layer.

## Experimental design

Two-stage audit, both single sub-agent calls:

**Stage 1 — Blinded auditor (opus, no gold access):**
- Read Q* question text from `pilot_questions_blinded.jsonl`.
- From the question alone, identified 5 candidate reasoning frameworks
  (F1–F5) that could plausibly produce different defended answers.
- Read all 35 decomposer outputs (7 pools × 5 N-values) and classified
  each of the 188 resulting personas by primary framework alignment
  (with optional secondary). Used "Other" for personas too generic to
  classify.
- Output: per-pool framework-coverage tables + per-persona table.

**Stage 2 — Gold-aware grader (opus, gold access):**
- Read gold answer from `pilot_questions.jsonl`.
- Labeled each F1–F5 as a-priori gold-aligned, not-gold-aligned, or
  ambiguous, based on what reasoning *within* that framework would
  produce.
- Read each persona's round-0 vote letter from existing transcripts
  (180 personas in pools A–F, plus 30 in pool G under v5 prompt).
- Computed empirical gold-vote rate per framework, per pool, and per
  pool×framework cell.
- Returned only abstract framework labels and numeric counts to main
  session — gold letter, choice text, and topical vocabulary kept off
  main-session context.

Total cost: 2 sub-agent calls (one auditor + one grader). No new
decomposer or eval calls.

## Framework taxonomy (abstract)

The auditor identified five reasoning frameworks discoverable from the
question text. Labels are kept abstract here; full descriptions are in
the gitignored `audit_full.md`.

- **F1**: the *direct-derivation* framework. Default: apply a
  domain-standard computation chain to the question's stated quantity
  and select the answer the chain points at.
- **F2**: the *relational/inverse* framework. Default: treat the
  question's stated quantity as referring to a phenomenon distinct from
  but linked by an inversion/pairing rule to the answer-relevant
  phenomenon.
- **F3**: the *specialist-asymmetry* framework. Default: invoke a
  specialist body of knowledge that posits a directional offset between
  the stated phenomenon and the answer-relevant phenomenon, and shift
  the answer accordingly.
- **F4**: the *empirical-pattern-match* framework. Default: classify
  the question into a familiar corpus pattern and apply the corpus's
  typical conclusion.
- **F5**: the *meta-framing* framework. Default: parse which
  interpretation of the question is intended; the conclusion follows
  from which framing is selected, not from within-framing computation.

Auditor's caveats (full text in `audit_full.md`): F1 vs F3 boundary is
genuinely fuzzy for ~3–6 personas; F2 was applied strictly only when
inversion logic was explicit; meta-framing personas frequently list
"specialist asymmetry" as a candidate alternate framing they
interrogate, so F5 and F3 are highly correlated in this corpus.

## Headline coverage metric

For each framework F, count of pools (out of 7) that include at least
one persona primarily aligned with F in at least one of their five
N-rosters:

| Framework | Pools-covering (out of 7) |
|---|---|
| F1 | 7 |
| F2 | 7 |
| F3 | 7 |
| F4 | 7 |
| F5 | 7 |

Every framework appears in every pool. The "framework α appears in
only 3 of 6 pools" claim from `findings_multipool.md` is **not
reproduced** at the roster level under this audit's framework
taxonomy.

### N-coverage breadth per pool (number of N-values per pool with ≥1 persona of that framework, max 5)

| | F1 | F2 | F3 | F4 | F5 | Other |
|---|---|---|---|---|---|---|
| A | 5 | 4 | 4 | 4 | 5 | 1 |
| B | 5 | 4 | 4 | 5 | 5 | 4 |
| C | 4 | 4 | 5 | 4 | 4 | 1 |
| D | 5 | 5 | 4 | 4 | 4 | 2 |
| E | 4 | 4 | 3 | 5 | 5 | 3 |
| F | 5 | 4 | 4 | 4 | 4 | 5 |
| G | 5 | 5 | 4 | 2 | 4 | 0 |

Variation is small at this level (most cells in the 4–5 range).
Pool E has F3 in only 3/5 N-values; pool G has F4 in only 2/5. These
are the only meaningful coverage-breadth gaps and they don't track the
multipool vote pattern.

### Total primary-classification counts (N=188 personas)

| F1 | F2 | F3 | F4 | F5 | Other |
|---|---|---|---|---|---|
| 52 (28%) | 37 (20%) | 33 (17%) | 33 (17%) | 33 (17%) | 20 (11%) |

F1 numerically dominates every pool; F3, F4, F5 are roughly equal at
the next tier; F2 sits between.

**The decomposer is highly stereotyped.** Across all 7 pools, every
roster instantiates approximately the same archetypal lineup: one
F1-style derivation persona, one F2-style relational persona, one
F4-style empirical persona, one F5-style meta-framing persona, plus
discretionary additions at higher N. Topical role names vary; framework
slot is reproducible.

## Empirical gold-vote rate per framework

Pools A–F, letter-only baseline (180 personas). Full per-persona table
in `grader_full.md`.

| F | gold votes / total | rate |
|---|---|---|
| F1 | 26 / 41 | 63.4% |
| F2 | 22 / 30 | 73.3% |
| F3 | 2 / 26 | **7.7%** |
| F4 | 5 / 28 | 17.9% |
| F5 | 15 / 35 | 42.9% |
| Other | 17 / 20 | **85.0%** |
| **All** | **87 / 180** | **48.3%** |

A-priori labels (from `grader_full.md`):

| F | A-priori | Empirical (A–F) | Aligned? |
|---|---|---|---|
| F1 | NOT-GOLD | 63% gold | **disagrees** — F1 votes gold more often than predicted |
| F2 | GOLD | 73% gold | aligned |
| F3 | NOT-GOLD | 8% gold | strongly aligned |
| F4 | AMBIGUOUS | 18% gold | aligns with NOT-GOLD-leaning side |
| F5 | AMBIGUOUS | 43% gold | aligns (mixed) |

Two surprises stand out:

1. **F1 a-priori NOT-GOLD but empirically 63% gold.** Inspection of
   transcripts (in `grader_full.md`) shows many F1 personas append the
   F2-style finishing move at the end of their derivation chain. The
   auditor classified these by their *operative primary* move
   (derivation), but the answer-generating move at the end is
   relational. The F1 label is collapsing two empirically distinct
   subpopulations: F1+relational-finish (gold-voting) and F1-pure
   (non-gold-voting).

2. **"Other" personas vote gold 85% of the time.** These are personas
   whose ROLE/CONCERNS were too generic for the auditor to assign a
   specific framework. They vote gold MORE often than any specific
   framework. The likely interpretation: generic personas default to
   the model's underlying belief on the question, and that belief is
   gold-aligned at the population level. Specific framework
   commitments pull personas *away* from this base rate.

## Per-pool gold-vote rate

Pools A–F, letter-only:

| Pool | gold / 30 | rate |
|---|---|---|
| A (canonical baseline) | 22 | **73.3%** |
| B | 16 | 53.3% |
| C | 13 | 43.3% |
| D | 12 | 40.0% |
| E | 11 | 36.7% |
| F | 13 | 43.3% |

Modal-gold cells per pool match `findings_multipool.md` exactly
(A=4/5, B=3/5, C=1/5, D=1/5, E=0/5, F=2/5; ties counted as no).

## Crowding hypothesis test

The hypothesis: F2 personas (the small gold-aligned cohort, ~17% of
personas) are outnumbered by F1 personas (28%) and other
non-gold-leaning frameworks, so F2 is reliably outvoted at the cell
level.

Test: for each (Pool, N) cell across pools A–F (n=30 cells), compute
F2-fraction = (# F2 personas) / N and check whether it predicts
modal=gold.

| | F2-fraction (mean) |
|---|---|
| Cells where modal = gold (n=11) | 0.176 |
| Cells where modal ≠ gold (n=19) | 0.179 |

Essentially identical. **F2-fraction has near-zero predictive power for
modal=gold at the cell level.** Counterexamples:

- One cell (F-N4) had F2-fraction = 0.500 — twice the corpus mean —
  yet produced a tie, not modal-gold. One of the two F2 personas voted
  outside the F2-gold path.
- One cell (D-N6) had F2-fraction = 0.333 — second-highest in the
  table — but modal was non-gold. The two F2 personas split votes; F1
  in that roster also split, dragging modal away from gold.
- One cell (A-N7) had F2-fraction = 0.000 — no F2 personas in the
  roster — yet modal was gold. F1+Other personas in that roster all
  voted gold.

**The narrow crowding hypothesis is refuted.** Roster composition in
gold-aligned-framework personas does not predict cell verdict. Whether
F2 is present (or how many F2 personas a roster has) matters less than
how the *non-F2* personas in the roster reason.

## The actual bottleneck — within-F1 heterogeneity

What does predict cell verdict? **Pool-level F1 gold-vote rate.**

| Pool | F1 gold / total | rate |
|---|---|---|
| A (canonical baseline) | 6 / 6 | **100%** |
| B | 5 / 7 | 71% |
| C | 5 / 8 | 63% |
| D | 4 / 7 | 57% |
| E | 2 / 6 | **33%** |
| F | 4 / 7 | 57% |

Pool A's F1 personas vote gold every time. Pool E's F1 personas vote
gold one third of the time. Same framework label. Same role archetype.
Very different empirical reasoning behavior.

The grader's reading of transcripts (in `grader_full.md`) attributes
this to a sub-path split *within* F1: some F1 personas append a
relational/inversion finishing move at the end of their derivation
chain (gold-voting), while others stop at the direct mapping
(non-gold-voting). The auditor's primary-label schema cannot
distinguish these subpopulations from role/concerns/traits text alone;
the difference manifests only in the persona's actual reasoning at
round 0.

This is the most load-bearing finding of the audit. **The
multipool-level pool-to-pool variance is driven by pool-to-pool
variation in the F1+finishing-move subpopulation's prevalence**, not
by framework-roster differences.

The combined gold-aligned-vote-fraction (F1+relational + F2 + Other-default) is
high in Pool A and low in Pool E, and that combined fraction *does*
predict modal=gold cleanly. F2-fraction alone does not.

## Implications for methodology

### Decomposer-level interventions are mistargeted

Interventions designed in `decomposer_interventions.md` (brainstorm-50,
framework-spotter, frame-explicit decomposer) all target
*roster-level* framework coverage. The audit shows roster-level
coverage is uniformly high already. A wider candidate-pool brainstorm
(50 vs 20) would surface more topical variants of the same archetypes;
it would not change F1-classified personas' tendency to apply or skip
the relational finishing move at round 0. The bottleneck is downstream
of role assignment.

### Specific framework commitments may hurt accuracy on this question

The "Other" finding (85% gold-vote rate among personas too generic to
classify, vs 48% across labeled personas) is methodologically
subversive. It suggests that on this question, **giving a persona a
specific framework commitment pulls them away from the model's
gold-aligned base rate**. The orthogonal-decomposer optimization may be
optimizing for diversity at the cost of accuracy on questions where
expert reasoning paths land further from the gold answer than generic
reasoning does.

This is a single-question finding and should not be over-interpreted,
but it warrants direct testing (see "next experiments" below).

### Specialist expert reasoning is reliably wrong on Q*

F3 — the framework whose role descriptions explicitly invoke
specialist domain knowledge applicable to Q*'s topic — votes gold only
8% of the time. The "expert persona" archetype is mostly wrong here.
This is striking given the methodology's stated motivation
(multi-persona reasoning → leverage specialist expertise on
graduate-level questions).

Three plausible interpretations:

1. **Q* is a question whose stated framing baits specialist reasoning
   into a wrong move**, and naive/generic reasoning is closer to gold.
   If true, this is a failure mode specific to "expertly-misdirected"
   contested questions.
2. **F3 personas in this corpus systematically over-apply their
   specialist knowledge** even when it doesn't apply to the question
   at hand, perhaps because the persona prompt instructs them to
   anchor in their concerns.
3. **The framework taxonomy is mis-classifying F3 — it captures
   personas who pre-commit to a specific specialist move** (the
   asymmetry move) regardless of whether that move matches the
   question. The F3-classified personas in pools where F3 is
   non-gold-leaning may all be applying the *same* wrong specialist
   move (since the framework label was defined to capture exactly
   that).

Either way, the multi-persona methodology's value proposition
("specialist personas surface specialist reasoning") is undermined for
contested questions where the gold answer doesn't follow from the
specialist move that anchors F3.

## Recommendations for next experiments

### Highest priority — Cross-question generalization (revised emphasis)

`cross_question_generalization.md` proposes testing pool asymmetry
across multiple questions. Given this audit's findings, the test
should focus on: **does within-framework heterogeneity (specifically:
the F1-style "derivation framework" splitting into gold-voting and
non-gold-voting subpaths) generalize to other contested questions?**
If yes, the methodology has a structural issue at the
persona-execution layer, not the decomposer layer. If no, Q* is a
specific pathology.

This experiment is now substantially more important than
brainstorm-50.

### Downgraded priority — brainstorm-50 (intervention #2)

Likely a small intervention. The decomposer is already producing the
right archetypes; widening the brainstorm pool would just give more
topical variants of the same five framework archetypes. It might
slightly increase F2 representation (if 30 extra candidates surface
two more F2-archetype roles), but F2-fraction doesn't predict cell
verdict, so the expected accuracy lift is small.

May still be worth running as a sanity check, but it's no longer the
highest-information-value intervention.

### Downgraded priority — framework-spotter (intervention #3)

Same reason. Framework-spotter would seed the decomposer with
"frameworks the question admits" and ensure ≥1 persona per framework.
Roster-level coverage is already at ceiling. The intervention's only
benefit would be if it surfaces frameworks the §3 prompt doesn't —
but the audit found that the §3 prompt produces all 5 plausible
frameworks already. Framework-spotter would mostly be redundant for Q*.

### New intervention class — persona-execution layer

The within-F1 heterogeneity finding suggests a new intervention
direction: **once a persona is dispatched, what can shift their
within-framework reasoning sub-path toward gold?** Candidates:

- **Per-persona cross-check prompt**: after the persona completes
  their primary derivation, prompt them to check whether an inversion
  or pairing rule applies before committing. Bias-leak hazard: this is
  exactly the kind of "watch out for X" prompt the protocol warns
  against. It encodes the gold path into the prompt, contaminating the
  test on Q*.
- **Untouched pre-mortem (v7 from `methodology_experiments.md`)**: the
  variant that has each persona declare their own falsification
  conditions at round 0 and re-evaluate at round 1. The
  bias-leak-clean version of "make personas reason more carefully
  within their framework" — bears the same goal but doesn't pre-tell
  them what to check for. Worth running as a clean test of
  persona-execution intervention.
- **Persona-execution diagnostic experiment**: re-run round 0 for the
  same Q* personas under two prompt variants — current (one persona,
  one shot) vs. self-doubt prompt ("after your first answer, consider
  one alternative reading and explain why you reject it"). If F1
  personas under the self-doubt variant shift toward gold reliably, the
  persona-execution layer is the right intervention layer. Bias-leak
  risk: medium — the self-doubt prompt is a generic move not specific
  to Q*'s gold framework. Distinguish carefully from the
  per-persona-cross-check above (which would name the specific
  finishing-move).

### Audit refinement (cheap, useful for methodology paper)

The auditor's F1/F3 boundary is fuzzy for ~3–6 personas. A second
auditor pass could refine F1 into F1-relational-finish and F1-pure
sub-categories using each persona's actual round-0 reasoning text
(not just role/concerns/traits). This would directly quantify the
within-F1 heterogeneity and produce a publishable persona-execution
metric. Cost: ~1 opus sub-agent call.

## Limitations

- **N=1 question.** Within-F1 heterogeneity is the bottleneck on Q*;
  whether it's the bottleneck on other contested questions is exactly
  what cross-question generalization tests. Don't generalize from this
  audit alone.
- **Auditor classification is primary-label-only.** The auditor's
  F1/F3 boundary is genuinely fuzzy; the F1 vs F1+finishing-move split
  was discovered by the grader at vote-extraction time, not by the
  auditor. A more robust audit would use full reasoning transcripts
  (not just role descriptions) to classify, which is more expensive.
- **Pool G (v5 prompt) behaves materially differently** from pools A–F
  on framework gold-rates (F1 jumps from 63% to 90%; F2 drops from
  73% to 43%). Pool G is excluded from headline gold-vote-rate numbers
  but reported separately in `grader_full.md`. The v5 prompt's effect
  on persona reasoning may itself be an interesting follow-up.
- **The "Other" 85% gold-vote rate has small denominator (n=20)** but
  the pattern is consistent across pools. Worth treating as a real
  signal pending replication.
- **Main session was exposed to question text** (read
  `pilot_questions_blinded.jsonl` early in this experiment to verify
  splitter pattern) and to gold answer indirectly (via reading the
  grader's `grader_full.md` to verify outputs). Audit and grader
  themselves were properly blinded/cleared as designed; main-session
  exposure was a procedural lapse that does not affect the data, but
  is noted for transparency. The §9.10 question-splitter pattern is
  still un-implemented and would prevent this in future iterations.

## Files

- Raw experiment data (gitignored):
  `eval/transcripts/experiment_decomposer_audit/audit_full.md`
  `eval/transcripts/experiment_decomposer_audit/grader_full.md`
- This findings doc: leak-clean (no question/choice text, no topical
  vocabulary, no gold letter, no role names).
