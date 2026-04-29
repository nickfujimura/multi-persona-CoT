# Decomposer-level interventions — design ideas

The `findings_multipool.md` headline is: **the bottleneck on
contested questions is at the decomposer layer, not at aggregation
or output format.** Multi-pool ensembling fails because pool draws
are asymmetric around gold — 5 of 6 §3 orthogonal decomposer draws
on Q* land in the wrong-framework persona space.

The natural next experiments are **decomposer-level interventions**
that try to make the persona pool more reliably surface the gold-
aligned reasoning framework — without leaking the answer.

This doc lists four candidate interventions in priority order. Run
the first (zero new dispatches) before any of the others; results
inform which of #2–#4 to pursue next.

---

## 1. Decomposer orthogonality audit (analytical, no new dispatches)

**Cost**: 0 sub-agent calls (analysis of existing decomposer outputs).
**Information value**: high — empirically grounds the multi-pool
finding.

We have decomposer outputs from 7 independent draws (Pools A through
F plus the v5_v9 Pool B = 7). Audit them on these dimensions:

- **Topical diversity**: do persona role names span different
  sub-fields (varied disciplinary sub-fields and applied-industry
  perspectives)?
- **Epistemic diversity**: do persona reasoning *modes* span
  different ways of evaluating truth (first-principles derivation /
  empirical pattern-match / phenomenological reasoning /
  meta-question-parsing / rule-of-thumb application)? Distinct from
  topical sub-field.
- **Framework coverage**: how many of the 7 pools include at least
  one persona whose role description is naturally aligned with
  framework α (the gold framework)? With framework β?

The hypothesis is that the §3 orthogonal prompt produces high
topical diversity within each pool but **low framework diversity
across pools** — every pool's "framework-β-aligned persona" slot
is filled with one of a small set of similar archetypes, while
specific framework-α-aligned roles are rare.

If the audit confirms low framework coverage variance, that
quantifies the bottleneck and motivates intervention #2 or #3.

If it shows high framework coverage variance (pools that *should*
cover framework α actually do), then the bottleneck is downstream
(in how personas reason despite their stated role) rather than at
decomposer-level role selection — and the priority list reorders.

**Tool**: dispatch one opus auditor sub-agent with Read access to
the 7 decomposer outputs. Returns a summary table.

---

## 2. Brainstorm-50 (vs current brainstorm-20)

**Cost**: ~140 calls (4 new pools × 5 N + r0 evals).
**Information value**: medium-high if intervention #1 confirms low
framework coverage.

The current §3 decomposer asks for **20 candidate personas, then
selects N**. Hypothesis: with 20 candidates, rare framework-aligned
roles may not be sampled often enough to be selected. Increasing the
brainstorm pool to 50 may surface them more reliably.

**Prompt change** (from PROTOCOL_v2 §3):
```
1. First, generate 20 candidate personas. ...
```
becomes
```
1. First, generate 50 candidate personas. Span a wide range of
   perspectives; vary along whatever dimensions you think most
   matter for genuine intellectual divergence.
```

Selection step (step 2) unchanged.

### Bias-leak hygiene

✅ No enumeration of axes. Same as current §3 prompt — describe
*what* (orthogonality) without specifying *how* (which dimensions).
Pure scale change.

⚠ Minor concern: a larger candidate pool may unintentionally
introduce specific framework-α-aligned roles even on pools where
those would otherwise have been rare. That's the desired effect, but
it means brainstorm-50's accuracy gains (if any) come partly from
*sampling* rather than *selection* — which is methodologically fine
but worth noting in any paper claim.

### Sweep recommendation

4 new pools × 5 N × round-0 evals. Compare per-pool modal
distributions to multipool's brainstorm-20 pools. If brainstorm-50
pools are materially less β-biased, the intervention works.

---

## 3. Framework-spotter pre-decomposer (with bias-leak risk)

**Cost**: ~150–200 calls (1 framework-spotter call per question + 4
pools × 5 N + r0).
**Information value**: high if it works; non-trivial leak risk.

Add a new sub-agent step before the decomposer:

```
You are a framework-spotter. Read the question and identify 2-4
candidate REASONING FRAMEWORKS that could produce different
answers. Describe each framework abstractly (one sentence per).
Do NOT solve the question or hint at which framework is correct.
Output only the framework descriptions.
```

Then the decomposer is seeded with these frameworks:

```
The following candidate reasoning frameworks have been identified:

Framework 1: <description>
Framework 2: <description>
[...]

Generate 20 candidate personas; ensure that AT LEAST ONE persona
is naturally aligned with each candidate framework. Otherwise
follow the standard orthogonal-selection process.
```

### Bias-leak warning (read carefully)

This intervention has a **specific bias-leak risk** that
PROTOCOL_v2 §3 explicitly warns against: enumerating axes for the
decomposer. The framework-spotter could:
- Plant a specific framework that happens to encode the answer
  (if the spotter's "Framework 1" description is detailed enough,
  the persona aligned with it will produce the correct answer
  *because the framework was planted*, not because of methodology
  improvement).
- Introduce wording from the question into the framework
  description, which then propagates into persona descriptions and
  could leak to round-0 evals.

The framework-spotter prompt above tries to mitigate this with
"abstract" and "do NOT hint" — but in practice, abstract framework
descriptions still encode framework structure. **This is testing
whether the methodology can *use* an explicit frame inventory, not
whether it can *discover* frames on its own.** Both are valid
research questions but they should not be conflated in any paper
claim.

### Sweep recommendation

If running this: pair it with one **control condition** that uses
the standard §3 decomposer (brainstorm-20, no framework-spotter)
on the same question, with 4 fresh pools each. The
framework-spotter pools should outperform the control if the
intervention works *and* if the framework-spotter's "framework
discovery" step is itself reliable.

If framework-spotter outperforms but only because its "candidate
frames" encode the answer, that's not a methodology gain — it's a
sneaky form of answer-leakage.

---

## 4. Frame-explicit decomposer (highest leak risk; document but don't run yet)

**Cost**: ~140 calls.
**Information value**: low — primarily a sanity check on framework-
spotter.

This is the most aggressive intervention: instead of asking the
decomposer to "select N orthogonal personas," directly tell the
decomposer "ensure your 4 personas span this specific list of
frameworks I'll provide."

The frameworks provided would be hand-curated for the question (or
generated by an offline framework-spotter and reviewed). The
decomposer becomes a constrained selector rather than an open
generator.

### Why this is the riskiest

The hand-curated framework list is by definition leak-prone — it
encodes whatever the experimenter believes the answer's structure
is. Personas seeded with explicit framework labels are predictably
going to vote for the framework's expected answer. **Accuracy gains
from this intervention can't be cleanly attributed to "methodology
improvement" vs "the experimenter knew the answer and engineered
the prompt to surface it."**

This is methodologically equivalent to: "what if we cheat?"

### Why document it anyway

Two reasons:

1. **Negative control.** Running it with hand-curated frameworks
   that *explicitly include the wrong framework* tests whether
   wrong-framework persona presence corrupts otherwise-correct
   pools. Useful as a sensitivity test.

2. **Upper bound on accuracy.** If frame-explicit decomposer with
   gold-aligned framework hand-curation achieves ~100% on a
   contested question, that establishes the upper bound the
   methodology *can* reach with perfect frame information. Then
   intervention #2 and #3 can be evaluated as fractions of that
   upper bound.

Don't run on an active question without explicit user sign-off.
Don't include in any methodology paper as a positive result.

---

## Suggested order of operations

1. **First**: dispatch the orthogonality audit (intervention #1).
   ~1 sub-agent call. If audit confirms low framework coverage, go
   to #2. If audit shows framework coverage was actually OK, the
   problem is elsewhere — re-read `findings_v_judge.md` for clues
   about persona-frame-locking-during-debate dynamics.

2. **Second**: brainstorm-50 (intervention #2). The cheap, low-leak-
   risk variant. ~140 calls. If it works (≥4 of 4 new pools land in
   gold-aligned framework majority), the methodology paper has a
   clean intervention.

3. **Third**: framework-spotter (intervention #3) — only if #2
   doesn't help and you want to test whether explicit frame
   inventory is necessary. Carefully document the bias-leak
   tradeoff.

4. **Fourth or never**: frame-explicit decomposer (intervention #4)
   — only as negative control or upper bound sanity check.

## Cross-question caveat

All four interventions are designed for Q* dynamics. If
`cross_question_generalization.md` reveals that pool asymmetry on
Q* is question-specific (i.e., other contested questions don't
exhibit the same biased pool draws), then these interventions may
be unnecessary and the focus should shift to identifying *which*
questions trigger asymmetric pool draws.

Run cross-question generalization in parallel with intervention #1
and #2 if budget allows; results will inform whether
interventions #3 and #4 are worth pursuing.
