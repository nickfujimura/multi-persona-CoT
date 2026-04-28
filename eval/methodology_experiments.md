# Methodology experiments — handoff doc for fresh sessions

This document hands off a series of prompt-engineering experiments on the
multi-persona-CoT debate methodology. **A fresh session is the intended
audience** — context-bias from the originating session is a known risk,
and is called out variant-by-variant below.

The fresh session should:
- Read this doc and `PROTOCOL_v2.md` (especially §9 amendments).
- Pick one variant (v5, v_judge, v7, or v9) and run it in isolation
  against the protocol baseline.
- Form its own interpretation of the data. Do NOT inherit prior-session
  narratives about why the methodology fails or succeeds — look at the
  transcripts and decide for yourself.

---

## What the protocol baseline is

The current baseline is `PROTOCOL_v2.md` including the §9 iter-2
amendments and the §3 orthogonal decomposer (§9.11). All four variants
below test ONE additional change against that protocol baseline, in
isolation, on the same contested question. Run them sequentially.

**Don't assume the protocol baseline is "good"** — the originating
session observed measurable round-0 distribution shifts when the
orthogonal decomposer was introduced, but whether the methodology
reaches correct verdicts on contested cases is the active research
question. Look at the data; form your own view.

For locally-stored experiment data from the originating session, see
`eval/transcripts/experiment_v1/` (no orthogonal decomposer — older
protocol baseline) and `eval/transcripts/experiment_v3/` (orthogonal
decomposer — current protocol baseline). Both directories are
gitignored and contain transcripts you can compare against.

---

## Test infrastructure

- **Question**: iter-2's `gpqa_007` from `eval/pilot_questions.jsonl`
  (gitignored). Read it via the standard blinded pattern from
  `NEXT_ITERATION.md` §4c.
- **Save transcripts to**: `eval/transcripts/experiment_v{N}/...`
  (gitignored). Path conventions follow PROTOCOL_v2 naming.
- **Per-variant**: 5 decomposers (one per N in {4, 5, 6, 7, 8}) + 30
  round-0 evals + 30 round-1 evals = ~65 sub-agent calls. Plus an
  opus auditor at the end.
- **Compare to**: prior baseline-protocol transcripts in
  `eval/transcripts/experiment_v3/` (gitignored, available locally).
  Note that older `eval/transcripts/experiment_v1/` data corresponds
  to the *pre-orthogonal-decomposer* protocol — useful as a contrast,
  but not the current baseline.

---

## Variants

Each variant gives:

- **Intent**: what's being tested.
- **Minimal prompt change**: a starting prompt the fresh session can
  use as-is, or revise.
- **Bias-leak callouts**: places where the originating session's
  experience may have crept into the wording. The fresh session
  should evaluate whether those callouts apply and revise as needed.
- **Analysis**: what to compute from the variant's data.

### v5 — probabilistic outputs + log-pooled aggregation

**Intent**: replace each persona's discrete letter answer with a
probability vector. Aggregate by log-pooled probability instead of
majority vote. Tests whether the verdict mechanism (count vs. probability)
is a meaningful lever.

**Minimal prompt change** — append to round-0 and round-1 templates:

```
After your reasoning, on the final line output:
PROBABILITIES: A=<p_A> B=<p_B> C=<p_C> D=<p_D>
where the four values sum to 1.0 and reflect your actual uncertainty.

Then on the next line:
ANSWER: <modal letter>
```

**Aggregation rule**: for each letter L, compute the geometric mean
across personas: `p_pool[L] = (∏_i p_i[L])^(1/N)`. Renormalize across
letters. Final answer = argmax.

**Bias-leak callouts**:

- The originating session considered adding "use specific values like
  0.62, 0.27 — not buckets like 0.25/0.25/0.25/0.25" to the prompt.
  This may bias toward extreme distributions. Decide for yourself
  whether to include this constraint.
- Consider whether to allow round-1 personas to make partial probability
  shifts (CONVINCED → 30 percentage point shift) or only full flips.
  This is a design choice, not a settled answer.

**Analysis**:

- Compute both letter-modal verdict (current method) and
  probability-pooled verdict from the same round-1 data. Are they
  the same letter? If not, which matches gold?
- Compare round-0 → round-1 probability mass shifts per letter.
- Compare to baseline-protocol round-0 and round-1 distributions in
  `eval/transcripts/experiment_v3/`.

---

### v_judge — impartial judge as aggregation rule

**Intent**: replace count/probability aggregation with a separate
impartial-judge sub-agent that reads all persona transcripts and
issues a structured verdict.

**Minimal prompt change** — after round 1 (or round 0), instead of
counting letters, dispatch a single judge sub-agent:

```
You are an impartial judge. N personas have argued for different
answers to a multiple-choice question. Read each persona's
reasoning and render a verdict.

Use whatever evaluation criteria you think appropriate, but state
your reasoning explicitly. Do not let majority count alone determine
your verdict.

Perspectives (Read each transcript via the Read tool):
- Persona 1: <path>
- Persona 2: <path>
... (N personas)

Question: <question text + choices>

End with: Verdict: <A|B|C|D>.
```

**Bias-leak callouts** — the originating session's first judge prompt
draft was much more elaborate, with steps like "Framing identification,"
"Framing applicability," "Operative-framing evaluation." Those steps
were a direct leak from the originating session's analysis of why the
methodology had failed on this specific question — telling the
judge how to reason in a way that essentially encoded the answer the
originating session had concluded was correct.

**The minimal version above is intentionally bare**. The fresh session
should consider whether structure should be added (and what kind), but
the structure should come from your own analysis of the data, not from
the originating session's hypothesis about what kind of failure to
target.

**Analysis**:

- Compare judge's verdict to baseline count-aggregation verdict on
  the same persona transcripts.
- Run the judge on (a) baseline-protocol round-0 transcripts (from
  `experiment_v3/`), (b) baseline-protocol round-1 transcripts, and
  (c) fresh round-0 + round-1 transcripts produced in this variant.
  Does the judge's verdict differ depending on which round it sees?
- The judge's reasoning is itself data — read several judge transcripts
  to see if the judge's verdicts cluster around a particular
  decision-rule pattern.

---

### v7 — pre-mortem / falsification-checklist

**Intent**: each persona pre-commits at round 0 to specific
falsification conditions. At round 1, "CONVINCED" requires citing a
specific pre-declared condition that the other persona's argument
satisfies.

**Minimal prompt change** — append to round-0 template:

```
After your reasoning and final answer, also report:
WOULD-CHANGE-MY-MIND: 1-3 specific conditions that, if shown to be
true, would make you change your answer.
```

Modify round-1 template's CONVINCED branch:

```
(c) CONVINCED — their argument specifically and clearly satisfies
    one of your declared falsification conditions. Quote which
    condition and explain how their argument meets it.
```

**Bias-leak callouts**:

- The originating session's first draft included examples like
  "if your answer rests on a contested framing assumption..." —
  this leaked from the originating session's analysis of one specific
  question's failure mode. The minimal version above doesn't include
  example types.
- Open question: should the prompt push back if a persona declares
  vague conditions ("if presented with a strong argument")? The
  minimal version doesn't enforce specificity. If the data shows
  personas declare lazy conditions, consider tightening — but only
  after seeing the data.

**Analysis**:

- Read the WOULD-CHANGE-MY-MIND declarations across all 30 round-0
  evals. Are they specific or vague?
- In round 1, when a persona marks CONVINCED, do they actually cite
  a pre-declared condition? Or do they capitulate without citing?
- Modal-answer comparison vs. baseline protocol.

---

### v9 — "bet on it" framing

**Intent**: behavioral reframing to sharpen probabilistic calibration.
Tests whether wagering language affects the persona's reported
probability or letter.

**Minimal prompt change** — add to the eval prompt (round 0 and/or
round 1):

```
You are wagering $100 on each letter at the probabilities you
will report. State only the probabilities and final answer
that reflect a real wager you'd make.
```

**Bias-leak callouts**:

- This is a behavioral framing common in superforecasting research,
  not specific to this methodology. Low risk of session-bias leak.
- Note: this change is most informative if combined with v5
  (probabilistic outputs). On its own (with letter-only outputs),
  the wagering frame may have minimal effect. The fresh session may
  want to either combine with v5 or skip until v5 results are in.

**Analysis**:

- Compare reported probabilities (or letter-confidence) between v5
  (no wager framing) and v9 (with wager framing) on otherwise-matched
  prompts. Does the wager framing produce more conservative or more
  extreme distributions?
- Modal-answer comparison vs. baseline protocol.

---

## Variants the originating session considered and skipped

Documenting these so the fresh session knows they were considered
and decided against (with reasons).

### Skipped: cascade avoidance via identifier-stripping + randomization

**Idea**: in round-N, strip persona identifiers from prior-round
transcripts and randomize their order, so the persona reading them
can't anchor on which-persona-said-what.

**Why skipped**: a participant in the originating discussion argued
this would likely homogenize the persona pool — turning the inputs
into a "soup of arguments" where the model's underlying belief
dominates because there's no diverse-perspective signal. The
prediction was that the variant would converge to the model's
zero-shot answer regardless of any persona-binding richness in
round 0.

**Caveat**: this prediction was not tested. The fresh session may
choose to test it if they want to verify the prediction, or skip
it on the same reasoning.

### Skipped: extremization in pooling

**Idea**: after probabilistic pooling, apply `p_final ∝ p_pooled^a`
with `a > 1` to extremize the aggregated distribution.

**Why skipped**: dependent on v5 producing useful probabilistic
output. Test only if v5 shows that probabilistic-pool aggregation
helps; otherwise extremization has nothing to extremize.

---

## Methodology hygiene for the fresh session

Three guardrails the originating session learned the hard way:

1. **Don't bake your hypothesis into prompts.** If you suspect
   the methodology fails because of X, do NOT write a prompt that
   pre-tells the personas/judge/decomposer to "watch out for X,"
   "consider whether X applies," or "vary along axes including X."
   Each of these forms encodes the answer rather than testing it.
   The decomposer's bias-leak warning in `PROTOCOL_v2.md` §3 walks
   through one specific instance: a draft that enumerated "wording
   skepticism" as a persona axis caused the decomposer to plant
   exactly the meta-persona the originating session expected to
   see, contaminating the experiment. **Read that section carefully
   before iterating on any prompt in this methodology** — the trap
   is subtle and reproducible. Write prompts as if you didn't know
   about your hypothesis. If you must include guidance, prefer
   abstract instructions ("orthogonal," "impartial," "rigorous")
   over concrete enumerations ("orthogonal across axes A, B, C").

2. **Tool returns come back in completion order, not dispatch
   order.** When mapping letters to personas, always Read the saved
   transcripts to verify rather than relying on the order the
   parallel sub-agent results came back in. (The originating session
   was incorrect about several persona/letter mappings until file-read
   verification caught it.)

3. **Sub-agents cannot recursively call Agent.** PROTOCOL_v2 §9.1.
   Plan dispatches accordingly; "delegate the whole experiment to a
   sub-agent" doesn't work.

---

## Status as of this handoff

- The pre-orthogonal-decomposer baseline (older PROTOCOL_v2 §3) was
  run on iter-2's gpqa_007 across N=4..8 and saved to
  `eval/transcripts/experiment_v1/`. Use as a contrast, not as
  current baseline.
- The current orthogonal-decomposer baseline was run on the same
  question across the same N range and saved to
  `eval/transcripts/experiment_v3/`. This is the data to compare new
  variants against.
- An intermediate experiment (`eval/transcripts/experiment_v2/`)
  used a contaminated decomposer prompt that hardcoded specific axes
  (e.g. "wording skepticism" listed explicitly as a selectable value).
  The contamination invalidated the test. Treat as not-evidence; the
  current §3 decomposer specifically warns against this kind of
  axis-enumeration leak.
- A partially-completed brainstorm-30 experiment is at
  `eval/transcripts/experiment_v4/` — only decomposer outputs exist
  (no round-0/round-1 evals). Usable if you want to compare
  brainstorm-30 personas to the current brainstorm-20 baseline at
  the persona-pool level only.

The fresh session should look at `experiment_v3/` transcripts as
primary baseline data. Prior interpretations from the originating
session live only in conversation history and should not be trusted
as ground truth.
