# Multi-Persona CoT — Pilot Evaluation v2 Protocol

This document supersedes Section 4 of `eval/README.md` based on lessons from
iteration 1. It describes the architecture, prompts, and run procedure that
the next iteration should follow.

---

## 1. Architecture: blinded planner + dumb transport

The orchestrator is decomposed into:

- **Main session** ("transport"): fires sub-agent calls, captures raw outputs,
  forwards to planners. Has access to gold answers but does no orchestration
  decisions and no prompt construction.
- **Blinded planners** (sonnet sub-agents): never see gold. Each cycle, a
  fresh planner is spawned with current state and asked to produce the next
  batch of `Agent(haiku|sonnet)` calls as JSON. The planner builds verbatim
  prompts; transport fires them; raw outputs flow back to the next planner
  cycle.
- **Eval sub-agents** (haiku or sonnet): never see gold. Run as `Agent(model="...", prompt="...")`.
  Produce reasoning + final-line `Answer: <A|B|C|D>`.

`SendMessage` is not exposed in this Claude Code environment, so the planner
is **stateless across cycles**: each cycle is a fresh planner instance that
receives the full state-so-far as part of its prompt.

### Why this matters

The original task brief had me orchestrate directly. In iteration 1, that
caused real leakage: when I summarized round-N transcripts to build round-N+1
prompts, my paraphrasing tracked toward the gold answer. Re-running with
blinded planners eliminated that vector — at the cost of more sub-agent
calls.

---

## 2. Models

- **Eval calls** (decomposer, personas, synthesizer): `model="sonnet"`.
  Iteration 1 tried haiku first; it produced off-script outputs (decomposers
  generated full debate roleplay instead of 4 persona blocks) and unreliable
  reasoning on graduate-level physics. Sonnet was clean.
- **Planners** (orchestration only): `model="sonnet"`. Need to be smart enough
  to parse outputs and build defensible prompts. Opus would also work.
- **Hold model constant within an iteration.** Mixing models within a run
  invalidates A-vs-B comparisons.

The user flagged that haiku-as-eval was not a fair test of the *methodology*
because the model was too weak; sonnet is the right floor for this benchmark.

---

## 3. Prompt templates (verbatim — do not edit)

Every eval prompt MUST end with:

> `Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).`

### Decomposer

The decomposer template uses a generate-and-select structure that
forces the model to consider a wide candidate pool and select for
orthogonality. This replaces an earlier "make perspectives genuinely
divergent" instruction that was found (in iter-2 retrospective
experiments) to produce homogeneous topical-only diversity. The
generate-and-select version measurably shifts round-0 distributions
on contested questions; see §9.11 for the change history.

```
You are a methodology decomposer. Your job: produce a roster of N
expert personas who will debate this problem from genuinely
ORTHOGONAL vantage points — not just different specialties, but
different ways of evaluating truth.

PROCESS:
1. First, generate 20 candidate personas. Span a wide range of
   perspectives: vary along whatever dimensions you think most
   matter for genuine intellectual divergence. Don't just vary
   topical specialty.

2. SELECT N personas from your 20 to MAXIMIZE PAIRWISE ORTHOGONALITY
   across whatever axes of variance you generated. Avoid selecting
   two personas redundant in their approach — minimize overlap on
   any single axis of variance.

3. Output ONLY the N selected PERSONA blocks. Do not include the
   candidate list, do not include selection rationale, do not
   include any intro or outro.

CRITICAL ANTI-LEAKAGE RULES:
- Persona descriptions must be ROLE-ONLY: a role title, three concerns, one trait line.
- Do NOT pre-solve the problem, derive any answer, perform any calculations, or hint at which option a persona will pick.
- Do NOT include numerical solutions, intermediate calculations, or solution commentary in any field.
- Do NOT reference or evaluate the answer choices (A, B, C, D) in the persona descriptions.
- Personas describe WHO debates, not WHAT they conclude.

Output format (only the N selected personas, no candidate list):

PERSONA N
ROLE: <role>
CONCERNS: <c1> | <c2> | <c3>
TRAITS: <one line>

Problem: <QUESTION TEXT — substitute verbatim>

[choices may be included if planner judges helpful, but anti-leakage rules above MUST be in the prompt]

Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

**Important — bias-leak warning when iterating on this prompt**:

The current prompt asks the decomposer to maximize orthogonality
"across whatever axes of variance you generated." It deliberately
does NOT enumerate what axes those should be. This is a load-bearing
design choice and easy to undo by accident — the seductive next move
is exactly the move that planted answers in an earlier draft.

**What "enumerating axes" looks like — and why it's a bias leak**:

Suppose you observe that the decomposer is producing personas with
too little epistemic diversity (e.g., all topical specialists, all
reasoning under the same epistemological framework). You may be
tempted to add specifics to the prompt — something like:

> *"Vary along: topical specialty, epistemological orientation
> (rigorist / empiricist / skeptic / pragmatist), question-reading
> style (literal / colloquial / adversarial-wording-skeptic),
> update orientation (only-on-error / weight-of-evidence /
> never-against-discipline)."*

This looks like good prompt engineering. It is in fact a bias leak.
The decomposer will faithfully produce personas matching each
enumerated value — including, in the example above, a "wording-
skeptic" or "adversarial-wording-skeptic" persona. If that persona
is exactly the kind of perspective whose surfacing you wanted to
test for, the experiment is contaminated: the persona exists
BECAUSE the prompt told the decomposer to create it, not because
orthogonality-reasoning surfaced it independently.

**The originating session learned this the hard way**. A draft of
this prompt enumerated "wording skepticism" and "adversarial-wording-
skeptic" as values for epistemological/question-reading axes. The
decomposer dutifully created Wording-Skeptic personas at every N.
Those personas' votes in round 0 looked like evidence that the
methodology had improved — but the test was invalid because the
"discovery" was just retrieval of what had been planted in the
prompt.

**How to iterate without enumerating axes**:

- Describe WHAT you want abstractly (orthogonality, divergence,
  distinct epistemic modes) without specifying HOW (which dimensions
  to vary, which categorical values to include).
- Trust the decomposer to identify the relevant axes for each
  question. Different questions may need different kinds of
  diversity; pre-specifying axes prevents that adaptive selection
  AND plants whatever the prompt-author thought mattered.
- If you suspect a specific kind of perspective is missing, the
  right test is comparison: run the prompt without enumeration vs.
  with enumeration on a question whose answer you don't know in
  advance, and check whether the enumeration improves accuracy or
  just reproduces what was enumerated.

**This warning generalizes to any prompt where you describe what
to look for**: judges, synthesizers, pre-mortem instructions,
round-N rebuttal templates. If you suspect failure-mode X, don't
write "watch out for X" or "consider whether X applies." Write the
prompt as if you didn't know about X, and see whether the
methodology surfaces it on its own.

### Round 0 (one per persona, parallel)

```
You are a {role}. Concerns: {concerns}. Traits: {traits}.

Answer this graduate-level multiple-choice question from your perspective. Give your reasoning, then on the final line output: `Answer: <A|B|C|D>`.

Question: <Q>
A) <A>
B) <B>
C) <C>
D) <D>

Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

### Debate round N ≥ 1 (one per persona, parallel)

`{TRANSCRIPT}` is verbatim concatenation of all prior rounds' raw outputs,
labeled `Round X, Persona Y ({role}):` then the raw text.

```
You are a {role}. Concerns: {concerns}. Traits: {traits}.

Below is the discussion across rounds 0..{N-1}:

{TRANSCRIPT}

{If N odd:  Propose your refined answer, addressing weaknesses you see in others' reasoning.}
{If N even: Critique each other persona's argument and update your answer if warranted.}

Question: <Q>
A) <A>
B) <B>
C) <C>
D) <D>

End with: `Answer: <A|B|C|D>`.
Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

### Synthesizer (conditional — see §4)

```
Given the full debate transcript below, determine the consensus (or best-supported) answer. Output a one-paragraph justification, then on the final line: `Answer: <A|B|C|D>`.

{FULL_TRANSCRIPT}

Question: <Q>
A) <A>
B) <B>
C) <C>
D) <D>

Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

---

## 4. Conditional synthesizer

The README v1 specifies "always run synthesizer." In iteration 1, every
question reached unanimous consensus, so the synthesizer was a rubber stamp —
9 redundant sonnet calls.

**v2 rule:**

```
After each round:
  if all 4 letters unanimous → final_answer = consensus_letter, DONE (skip synthesizer)
  elif r < 4                  → continue to round r+1
  else (r=4 and split)        → run synthesizer to break the tie
```

The synthesizer earns its keep only when 4 debate rounds end without
unanimity. Skip it when consensus is already reached.

---

## 5. Anti-leakage guards (lessons from iteration 1)

### 5.1 Planners must not call tools

Iteration 1's gpqa_004 cycle-2 planner ran Python to compute the answer
(`(3.5/2)^(-1/3) ≈ 0.83`), then announced "matches answer choice C" in its
preamble before emitting the JSON. Add to every planner prompt:

> `You will NOT call any tools. Output exactly ONE JSON object on a single line. Do not pre-solve the problem.`

Treat `tool_uses > 0` on a planner as a hard fail and re-run.

### 5.2 Decomposer must not pre-solve in persona traits

Iteration 1's gpqa_002 v1 decomposer wrote
*"...arrives at E_γ ≈ 2.6×10⁵ GeV, selecting option C"* in a persona's
TRAITS field. The v3 decomposer prompt (§3) prevents this — don't weaken the
"do NOT pre-solve" / "do NOT include numerical solutions" / "do NOT reference
or evaluate the answer choices" lines.

### 5.3 Eval prompts must explicitly forbid external tools

Without the no-tools line, sonnet sub-agents reach for Bash → Python → numpy
to verify arithmetic, generating cascading permission prompts (and changing
the eval — Condition A vs Condition B should have identical tool access).

### 5.4 Hold the eval prompt template fixed across A and B

Both conditions must use the same question/choice formatting and the same
no-tools line. Iteration 1's first attempt had A with no-tools and B
without, which made the comparison apples-to-oranges.

---

## 6. Question selection — no replacement across iterations

`csv_to_questions.py` v2 uses a ledger file
(`eval/used_csv_indices.txt`, gitignored) listing CSV row indices already
sampled in prior iterations. Each run:

1. Reads the ledger and excludes those indices.
2. Samples N questions from the remaining pool with the seed.
3. With `--update-ledger`, appends the new indices to the ledger.

Iteration 1 used CSV indices `[6, 26, 28, 35, 57, 62, 70, 163, 188, 189]`
(seed=42, n=10, sonnet eval). Those are pre-loaded into the ledger.

### Iteration 2 command

```bash
python3 eval/csv_to_questions.py \
    --csv eval/GPQA/gpqa_diamond.csv \
    --n 10 --seed 42 \
    --update-ledger
```

This will sample 10 *new* questions (the 188 not yet used) and append them to
the ledger. Verified by test in `csv_to_questions.py`'s output: `Pool: 198
total rows, 10 used, 188 available. Sampling 10.`

---

## 7. Run procedure for next session (fresh context)

The next iteration should start a brand-new Claude Code session to ensure no
gold-answer contamination from prior runs. Steps:

1. **Confirm clean state**:
   ```bash
   git status                          # working tree clean
   git branch --show-current           # if on `main`, branch off: git checkout -b iteration-N
                                       # if on a `claude/...` auto-branch, you're already set
   ls eval/transcripts/ | head         # iteration-1 transcripts present
   wc -l eval/used_csv_indices.txt     # ledger has prior indices
   ```

2. **Sample new questions**:
   ```bash
   python3 eval/csv_to_questions.py --csv eval/GPQA/gpqa_diamond.csv \
       --n 10 --seed 42 --update-ledger
   ```
   This writes `eval/pilot_questions.jsonl` (gitignored) and appends to
   `eval/used_csv_indices.txt`.

3. **Run Condition A**: 10 sonnet calls in parallel from the main session,
   one per question, using the round-0 prompt template (no persona —
   pure zero-shot CoT). Save each to
   `eval/transcripts/{id}_A.txt`.

4. **Run Condition B per question** using the planner-as-orchestrator pattern:
   - Cycle-1 planner builds decomposer prompt (with anti-leakage guards from §5.2)
   - Decomposer call → save to `{id}_B_decomposer.txt`
   - Cycle-2 planner parses 4 personas, builds round-0 prompts
   - Round-0 calls (4 in parallel) → save to `{id}_B_round0_p{1..4}.txt`
   - Cycle-3 planner: if unanimous, jump to step 5; else build round-1 prompts
   - Continue until unanimous OR round 4 ends
   - If round 4 ends without consensus: run synthesizer

5. **Grade**: build `eval/pilot_results.jsonl` (gitignored) with one row per
   question matching A and B answers against gold from
   `pilot_questions.jsonl`'s `answer` field.

6. **Append to summary.md** under "Iteration 2" — aggregate accuracy, flip
   cases by ID, debate-rounds-used distribution.

7. **Commit**: `summary.md`, `PROTOCOL_v2.md` (if updated),
   `csv_to_questions.py` (if updated), `used_csv_indices.txt` is **gitignored**
   (do not commit — contains question identifiers via row indices).

---

## 8. Known issues and future improvements

- **Permission cascade**: every `Agent` call generates a permission prompt
  unless pre-approved. A 10-question B run = 100–250+ prompts. For
  unattended runs, pre-approve `Agent` and `Bash` (for the script) in
  `/permissions`.
- **Planner instructions could be smarter**: iteration 1's planners didn't
  spontaneously identify methodology issues. They added anti-leakage guards
  only when explicitly told to. A truly autonomous planner would design
  defensively without prompting. Worth experimenting with a meta-instruction:
  *"You are responsible for methodology quality. Identify and prevent any
  leakage risks you anticipate, without being prescribed the specific guard."*
- **Variance check protocol** (resolved in iter-2 amendments §10): iter-1
  sampled A 10× post-hoc on the contested case; iter-2 adopted k=4 inline,
  fired alongside the rest of A. Iter-3 should run k=4 on every question to
  build a baseline distribution, not just on contested ones.
- **Synthesizer scope**: even when triggered (no consensus after 4 rounds),
  the synthesizer's job is ill-defined. Better tiebreaker: weighted majority
  vote with persona-confidence elicited in their final round.

---

## 9. Iteration-2 amendments (apply to all subsequent iterations)

These amendments codify patterns the iter-2 session arrived at after
several mid-run course corrections. They override or extend earlier sections
where noted. Treat them as part of the protocol going forward.

### 9.1 Hard architectural constraint: sub-agents cannot fan out

The Claude Code `Agent` tool is only available to the **main session**.
Sub-agents (any model: sonnet, opus, haiku) cannot recursively call `Agent`.
They can use Read, Write, Bash, etc. — but they cannot dispatch their own
sub-agents.

Implications:
- Don't try to delegate a "per-question end-to-end orchestrator" sub-agent
  that itself fires decomposer/persona/synthesizer calls. It will fail.
- The blinded planner pattern from §1 is the right shape: planner sub-agents
  return JSON describing the next batch of `Agent` calls; main session
  is the dumb transport that fires them.
- For long fan-outs (40 round-0 evals, etc.), main session must dispatch
  in batches of parallel tool_uses within single messages. Iter-2 confirmed
  20 parallel `Agent` calls per message works reliably.

### 9.2 Sub-agent self-save pattern (massive context-cost reduction)

**The single most impactful pattern this iteration.** Each eval sub-agent is
instructed to save its full reasoning to disk via Write before returning,
and to return ONLY `Answer: <letter>` (or for decomposers, just the persona
blocks) to main. This collapses the main-context cost of a 40-eval round
from ~40,000 tokens of reasoning text down to ~400 tokens of letters.

**Tool-permission rule for eval sub-agents going forward:**
- Reasoning phase: NO tools (no Bash, Python, web, Read, anything).
- After reasoning is complete: Write tool ONLY, used exactly once, to save
  the full response (including the final-line `Answer: <letter>`) to a
  pre-specified path.
- Final reply text to main: just `Answer: <letter>` — no preamble, no
  commentary, no echo of reasoning.

The §3 round-0 and round-N templates should be amended with this addendum
for every eval call. Pre-iter-2 templates (without the save instruction)
are obsolete; do not use them.

### 9.3 Planner Read pattern (avoids main-session contamination)

**Round-N+1 planners should Read prior-round transcripts themselves** via
the Read tool, rather than receiving them inline-embedded in the planner
prompt by main session. This prevents two failure modes seen in iter-2:

1. **Cycle-N planner re-emitting transcripts**: when a planner re-types a
   prior-round transcript into its output prompt, it can introduce typos
   and other transcription drift. (Iter-2 cycle-3 planner introduced one
   benign typo this way before the pattern was introduced.)
2. **Main session paraphrasing**: when main reads a transcript and writes a
   summary of it into a prompt, that's main-session prompt construction —
   exactly what §1 forbids. (Iter-2 main session did this once before
   reverting.)

**Tool-permission rule for planner sub-agents going forward:**
- Read tool ONLY (to fetch prior-round transcripts and decomposer output
  from disk).
- No Bash, Python, web search, Write — and no inline tool use that performs
  computation. Read is just retrieval.
- Returns single-line JSON describing the next batch of `Agent` calls.

Round-N+1 prompts that planners build should typically NOT inline
prior-round transcripts. Instead, they should provide the round-N+1 eval
sub-agent with file paths, and instruct that sub-agent to Read the
transcripts itself (the eval sub-agent gets Read access for this round
only, scoped to the named transcript paths).

### 9.4 Opus auditor after each round

Because the self-save pattern means main only ever sees `Answer: <letter>`,
main cannot directly verify reasoning quality. After each round of evals
fires, dispatch an opus sub-agent with Read access to the saved transcripts
and ask it to spot-check:

1. Each file exists and is non-empty.
2. Each contains substantive reasoning (not just the letter).
3. The final-line `Answer: <letter>` matches what main session received.
4. Each persona retained its assigned identity (no persona drift / generic
   reasoning).
5. No tool-misuse signals (mentions of "I ran Python", "I called Bash",
   "according to <URL>", etc. — should be absent because eval prompts
   forbid those tools).
6. No leakage signals (the prompt giving away the gold answer, planner
   pre-solving, etc.).
7. No refusal/error/truncation patterns.

For contested questions, also ask the auditor to describe each persona's
reasoning frame in 1 sentence (without revealing what it thinks the
correct answer is).

**Tool-permission rule for auditor sub-agents:** Read only. No Write.

### 9.5 Opus blinded grader at end (replaces "main reads gold")

The original protocol said main session reads gold answers from
`pilot_questions.jsonl` only at the very end, when grading. Iter-2
amended this further: **the parent session never reads gold at all.**

Instead, dispatch an opus sub-agent with:
- The path to `pilot_questions.jsonl` (the only place gold lives).
- A small dict mapping each id to its observed A_answer, B_answer,
  B_rounds_used, B_consensus_reached, B_synthesizer_used, and (for
  contested ids) A_variance_check letters.

The grader Reads `pilot_questions.jsonl`, computes A_correct / B_correct
per row, writes `eval/pilot_results.jsonl`, and returns to main:
- Headline accuracy: A=X/N, B=Y/N
- Agreement rate
- Flip-case list by id (A letter, B letter, gold letter, B_correct?)
- Variance-check breakdown for any contested id

The grader's return must NOT echo the question text or gold letters of
non-contested ids back to main. Just numeric counts and contested-id
specifics. This keeps main fully blinded even at grading time.

**Tool-permission rule for the grader sub-agent:** Read + Write only.
Read for `pilot_questions.jsonl`; Write for `pilot_results.jsonl`.

### 9.6 Opus leak-checker before public commit

The dataset terms forbid GPQA verbatim text in public commits. Even when
main session has been blinded, summary text it writes can still slip
identifying information through topic-specific vocabulary, unique
numerical values, near-verbatim question fragments, or framing
descriptions specific enough to fingerprint the question.

**Before any `git commit` of `summary.md`** (or any other file that
discusses contested cases), dispatch an opus leak-checker sub-agent with:
- Path to the staged file (e.g. `eval/summary.md`).
- Path to `pilot_questions.jsonl` (the source-of-truth).

The leak-checker Reads both, cross-checks the staged file for:
- Verbatim ≥8-word substrings of any question or choice text.
- Unique numerical values from the questions (energies, masses, lengths,
  weights, etc. — even rounded versions).
- Distinctive named entities (compounds, isotopes, named equations).
- Topic-specific vocabulary that fingerprints the question (named
  techniques, named effects, named regimes).

Returns either "CLEAN — safe to commit" or a per-issue redaction list.
Iterate redactions + recheck until clean before committing.

**Tool-permission rule for the leak-checker:** Read only. No Write
(it must not auto-redact; redactions are reviewed by main session).

### 9.7 Contamination patterns to actively prevent

Before any commit, scan the staged `summary.md` for these classes of
contamination (the leak-checker will catch them, but writing carefully
the first time is cheaper):

- **Verbatim numerical signatures.** Even a rounded version of a unique
  question value (rounded to 3 sig figs from a 5-sig-fig source) can
  uniquely fingerprint the question against the public dataset. Use
  generic placeholders ("the stated <quantity>") instead.
- **Topic-specific framing names.** Names of physical effects, named
  regimes, and named techniques that appear in the question or are
  obviously implied by it can fingerprint the question. Abstract to
  "framing α / framing β" with no semantic vocabulary.
- **Persona-name reveals tied to topic.** Don't list persona roles when
  describing a contested-case round trajectory if the persona role names
  are topic-specific (e.g. a sub-discipline label that points at the
  exact phenomenon the question concerns). Use generic groupings at most
  ("two from theoretical backgrounds defaulted to framing α").
- **Convince-mechanism quotes.** Don't include direct quotes from
  persona reasoning, even paraphrased — they're the densest source of
  question content. Describe the rhetorical move abstractly.
- **Choice-text reveals.** When showing per-question outcomes, the
  letter is fine but the underlying choice text is not. Don't write
  "gold = X (<choice word>)"; write just "gold = X".

### 9.8 Iteration cleanup at start

At the start of each iteration, archive the previous iteration's
transcript files into a subdirectory:

```bash
mkdir -p eval/transcripts/iteration_<N-1>
mv eval/transcripts/gpqa_*.txt eval/transcripts/iteration_<N-1>/
```

Reason: the iter-N question pool is sampled fresh, but the gpqa_XXX id
naming restarts from 0 each iteration. Without archiving, iter-N
transcripts collide with iter-(N-1) transcripts at the same paths.
Iter-2 hit this and had to archive mid-run.

### 9.9 Tool-permission summary (per role)

| Role | Read | Write | Bash | Other tools |
|---|---|---|---|---|
| Eval sub-agent (decomposer, persona, synthesizer) | ❌ | ✅ once after reasoning | ❌ | ❌ |
| Planner sub-agent | ✅ (transcripts only) | ❌ | ❌ | ❌ |
| Auditor sub-agent | ✅ | ❌ | ❌ | ❌ |
| Grader sub-agent | ✅ (`pilot_questions.jsonl`) | ✅ (`pilot_results.jsonl`) | ❌ | ❌ |
| Leak-checker sub-agent | ✅ | ❌ | ❌ | ❌ |
| Variance trial (= zero-shot CoT eval) | ❌ | ✅ once after reasoning | ❌ | ❌ |
| Main session | only when unavoidable | ✅ | ✅ (script invocation, file ops) | Agent (only main has it) |

Round-N+1 evals (debate continuations) are an exception: they get Read
access scoped to specific round-N transcript paths, plus Write at the end.
This is the planner-Read pattern (§9.3) — the planner builds a prompt
that gives the eval sub-agent the file paths to Read.

### 9.10 Question splitter (proposed for iter-3+, not yet tested)

Iter-2's main session read `pilot_questions_blinded.jsonl` (a stripped
version of `pilot_questions.jsonl` with the `answer` field removed) once
to extract id+question+choices for prompt construction. That works but
ingests all 10 question texts into main context.

For iter-3+, consider a "question splitter" sub-agent at the very start:
Reads `pilot_questions.jsonl`, writes 10 per-id files like
`eval/transcripts/{id}_question.txt` containing question + choices but
NOT the answer field. Then cycle-1 planners get the path to their target
question file, Read it themselves, and build the decomposer prompt.

This eliminates main-session ingestion of question texts entirely. The
splitter sub-agent is Read+Write, blinded against persisting the gold
field (it can see gold while it filters but must not Write it anywhere).

Untested in iter-2; iter-3 should validate that planners can correctly
substitute Read'd question text into the decomposer template.

### 9.11 Decomposer change: orthogonal generate-and-select

Post-iter-2 retrospective experiments on a contested case found that
the original "make perspectives genuinely divergent" instruction in §3
produced personas that varied by topical specialty but not by epistemic
mode — they were all chemistry/physics personas with different sub-fields,
all reasoning under the same epistemological framework. Replacing the
instruction with a generate-and-select-orthogonally structure (generate
20 candidates, select N to maximize pairwise orthogonality) produced
qualitatively different rosters that included philosopher-of-science,
information-theoretic, Bayesian-epistemologist, and similar meta-personas
alongside the topical specialists. On the contested case the round-0
distribution shifted substantially toward the gold-aligned answer
without affecting questions where round 0 was already unanimous.

The §3 decomposer template has been updated to the orthogonal version.
**Do not enumerate specific axes** (epistemological / question-reading /
update-orientation, etc.) inside the prompt — see the bias-leak warning
in §3. Keep the axes of variance for the decomposer to discover.

Open questions left for further experimentation: whether brainstorm-30
or larger candidate pool produces meaningfully better orthogonal
selection vs brainstorm-20; whether per-persona confidence elicitation
or probabilistic outputs interact usefully with the orthogonal pool.
See `eval/methodology_experiments.md` for variants currently under
investigation.
