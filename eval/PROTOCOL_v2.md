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

```
Generate exactly 4 distinct expert personas to debate this problem. For each, give: ROLE, CONCERNS (3 items separated by ' | '), TRAITS (one line). Make the perspectives genuinely divergent.

CRITICAL ANTI-LEAKAGE RULES:
- Persona descriptions must be ROLE-ONLY: a role title, three concerns, one trait line.
- Do NOT pre-solve the problem, derive any answer, perform any calculations, or hint at which option a persona will pick.
- Do NOT include numerical solutions, intermediate calculations, or solution commentary in any field.
- Do NOT reference or evaluate the answer choices (A, B, C, D) in the persona descriptions.
- Personas describe WHO debates, not WHAT they conclude.

Output ONLY the 4 PERSONA blocks in this exact format. No introduction, no extra commentary, no debate roleplay, no concluding remarks.

PERSONA N
ROLE: <role>
CONCERNS: <c1> | <c2> | <c3>
TRAITS: <one line>

Problem: <QUESTION TEXT — substitute verbatim>

[choices may be included if planner judges helpful, but anti-leakage rules above MUST be in the prompt]

Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

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
   git status                          # should be on the eval branch
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
- **Variance check protocol**: iteration 1 found that A's accuracy on
  gpqa_006 was ~80% across 10 trials despite being correct in our single
  observation. Future iterations should sample A multiple times per question
  (k≥3) when A's first answer disagrees with B's consensus, to distinguish
  signal from variance.
- **Synthesizer scope**: even when triggered (no consensus after 4 rounds),
  the synthesizer's job is ill-defined. Better tiebreaker: weighted majority
  vote with persona-confidence elicited in their final round.
