# Multi-Persona CoT — Pilot Evaluation

> **If you are a Claude Code session reading this in iteration 2 or later:**
> this file is the original v1 task brief, kept for historical context. It
> describes the simpler architecture used in iter-1 before the v2 protocol
> introduced blinded planners and the iter-2 amendments codified the
> sub-agent self-save / planner Read / opus auditor / opus blinded grader /
> opus leak-checker patterns.
>
> **For your iteration, your task brief is `eval/NEXT_ITERATION.md`** and
> the methodology is **`eval/PROTOCOL_v2.md`** (especially the §9
> amendments — they're mandatory for iter-3+). Prior-iteration results are
> in **`eval/summary.md`**. Read those three files before proceeding.
>
> ---
>
> **If this is iteration 1 (no prior runs):** this file is your task
> brief. Read all of it, then execute Section 4 (Runbook) end-to-end —
> including running `python eval/csv_to_questions.py` yourself to prep the
> question file. The user's only manual step is dropping
> `eval/gpqa_diamond.csv` in place; you do everything after that. Do not
> modify the design without checking with the user.

---

## 1. Background

The repo (`Multi_Persona_Problem_Solver.py`) implements a framework that:
1. Takes a problem statement.
2. Auto-generates 3–6 expert "personas" via a decomposer LLM call
   (`ProblemDecomposer.analyze_problem`, line 1209).
3. Runs an orchestrated multi-round debate — opening → propose/critique loop
   → consensus check → synthesis (`DebateOrchestrator`, line 1324).

**Hypothesis under test:** seeding *divergent* perspectives and letting them
debate produces better answers than a single-agent zero-shot CoT, especially
on problems where the dominant intuitive frame is wrong (the canonical
tire-pressure example in the repo's main README is the motivating case).

**What this pilot is:** a low-N sanity check (N=10) to see if the hypothesis
shows *any* signal before investing in a full eval. Not stat-sig. Treat the
result as a direction indicator, not a verdict.

**What a real eval would look like (out of scope here):** N≈100 GPQA Diamond +
MUSR, paired across 4 conditions (zero-shot CoT, self-consistency K=5,
single-prompt multi-persona, full orchestrator), McNemar's test, paired
bootstrap CIs, per-token cost-effectiveness curves.

---

## 2. Pilot design — two conditions, same 10 questions

Base model = Claude (via sub-agents in this session). Same questions hit both
conditions for paired comparison.

### Condition A — Zero-shot CoT
- **One** sub-agent per question (`general-purpose`).
- Prompt: present the question + 4 choices, instruct "think step by step,
  then output `Answer: <letter>` on the final line."
- Record: chosen letter, full reasoning.

### Condition B — Multi-persona debate (faithful sub-agent simulation)
For each question:
1. **Decomposer** (1 sub-agent): given the problem, output 4 distinct expert
   personas (role + concerns + traits) — mirrors
   `ProblemDecomposer.analyze_problem` at line 1209.
2. **Round 0 (opening)**: 4 sub-agents in parallel, one per persona. Each
   gets {problem, choices, its persona}, returns initial answer + reasoning.
3. **Rounds 1..N (debate, up to 4 rounds)**: in each round, 4 sub-agents in
   parallel. Each gets {problem, choices, its persona, full transcript of
   prior rounds} and returns a refined answer + critique. Alternate
   "proposal" and "critique" framings to match the orchestrator at line 1384.
4. **Consensus check** after each round: extract each persona's current
   letter answer; if all 4 agree, exit early.
5. **Synthesizer** (1 sub-agent): given the full transcript, outputs
   `Answer: <letter>` plus a short justification.

Sub-agents are single-turn but multi-round dynamics are preserved by passing
the prior transcript into each round's prompt. Run questions in parallel
across the batch and personas in parallel within a round wherever possible.

---

## 3. Privacy / dataset terms (CRITICAL)

GPQA terms forbid exposing the questions publicly in a way that could be
crawled into training data. **This repo is public.**

- `eval/gpqa_diamond.csv`, `eval/pilot_questions.jsonl`,
  `eval/pilot_results.jsonl`, and `eval/transcripts/` are gitignored.
- **Do NOT commit any file containing verbatim question text.**
- Aggregate results (accuracy numbers, flip counts, domain breakdown) are
  fine to commit. Per-question transcripts must stay local.
- Passing questions to Claude via sub-agents / API is acceptable —
  Anthropic doesn't train on API traffic by default, so this isn't
  "exposure" under the GPQA terms.
- When summarizing or discussing results in committed files (or chat with
  the user when they may copy it elsewhere), refer to questions by **id
  only** — never quote the question text.

---

## 4. Runbook — execute these steps in order

### 4a. Pre-flight and dataset prep (you do this, not the user)
1. Confirm you are on branch `claude/eval-multi-perspective-agents-X8tvr`
   (`git branch --show-current`).
2. Confirm `.gitignore` already covers the question/result/transcript paths
   (see Section 7). If not, stop and fix that before doing anything else.
3. Convert the CSV the user has placed at `eval/gpqa_diamond.csv` into the
   pilot JSONL by running the script directly via the Bash tool:
   ```
   python eval/csv_to_questions.py --n 10 --seed 42
   ```
   This writes `eval/pilot_questions.jsonl` (gitignored). Verify the file
   exists and has 10 lines before proceeding. If `eval/gpqa_diamond.csv`
   does not exist, stop and ask the user to drop it in.
4. Create `eval/transcripts/` if it doesn't exist. (It's gitignored.)

### 4b. Run Condition A (zero-shot CoT) for all 10 questions
For each question in `eval/pilot_questions.jsonl`:
- Spawn one `general-purpose` sub-agent with this prompt template:

  > You are answering a graduate-level multiple-choice question. Think
  > step by step, then on the final line output exactly:
  > `Answer: <A|B|C|D>`
  >
  > Question: {question}
  > A) {A}
  > B) {B}
  > C) {C}
  > D) {D}

- Parse the final line for the letter. Save the full reasoning to
  `eval/transcripts/{id}_A.json`.

You can run all 10 question-A calls in parallel (10 sub-agent tool calls in
a single message).

### 4c. Run Condition B (multi-persona debate) for all 10 questions
For each question, in this order (per-question dependencies are sequential
but you can interleave across questions where parallelism is safe):

1. **Decomposer call** — one sub-agent, prompt:

   > Generate exactly 4 distinct expert personas to debate this problem.
   > For each, give: ROLE, CONCERNS (3 items), TRAITS (one line). Make the
   > perspectives genuinely divergent. Problem: {question}

   Parse the 4 personas. If parsing fails, fall back to: Mechanical
   Engineer, Theoretical Physicist, Mathematician (rigor-focused),
   Contrarian First-Principles Thinker.

2. **Round 0 (opening)** — 4 sub-agents in parallel, prompt for each:

   > You are a {role}. Concerns: {concerns}. Traits: {traits}. Answer this
   > graduate-level MCQ from your perspective. Give your reasoning, then
   > on the final line output: `Answer: <A|B|C|D>`.
   >
   > Question + choices.

3. **Debate loop, rounds 1..4**: in each round, 4 sub-agents in parallel,
   each prompt:

   > You are {role}. Below is the discussion so far across rounds 0..{r-1}.
   > {transcript}
   >
   > {If round is odd: "Propose your refined answer, addressing weaknesses
   > you see in others' reasoning."
   > If round is even: "Critique each other persona's argument and update
   > your answer if warranted."}
   >
   > End with: `Answer: <A|B|C|D>`.

   After each round, parse all 4 letters. If unanimous, exit the loop.

4. **Synthesizer** — one sub-agent:

   > Given the full debate transcript below, determine the consensus (or
   > best-supported) answer. Output a one-paragraph justification, then on
   > the final line: `Answer: <A|B|C|D>`.
   >
   > {full transcript}

Save the full per-round transcript and the synthesizer output to
`eval/transcripts/{id}_B.json`.

### 4d. Grade and write results
For each question, append a row to `eval/pilot_results.jsonl`:

```json
{"id": "...", "domain": "...", "gold": "C",
 "A_answer": "B", "A_correct": false,
 "B_answer": "C", "B_correct": true,
 "B_rounds_used": 3, "B_consensus_reached": true}
```

### 4e. Summarize and commit
1. Compute: A accuracy, B accuracy, agreement rate (A==B), and the list of
   IDs where A and B disagreed (note which was correct).
2. Write `eval/summary.md` with **aggregate numbers and IDs only — no
   verbatim question text**. Include the table from Section 5.
3. Commit `eval/summary.md` only. Confirm via `git status` that no
   gitignored files are staged. Push.
4. Report the table to the user in chat, plus a short list of flip cases by
   ID with the *persona names* involved (no question text), so they can
   spot-check transcripts locally.

---

## 5. Interpretation guide (report this to the user)

Because N=10, no single number is conclusive. Frame the result as one of:

| Signal | Interpretation |
|---|---|
| B beats A by ≥3, flips concentrated in trap-style items | Encouraging — worth scaling to N=100 GPQA Diamond. |
| A beats B by ≥2 | Suggests debate flattens correct outliers. Investigate transcripts before scaling. |
| A == B, low disagreement | Technique adds cost without benefit on Claude. Might still help on weaker models. |
| A == B, high disagreement | Noisy, not directional. Needs design changes (e.g. better consensus rule). |

The deliverable is a **go / no-go signal on running the full N=100 paired
eval**, not a final verdict on the technique. Do not compute p-values for
N=10.

---

## 6. Things that will go wrong — handle gracefully

- **Letter not on the final line.** Fall back to a regex over the whole
  response for `Answer:\s*([A-D])`. If still nothing, mark `A_answer:
  "PARSE_FAIL"` and continue (don't crash the batch).
- **Decomposer returns junk.** Use the fallback persona set in Section 4c
  step 1 and note it in the transcript.
- **Sub-agent times out or errors.** Retry once. If it fails again, mark
  that round/condition as `ERROR` in the result row and continue.
- **All 4 personas already agree at round 0.** That's a valid early exit;
  record `B_rounds_used: 0`.

---

## 7. Files in this folder

| File | Committed? | Purpose |
|---|---|---|
| `README.md` | yes | This handoff doc / task brief. |
| `csv_to_questions.py` | yes | Converts downloaded GPQA CSV → JSONL. |
| `gpqa_diamond.csv` | **no** (gitignored) | The downloaded dataset. |
| `pilot_questions.jsonl` | **no** (gitignored) | 10 sampled questions in pilot format. |
| `pilot_results.jsonl` | **no** (gitignored) | Per-question grading rows. |
| `transcripts/` | **no** (gitignored) | Per-question raw sub-agent outputs. |
| `summary.md` | yes (after run) | Aggregate-only results, safe to commit. |
