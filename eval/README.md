# Multi-Persona CoT — Pilot Evaluation Handoff

This folder contains everything needed to run a **10-question sanity-check
pilot** of the multi-persona chain-of-thought hypothesis. It is designed to be
picked up and executed by a fresh Claude Code session on a computer.

---

## 1. Background (read first)

The repo (`Multi_Persona_Problem_Solver.py`) implements a framework that:
1. Takes a problem statement.
2. Auto-generates 3–6 expert "personas" via a decomposer LLM call.
3. Runs an orchestrated multi-round debate (opening → propose/critique loop →
   consensus check → synthesis) until consensus or a round cap.

**Hypothesis under test:** seeding *divergent* perspectives and letting them
debate produces better answers than a single-agent zero-shot CoT, especially
on problems where the dominant intuitive frame is wrong (e.g. the canonical
tire-pressure example in the repo's main README).

**What this pilot is:** a cheap, low-N sanity check to see if the hypothesis
shows *any* signal before investing in a full eval. It is **not** a rigorous
benchmark — N=10 is far too small for stat sig. Treat the result as a
direction indicator, not a verdict.

**What a real eval would look like (out of scope for this pilot):**
- N≈100 GPQA Diamond + MUSR, paired across 4 conditions (zero-shot CoT,
  self-consistency K=5, single-prompt multi-persona, full orchestrator),
  McNemar's test, paired bootstrap CIs, per-token cost-effectiveness curves.
- See conversation history with the repo owner for the full design.

---

## 2. Pilot design

**Two conditions, same 10 questions, same base model (Claude via sub-agents).**

### Condition A — Zero-shot CoT
- **One** sub-agent per question.
- Prompt: present the question + 4 choices, instruct "think step by step,
  then output `Answer: <letter>` on the final line."
- Record: chosen letter, full reasoning.

### Condition B — Multi-persona debate (faithful sub-agent simulation)
For each question:
1. **Decomposer call** (1 sub-agent): given the problem, output 4 distinct
   expert personas (role + concerns + traits) — mirrors
   `ProblemDecomposer.analyze_problem` at line 1209 of the main script.
2. **Round 0 (opening)**: spawn 4 sub-agents in parallel, one per persona.
   Each gets {problem, choices, its persona}, returns initial answer +
   reasoning.
3. **Rounds 1..N (debate, up to 4 rounds)**: in each round, spawn 4
   sub-agents in parallel. Each gets {problem, choices, its persona, full
   transcript of all prior rounds} and returns a refined answer +
   critique of others. Alternate "proposal" and "critique" framings to
   match the orchestrator at line 1384.
4. **Consensus check** after each round: extract each persona's current
   letter answer; if all 4 agree, exit the loop early.
5. **Synthesizer (1 sub-agent)**: gets the full transcript, outputs
   `Answer: <letter>` plus a short justification.

**Why faithful sub-agent simulation works despite single-turn sub-agents:**
each round spawns *fresh* sub-agents that are handed the prior transcript in
their prompt, so multi-round dynamics are preserved. Rounds within a
condition can be run in parallel across personas; questions can be run in
parallel across the whole batch.

---

## 3. Privacy / terms (IMPORTANT)

GPQA's dataset terms require that the questions **not be exposed publicly** in
a way that could be crawled into training data. This repo is public.

- `eval/gpqa_diamond.csv`, `eval/pilot_questions.jsonl`,
  `eval/pilot_results.jsonl`, and `eval/transcripts/` are gitignored.
- **Do NOT commit any file containing verbatim question text.**
- Aggregate results (accuracy numbers, flip counts, domain breakdown) are
  fine to commit. Per-question transcripts must stay local.
- Passing questions to Claude via sub-agents / API is acceptable — Anthropic
  doesn't train on API traffic by default, so this isn't "exposure" under the
  GPQA terms.

---

## 4. Runbook (for the user, on a computer)

```bash
# 0. From the repo root, on branch claude/eval-multi-perspective-agents-X8tvr.
git checkout claude/eval-multi-perspective-agents-X8tvr
git pull

# 1. Place the downloaded GPQA Diamond CSV here:
#    eval/gpqa_diamond.csv
#    (Get it from https://huggingface.co/datasets/Idavidrein/gpqa after
#     accepting the gated-access terms, OR from the password-protected zip
#     in https://github.com/idavidrein/gpqa.)

# 2. Convert CSV -> JSONL (10 questions, deterministic seed):
python eval/csv_to_questions.py --n 10 --seed 42
# This writes eval/pilot_questions.jsonl (gitignored).

# 3. In a new Claude Code session in this repo, paste the prompt from
#    Section 5 below to kick off the pilot.
```

---

## 5. Session prompt (paste into the new Claude Code session)

> I'm continuing a pilot eval from `eval/README.md`. The questions file is at
> `eval/pilot_questions.jsonl` (10 GPQA Diamond MCQs, gitignored). Please run
> the two-condition pilot exactly as described in Section 2 of that README,
> using sub-agents (general-purpose) to simulate both conditions:
>
> 1. For each of the 10 questions, run **Condition A (zero-shot CoT)** by
>    spawning one sub-agent. Then run **Condition B (multi-persona debate)**
>    by spawning a decomposer sub-agent, then 4 persona sub-agents per round
>    (in parallel within a round), up to 4 rounds with early exit on
>    consensus, then a synthesizer sub-agent.
> 2. Parse `Answer: <letter>` from each condition's final output and grade
>    against the `answer` field in the JSONL.
> 3. Write per-question raw outputs to `eval/transcripts/<id>.json`
>    (gitignored). Write a row per question to `eval/pilot_results.jsonl`
>    (gitignored) with: id, domain, gold, A_answer, A_correct, B_answer,
>    B_correct, B_rounds_used, B_consensus_reached.
> 4. Print a summary table: A accuracy, B accuracy, agreement rate, and the
>    list of question IDs where A and B disagreed (with which one was
>    correct). Also commit a short `eval/summary.md` with **only**
>    aggregate numbers — no verbatim question text.
>
> Re-read `eval/README.md` for full context before starting. Pre-registered
> analysis plan: report raw counts; do not compute p-values for N=10.
> Highlight any "flip" cases (A wrong, B right, or vice versa) for me to
> spot-check, but do so by ID and the orchestrator's persona names only —
> not by quoting the question.

---

## 6. What "good" looks like

Because N=10, no single number will be conclusive. Things to look for:

| Signal | Interpretation |
|---|---|
| B beats A by ≥3 questions, with flips concentrated in "trap-style" items | Encouraging; worth scaling to N=100 GPQA Diamond. |
| A beats B by ≥2 | Suggests debate flattens correct outliers — investigate transcripts before scaling. |
| A == B, low disagreement | Technique adds cost without benefit on Claude; might still help on smaller models. |
| A == B, high disagreement (lots of flips both ways) | Technique is noisy, not directional — needs design changes (e.g. better consensus rule). |

Whatever the result, the deliverable from this pilot is a go/no-go signal
on running the full N=100 paired eval, not a final verdict on the technique.

---

## 7. Files in this folder

| File | Committed? | Purpose |
|---|---|---|
| `README.md` | yes | This handoff doc. |
| `csv_to_questions.py` | yes | Converts downloaded GPQA CSV → JSONL. |
| `gpqa_diamond.csv` | **no** (gitignored) | The downloaded dataset. |
| `pilot_questions.jsonl` | **no** (gitignored) | 10 sampled questions in pilot format. |
| `pilot_results.jsonl` | **no** (gitignored) | Per-question grading rows. |
| `transcripts/` | **no** (gitignored) | Per-question raw sub-agent outputs. |
| `summary.md` | yes (after run) | Aggregate-only results, safe to commit. |
