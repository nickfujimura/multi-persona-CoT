# Task brief for the next Claude Code session

> **If you are a Claude Code session reading this:** this file is your task
> brief. Read all of it, then execute Section 4 (Runbook) end-to-end. Read
> `eval/PROTOCOL_v2.md` for the architecture, prompt templates, and
> anti-leakage rules. Read `eval/summary.md` for the iteration-1 baseline.
> Do not read the GPQA CSV or the gitignored question files yourself —
> the runbook delegates question handling to a script and to blinded
> sub-agents.

---

## 1. Context

This repo runs a multi-persona-debate vs zero-shot-CoT pilot eval on GPQA
Diamond. **Iteration 1 has run.** Results are in `eval/summary.md`. Both
conditions hit 10/10 on a draw of 10 questions; the only meaningful
data point was gpqa_006, where A has ~80% accuracy across 10 trials but
B's debate converged unanimously to the correct answer in 2 rounds.

**Your job: run iteration 2** on a *different* draw of 10 questions using
the v2 protocol, append results to `eval/summary.md`, and commit.

---

## 2. Critical: do NOT see the gold answers yourself

Methodology blinding requires that **you (the parent session) never read
the `answer` field** of any question file before grading. The protocol
keeps gold-answer access mechanical:

- `eval/pilot_questions.jsonl` (gitignored) contains questions WITH gold
  answers. Do not Read this file directly. Only blinded sub-agents access
  question content via the prompts you build.
- For each question, you'll work from id + question text + choices A–D
  (no `answer` field) when constructing prompts.
- Grading happens at the very end: read each row's `answer` field once,
  compare to the recorded A and B answers, write to
  `pilot_results.jsonl`. **Do this only after all sub-agent runs are
  complete.**

If you find yourself reading any file with question text + gold mid-run,
stop and reconsider.

---

## 3. Pre-flight checks

Run these in the listed order. Halt and ask the user if any check fails.

```bash
# 3a. Branch handling
git status --short
git branch --show-current
# If your environment (e.g. claude.ai/code) auto-created a branch like
# `claude/<task>-<suffix>`, you're already on the right branch — proceed.
# If you're on `main`, create a fresh branch for this iteration:
#     git checkout -b iteration-2     (or iteration-N for subsequent runs)
# Either way, the working tree should be clean.

# 3b. Ledger and CSV present
test -f eval/used_csv_indices.txt && wc -l eval/used_csv_indices.txt
# expect: at least 10 non-comment lines (iteration 1's indices)

test -f eval/GPQA/gpqa_diamond.csv && wc -l eval/GPQA/gpqa_diamond.csv
# expect: ~199 lines (198 rows + header)

# 3c. Iteration 1 artifacts present
ls eval/transcripts/ | wc -l
# expect: 50+ files from iteration 1

test -f eval/PROTOCOL_v2.md && test -f eval/summary.md
# expect: both exist
```

If `eval/GPQA/gpqa_diamond.csv` is missing, ask the user to drop the CSV at
that path and stop.

---

## 4. Runbook

### 4a. Sample 10 new questions (no replacement)

```bash
python3 eval/csv_to_questions.py \
    --csv eval/GPQA/gpqa_diamond.csv \
    --n 10 --seed 42 \
    --update-ledger
```

Verify output: should report `Pool: 198 total rows, 10 used, 188 available.
Sampling 10.` and `Appended 10 indices to eval/used_csv_indices.txt.`

`eval/pilot_questions.jsonl` is now populated with 10 new questions
(gpqa_000 … gpqa_009). It is gitignored.

Do **not** Read `pilot_questions.jsonl` directly. Build a blinded view by
running:

```bash
python3 -c "
import json
for line in open('eval/pilot_questions.jsonl'):
    d = json.loads(line)
    print(d['id'], '|', d['domain'])
    print('  Q:', d['question'][:80] + ('...' if len(d['question'])>80 else ''))
    for letter, choice in d['choices'].items():
        print(f'  {letter}) {choice[:60]}')
    print()
"
```

This prints id, domain, question preview, and choices — enough to construct
sub-agent prompts. It does NOT print the `answer` field.

For full question text (which you need verbatim in prompts), build prompts
with:

```bash
python3 -c "
import json, sys
qid = sys.argv[1]
for line in open('eval/pilot_questions.jsonl'):
    d = json.loads(line)
    if d['id'] == qid:
        print('QUESTION:', d['question'])
        for k,v in d['choices'].items():
            print(f'{k}) {v}')
        break
" gpqa_000
```

This prints question + choices for one id, no answer. Use this output in
the prompts you build.

### 4b. Run Condition A — 10 sonnet calls in parallel

For each id in gpqa_000..gpqa_009, fire one Agent call (model="sonnet")
with this prompt template (substitute `{question}` and `{A}{B}{C}{D}`):

```
You are answering a graduate-level multiple-choice question. Think step by step, then on the final line output exactly: `Answer: <A|B|C|D>` (just one of those four letters).

Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Reason from your own knowledge only. Do not use external tools (no Bash, Python, web search, etc.).
```

Save each raw response to `eval/transcripts/{id}_A.txt`. Parse the final
letter via regex `Answer:\s*([A-D])` (fall back to whole-text scan if
final-line match fails; mark `PARSE_FAIL` if both fail).

Fire all 10 in one message for parallelism.

### 4c. Run Condition B per question — blinded planner pattern

Per `eval/PROTOCOL_v2.md` §1 and §3. Fresh planner each cycle (no
SendMessage continuation in this environment).

For each question, the cycle is:

1. **Cycle-1 planner** (sonnet, blinded): builds decomposer prompt with
   §3 anti-leakage rules. Returns `{action:"fire", calls:[{id:"decomposer", prompt:"..."}]}`.
2. **Decomposer** (sonnet eval): produces 4 personas. Save raw to
   `{id}_B_decomposer.txt`.
3. **Cycle-2 planner** (sonnet, blinded): parses 4 personas, builds 4
   round-0 prompts. Returns 4 fire instructions.
4. **Round 0** (4 sonnet evals in parallel): save each to
   `{id}_B_round0_p{1..4}.txt`. Parse letters.
5. **If unanimous → DONE.** Final B answer = consensus letter. Skip
   synthesizer (per PROTOCOL_v2 §4).
6. **If split**: cycle-3 planner builds round-1 prompts (TRANSCRIPT =
   verbatim concat of round 0). Round 1 odd → "propose refined answer."
   Save responses, check unanimous.
7. **Continue** rounds 2 (even, "critique"), 3 (odd, "propose"),
   4 (even, "critique") if needed. Stop on unanimous.
8. **If round 4 ends without consensus**: run synthesizer on full
   transcript. Save to `{id}_B_synthesizer.txt`. Final B answer = parsed
   from synthesizer.

**Important reminders from PROTOCOL_v2 §5:**

- Every planner prompt must include: *"You will NOT call any tools.
  Output exactly ONE JSON object on a single line. Do not pre-solve
  the problem."* If a planner returns with `tool_uses > 0`, treat as
  hard fail and re-fire that planner.
- Every eval prompt must end with: *"Reason from your own knowledge
  only. Do not use external tools (no Bash, Python, web search, etc.)."*
- Decomposer prompt must include the anti-leakage block from
  PROTOCOL_v2 §3 ("Persona descriptions must be ROLE-ONLY..." etc.).

### 4d. Variance check on contested questions

For any question where **A's answer disagrees with B's consensus**, fire
**3 additional A-instances** (sonnet, same prompt as 4b) to estimate A's
variance on that question. Record letters; this gives a 4-trial
distribution per contested question.

(In iteration 1, a 10-instance variance check on gpqa_006 — the only
contested-within-B question — revealed A had ~80% accuracy there. Future
iterations should at least sample k=4 on contested questions.)

### 4e. Grade and write results

**Now** read `eval/pilot_questions.jsonl` to access gold answers. For each
question append a row to `eval/pilot_results.jsonl`:

```json
{"id": "gpqa_000", "domain": "...", "gold": "C",
 "A_answer": "C", "A_correct": true,
 "A_variance_check": null,
 "B_answer": "C", "B_correct": true,
 "B_rounds_used": 0, "B_consensus_reached": true,
 "B_synthesizer_used": false}
```

For contested questions add `"A_variance_check": ["C","C","D","C"]` (the
4 sampled letters).

### 4f. Append to eval/summary.md

Add a new section "## Iteration 2 (N=10, sonnet eval)" with:

- Headline accuracy table for A and B
- Per-question outcome table (id, A, B, gold, B_rounds_used, notes)
- Detail on any contested case (round-by-round dynamics)
- Variance-check results for any question where A and B disagreed
- Caveats: N still small, sample bias, anything else specific to this draw

Do NOT include verbatim question text in summary.md (per dataset terms).
Refer to questions by id only.

### 4g. Commit and merge to main

```bash
git status --short
# expect: only eval/summary.md modified.
# Verify NOT listed: pilot_questions.jsonl, pilot_results.jsonl,
# transcripts/, used_csv_indices.txt — all gitignored. If any of those
# show, the gitignore is broken and the commit will leak question text.
```

Stage and commit on the iteration branch:

```bash
git add eval/summary.md
# only summary.md unless you intentionally updated PROTOCOL_v2.md or
# csv_to_questions.py during this iteration

git diff --cached --stat
# sanity-check the staged set

git commit -m "Iteration N results: <one-line summary>" \
    --message "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Push the branch and fast-forward main:

```bash
# Push the iteration branch first
git push -u origin HEAD

# Fast-forward merge into main (same pattern iteration 1 used)
git checkout main && git pull --ff-only origin main
git merge --ff-only -    # the dash references the previous branch (iteration-N)
git push origin main

# Confirm
git log --oneline -3
```

If the fast-forward fails (because main has moved ahead of the iteration
branch), stop and ask the user how to proceed — don't force-push, don't
attempt a merge commit without confirmation.

---

## 5. Failure modes — handle gracefully

- **Sub-agent refuses with Usage Policy error.** Retry once. If still
  refuses, mark that call as ERROR and continue. Persistent refusals on
  the same question → swap that question for the next available one
  (run `csv_to_questions.py --n 1 --seed 42 --update-ledger` to draw a
  replacement; remove the failing question from results).
- **Planner returns non-JSON.** Re-fire with same prompt. If still bad,
  fall back to mechanical prompt construction (this is editorializing;
  document it in summary.md as a methodology deviation).
- **Decomposer fails to produce 4 parseable personas.** Fall back to
  the generic set (Mechanical Engineer, Theoretical Physicist,
  Mathematician, Contrarian First-Principles Thinker) and note
  `fallback_personas_used: true`.
- **Round 4 ends without consensus.** Fire synthesizer. Record
  `B_synthesizer_used: true`.
- **Cost concern**: full B run is ~70-150 sub-agent calls per question
  in the worst case (each round = 1 planner + 4 evals; up to 4 rounds +
  synthesizer). For 10 questions plan for 200-500 calls total, plus 10
  for A and ~30 for variance checks. Pre-approve `Agent` in
  `/permissions` before starting.

---

## 6. Quick reference — what NOT to do

- Don't compose round-N prompts yourself by paraphrasing prior rounds —
  always delegate prompt construction to a blinded planner with verbatim
  transcript concatenation. Iteration 1 lost methodology rigor here.
- Don't add the answer choices A–D to the decomposer prompt without the
  full anti-leakage guard block — sonnet has been observed to embed the
  answer letter in a persona's TRAITS field when given choices without
  guards.
- Don't skip the no-tools rule on eval prompts. Without it, sub-agents
  reach for Bash/numpy and the conditions become non-comparable.
- Don't run the synthesizer when consensus is already reached. Per
  PROTOCOL_v2 §4 it's conditional.
- Don't commit `eval/transcripts/`, `eval/pilot_*.jsonl`, or
  `eval/used_csv_indices.txt`. They're gitignored — verify before
  committing.

---

## 7. When you're done

You should have:

- `eval/pilot_questions.jsonl` (gitignored): 10 new questions
- `eval/transcripts/gpqa_000_A.txt … gpqa_009_A.txt` (gitignored)
- `eval/transcripts/gpqa_000_B_*.txt … gpqa_009_B_*.txt` (gitignored)
- `eval/pilot_results.jsonl` (gitignored): 10 graded rows
- `eval/summary.md` updated with iteration 2 section (committed)
- `eval/used_csv_indices.txt` updated (gitignored)

Report to the user:

- Iteration 2 accuracy for A and B
- Any flip cases (A and B disagree)
- Variance results on contested questions
- Recommended next step (iterate again? expand N? change conditions?)
