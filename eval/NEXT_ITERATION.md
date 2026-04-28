# Task brief for the next Claude Code session (iteration 3)

> **If you are a Claude Code session reading this:** this file is your task
> brief. Read all of it, then execute Section 4 (Runbook) end-to-end.
>
> Read first:
> - `eval/PROTOCOL_v2.md` — architecture, prompt templates, anti-leakage
>   rules, and **§9 amendments** (mandatory for iter-3+; codifies patterns
>   iter-2 arrived at after several mid-run course corrections).
> - `eval/summary.md` — iter-1 and iter-2 results.
>
> **Do not read** `eval/pilot_questions.jsonl` or any iter-3 transcript
> file directly. The protocol delegates question text and gold answers to
> blinded sub-agents.

---

## 1. Context

This repo runs a multi-persona-debate vs zero-shot-CoT pilot eval on GPQA
Diamond. **Iterations 1 and 2 have run.** Both achieved A=10/10. Iter-1
got B=10/10 (1 contested case where B's debate added robustness over noisy
A). Iter-2 got B=9/10 (1 contested case where B's debate flattened a
correct outlier under a sophisticated-sounding wrong argument).

Across iter-1 + iter-2, the 2 contested cases are split 1-1 — no directional
signal yet. Iter-3 is meant to disambiguate, but per `summary.md`'s closing
note, an even more interesting question is whether B's win/loss direction
is predictable from prompt features.

Iter-3's job: run the v2-amended protocol on a fresh draw of 10 questions,
plus k=4 A-variance per question (broader coverage than iter-2's
contested-only k=4), append results to `summary.md`, commit (after
leak-check), push the iteration branch, and fast-forward main.

---

## 2. Critical: blinding

**The parent session never reads gold answers.** Iter-2 introduced an
opus blinded grader (PROTOCOL_v2 §9.5) that reads `pilot_questions.jsonl`
itself and returns just headline counts. Use that pattern. Do NOT manually
read `pilot_questions.jsonl` at any point.

Also: do not Read the answer field of any question file even via Bash
(`grep`, `head`, `cat`, etc.) — those land in main context as stdout.

If you find yourself reading any file that contains the `answer` field,
stop and reconsider — that's a blinding violation.

---

## 3. Pre-flight checks

Run these in the listed order. Halt and ask the user if any check fails.

```bash
# 3a. Branch handling
git status --short
git branch --show-current
# If on `main`, branch off: git checkout -b iteration-3
# If your environment auto-created a branch (e.g. `claude/...`), you may
# already be on the right branch — proceed.
# Either way, working tree should be clean (a stray .DS_Store is fine).

# 3b. Ledger and CSV present
test -f eval/used_csv_indices.txt && wc -l eval/used_csv_indices.txt
# expect: at least 25 lines (iter-1 + iter-2 indices + comment header).
# 178 of 198 GPQA Diamond rows remain unused.

test -f eval/GPQA/gpqa_diamond.csv && wc -l eval/GPQA/gpqa_diamond.csv
# expect: ~9000+ lines (CSV cells contain newlines from question text).

# 3c. Iter-1 + iter-2 transcripts present
ls eval/transcripts/iteration_1/ | wc -l   # expect ~68 files
test -d eval/transcripts/iteration_1 && echo "iter-1 archived"
ls eval/transcripts/                       # iter-2's transcripts are at
                                           # the top level still — you'll
                                           # archive them in step 4a.

test -f eval/PROTOCOL_v2.md && test -f eval/summary.md
# expect: both exist; PROTOCOL_v2 has a §9 amendments section.
```

If `eval/GPQA/gpqa_diamond.csv` is missing, ask the user to drop the CSV at
that path and stop.

---

## 4. Runbook

### 4a. Archive iter-2 transcripts (PROTOCOL_v2 §9.8)

```bash
mkdir -p eval/transcripts/iteration_2
mv eval/transcripts/gpqa_*.txt eval/transcripts/iteration_2/
mv eval/transcripts/contaminated_round1 eval/transcripts/iteration_2/ \
   2>/dev/null || true   # iter-2 had a contaminated-round-1 archive subdir
ls eval/transcripts/   # expect: iteration_1, iteration_2 (and possibly
                       # already-empty top level)
```

iter-3's transcript files will use the same `gpqa_XXX_*.txt` naming at the
top level, distinct from the archived iter-1/iter-2 paths.

### 4b. Sample 10 new questions

```bash
python3 eval/csv_to_questions.py \
    --csv eval/GPQA/gpqa_diamond.csv \
    --n 10 --seed 42 \
    --update-ledger
```

Verify output reports something like `Pool: 198 total rows, 20 used,
178 available. Sampling 10.` and `Appended 10 indices to
eval/used_csv_indices.txt.`

`eval/pilot_questions.jsonl` is now populated with 10 new questions and
is gitignored.

**Do not Read this file directly.** Skip the iter-2 pattern of writing a
"blinded copy" via Python in main; instead, use the question-splitter
sub-agent below (§4c).

### 4c. Question splitter (PROTOCOL_v2 §9.10)

Dispatch one sub-agent (sonnet, Read+Write only) with this brief: "Read
`eval/pilot_questions.jsonl`. For each of the 10 lines, write
`eval/transcripts/{id}_question.txt` containing the question text and the
4 choices (A/B/C/D) — but NOT the `answer` field. Return only a list of
the 10 ids and their domains." That gives main session (id, domain) pairs
without ingesting question content.

The 10 per-id question files are what cycle-1 and cycle-2 planners will
Read in subsequent steps.

### 4d. Run Condition A — 10 sonnet zero-shot CoT calls + k=4 variance

For each id: fire 4 sonnet `Agent` calls in parallel. Each gets a prompt
that:
- Tells the sub-agent to Read `eval/transcripts/{id}_question.txt` for
  question + choices.
- Asks for step-by-step reasoning.
- Requires `Answer: <A|B|C|D>` on the final line.
- Forbids external tools (no Bash, Python, web search) but permits Write
  once at the end to save full reasoning.
- Specifies the save path: `eval/transcripts/{id}_A_t{1..4}.txt` (where
  t1 is the canonical A trial; t2..t4 are variance check trials).

Total: 40 A-instances. Fire in batches of ≤20 parallel `Agent` calls per
message.

Each sub-agent returns only `Answer: <letter>`. Record the 4 letters per
id; the modal letter (or t1 if multimodal) is `A_answer`. The full
4-letter sequence is `A_variance_check` (for every id, not just contested).

After dispatch, run an opus auditor (PROTOCOL_v2 §9.4) over the 40 A
transcripts to spot-check.

### 4e. Run Condition B per question — blinded planner pattern

Per PROTOCOL_v2 §1, §3, and §9 amendments. For each id:

1. **Cycle-1 planner** (sonnet, Read tool only): Reads
   `eval/transcripts/{id}_question.txt`, builds the decomposer prompt with
   §3 anti-leakage guards, returns JSON: `{action:"fire", calls:[{role:"decomposer", prompt:"..."}]}`. Decomposer prompt should NOT include answer choices (per §5.2).
2. **Decomposer** (sonnet eval, Write at end only): per §3 template +
   §9.2 self-save. Saves persona blocks to `{id}_B_decomposer.txt`,
   returns only the persona blocks in its final reply (those are short).
3. **Cycle-2 planner** (sonnet, Read tool only): Reads
   `{id}_B_decomposer.txt` AND `{id}_question.txt`, builds 4 round-0
   prompts (one per persona) per §3 round-0 template + §9.2 self-save
   addendum. Returns JSON with 4 calls.
4. **Round 0** (4 sonnet evals in parallel, Write at end only): each
   saves to `{id}_B_round0_p{1..4}.txt`, returns only `Answer: <letter>`.
5. **Unanimity check.** If all 4 letters agree → DONE. Final B answer =
   consensus. Skip synthesizer (§4 conditional).
6. **If split** — invoke cycle-3 planner (sonnet, Read tool only): Reads
   the 4 round-0 transcripts AND the decomposer file AND the question
   file, builds 4 round-1 prompts using the planner Read pattern (§9.3):
   each round-1 prompt should NOT inline transcripts; instead it should
   instruct the round-1 eval to Read its own copy of the round-0 files.
   Round-1 framing: "propose your refined answer, addressing weaknesses
   you see in others' reasoning."
7. **Round 1** (4 sonnet evals in parallel, Read+Write only): each Reads
   the 4 round-0 transcripts itself, then saves its round-1 reasoning to
   `{id}_B_round1_p{1..4}.txt`, returns only `Answer: <letter>`.
8. **Continue rounds 2 (even, "critique"), 3, 4** (alternating
   propose/critique) if needed, with cycle-{4,5,6} planners and round-N
   evals following the same Read-themselves pattern.
9. **If round 4 ends without consensus**: run synthesizer (§3 template).
   Save to `{id}_B_synthesizer.txt`. Final B answer = parsed from
   synthesizer.

After each round, run an opus auditor (§9.4) over that round's
transcripts. For contested rounds, ask the auditor to describe each
persona's reasoning frame in 1 sentence.

### 4f. Variance check protocol (broadened from iter-2)

Iter-2 ran k=4 only on the contested case. Iter-3 runs k=4 on every
question — that's already covered by §4d above (4 A-instances per id
fired up front). No additional dispatch needed for variance.

### 4g. Grade with the opus blinded grader

Pass the grader (opus, Read+Write only) a per-id dict like:

```json
{
  "gpqa_000": {"A_answer":"<modal letter>","B_answer":"<final letter>","B_rounds_used":<int>,"B_consensus_reached":<bool>,"B_synthesizer_used":<bool>,"A_variance_check":["<l1>","<l2>","<l3>","<l4>"]},
  ...
}
```

Grader Reads `pilot_questions.jsonl` itself, computes correctness, writes
`eval/pilot_results.jsonl`, and returns only headline counts + flip cases
+ contested-case variance breakdown. Do NOT have it echo gold letters of
non-contested ids back to main.

### 4h. Append to summary.md (with leak-aware writing)

Add a new section "## Iteration 3 (N=10, sonnet eval, ...)" with:
- Headline accuracy table for A and B.
- Per-question outcome table (id, domain, A, B, gold, B_rounds_used,
  notes). The id+domain+letter table is permissible per dataset terms;
  the underlying choice text is not.
- A_variance_check distribution per id (just letter sequences, no
  semantic content).
- For any contested case: an abstracted round-trajectory using
  Lα/Lβ-style placeholders. NO topic-specific vocabulary, NO unique
  numerical values, NO persona-role names. See PROTOCOL_v2 §9.7 for
  the contamination patterns to actively prevent.
- Methodology notes: what worked, what didn't, anything new this
  iteration.

### 4i. Pre-commit leak check (PROTOCOL_v2 §9.6)

Before staging summary.md, dispatch an opus leak-checker (Read only)
with paths to the staged file and `pilot_questions.jsonl`. It returns
either "CLEAN" or a redaction list. Iterate redactions until clean.
**This step is mandatory** — iter-2 caught 5 leak vectors only by
running the leak-checker, after several rounds of "looks fine" intuition.

### 4j. Commit and merge to main

```bash
git status --short
# expect: only eval/summary.md modified (plus possibly PROTOCOL_v2.md or
# csv_to_questions.py if you intentionally updated them — and any planner
# instruction docs you may have authored).
# Verify NOT listed: pilot_questions.jsonl, pilot_results.jsonl,
# transcripts/, used_csv_indices.txt — all gitignored. If any show, the
# gitignore is broken and the commit will leak question text.
```

Stage and commit:

```bash
git add eval/summary.md
# also stage PROTOCOL_v2.md / csv_to_questions.py / NEXT_ITERATION.md if
# you intentionally updated them
git diff --cached --stat
git commit -m "Iteration 3 results: <one-line summary>" \
    --message "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Push the iteration branch and fast-forward main:

```bash
git push -u origin HEAD
git checkout main && git pull --ff-only origin main
git merge --ff-only -    # references the previous branch (iteration-3)
git push origin main
git log --oneline -3
```

If the fast-forward fails (because main has moved ahead), stop and ask
the user how to proceed. Do not force-push, do not attempt a merge
commit without confirmation.

---

## 5. Failure modes — handle gracefully

- **Sub-agent refuses with Usage Policy error.** Retry once. If still
  refuses, mark that call as ERROR and continue. Persistent refusals on
  the same question → swap that question for the next available one.
- **Transient API 500s on planner waves.** Iter-2 hit 4/10 cycle-1
  planners with 500s; retry resolved them. Plan for retry capacity.
- **Planner returns non-JSON.** Re-fire with same prompt. If still bad,
  fall back to mechanical prompt construction in main, BUT note this as
  a methodology deviation in summary.md (it violates PROTOCOL_v2 §1).
- **Decomposer fails to produce 4 parseable personas.** Fall back to the
  generic set (Mechanical Engineer, Theoretical Physicist, Mathematician,
  Contrarian First-Principles Thinker) and note `fallback_personas_used:
  true`.
- **Round 4 ends without consensus.** Fire synthesizer. Record
  `B_synthesizer_used: true`.
- **Cost concern.** A 10-question B run with k=4 A is ~140-220 sub-agent
  calls (4 × 10 A trials = 40, 1 cycle-1 + 1 decomposer + 1 cycle-2 + 4
  round-0 = 7 per question × 10 = 70, plus contested-question cycle-3
  planner + 4 round-1 ≈ 5 per contested question, plus auditors after
  each round, plus the grader, plus the leak-checker). Pre-approve
  `Agent` and `Bash` in `/permissions` before starting.

---

## 6. Quick reference — what NOT to do

- Don't compose round-N prompts yourself in main session by paraphrasing
  prior rounds. Always delegate to a blinded planner with verbatim
  transcript content; the planner Read pattern (§9.3) is preferred over
  inline embedding.
- Don't add answer choices to the decomposer prompt without the full
  anti-leakage guard block (§5.2).
- Don't skip the no-tools rule on eval prompts. Without it, sub-agents
  reach for Bash/numpy and the conditions become non-comparable.
- Don't run the synthesizer when consensus is already reached (§4
  conditional).
- Don't read `pilot_questions.jsonl` directly in main session — use the
  question splitter (§4c) and the opus blinded grader (§4g + §9.5).
- Don't `cat` or Read transcripts in main session when downstream
  sub-agents can Read them themselves (§9.3).
- Don't commit `eval/transcripts/`, `eval/pilot_*.jsonl`, or
  `eval/used_csv_indices.txt`. They're gitignored — verify before
  committing.
- Don't commit summary.md without running the leak-checker (§9.6).

---

## 7. When you're done

You should have:

- `eval/pilot_questions.jsonl` (gitignored): 10 new questions for iter-3
- `eval/transcripts/iteration_2/...` (gitignored): archived iter-2 files
- `eval/transcripts/{id}_question.txt` (gitignored): per-id question
  files written by the splitter
- `eval/transcripts/{id}_A_t{1..4}.txt` (gitignored): 40 A trial files
- `eval/transcripts/{id}_B_*.txt` (gitignored): decomposer + round-0
  (+ rounds 1..N if contested) for each id
- `eval/pilot_results.jsonl` (gitignored): 10 graded rows
- `eval/used_csv_indices.txt` (gitignored): updated with 10 new indices
- `eval/summary.md` updated with iter-3 section (committed, leak-checked)

Report to the user:
- Iteration 3 accuracy for A and B
- Per-id A_variance_check distributions (any non-unanimous A trials are
  notable — they indicate questions where prompt-level noise is real)
- Any flip cases (A and B disagree)
- Variance-check breakdown on contested questions
- Recommended next step (iterate again? expand N? change conditions?
  shift to studying when B wins vs loses?)
