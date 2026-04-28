# Multi-Persona CoT — Pilot Eval Summary

## Iteration 1 (N=10, sonnet eval, blinded planner architecture)

### Headline

**Both conditions: 10/10 correct. Tie at ceiling.**

| Condition | Accuracy | Sub-agent calls per question (avg) |
|---|---|---|
| A (zero-shot CoT, single sonnet agent) | 10/10 | 1 |
| B (multi-persona debate, blinded planner pattern) | 10/10 | ~6.5 (decomposer + 4 round-0 + ~1.3 cycle planners; 1 question went to round 2) |

### Per-question outcome

| ID | Domain | A | B | Gold | B rounds used | Notes |
|---|---|---|---|---|---|---|
| gpqa_000 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_001 | Biology | C | C | C | 0 | unanimous round 0 |
| gpqa_002 | Physics | C | C | C | 0 | unanimous round 0 |
| gpqa_003 | Chemistry | A | A | A | 0 | unanimous round 0 |
| gpqa_004 | Physics | C | C | C | 0 | unanimous round 0 |
| gpqa_005 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_006 | Physics | D | D | D | **2** | round 0 split 2-2; round 1 → 3-1; round 2 → unanimous, matched gold |
| gpqa_007 | Physics | C | C | C | 0 | unanimous round 0 |
| gpqa_008 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_009 | Biology | A | A | A | 0 | unanimous round 0 |

Sample composition: 7 Physics, 2 Biology, 1 Chemistry (seed=42 draw of 10 from
the 198-question Diamond set). A and B agreed on every question. No flip cases.

### gpqa_006 — the only contested question (aggregate dynamics)

Domain: Physics. Per-question content (question text, choices, persona
reasoning, gold answer) is in local-only transcripts under
`eval/transcripts/gpqa_006_B_*.txt` and is NOT reproduced here.

Round-letter trajectory (anonymized as L1/L2 to refer to the two contested
options without revealing which letter was correct):

- **Round 0**: 2 personas → L1, 2 personas → L2 (split).
- **Round 1** (odd → propose): 1 persona flipped L1→L2; 1 held L1.
  Trajectory: L2, L2, L2, L1 (3-1).
- **Round 2** (even → critique): the remaining L1 persona flipped to L2.
  Trajectory: L2, L2, L2, L2 (unanimous). B's consensus answer matched
  gold.

The qualitative pattern: the two opposing camps anchored on different
readings of the question, debate flipped the dissenters one at a time
across two rounds, and the eventual consensus matched the correct answer.
This is the only case in iteration 1 where B's debate dynamics did
something A couldn't do in a single shot — but only because A happened to
get this question right; A's accuracy on gpqa_006 across 10 trials was
~80% (see variance check below).

### Variance check on gpqa_006

We re-ran A 10 times on gpqa_006 to test whether A's correct answer was
robust:

```
D, C, D, D, C, D, D, D, D, D  →  8/10 D, 2/10 C
```

A scores ~80% on this question across trials. **Our single observation of
A=D in the main run was the lucky case.** B's debate, though it cost ~10×
more sub-agent calls, converged unanimously to D in 2 rounds — the only
question where B's debate dynamics added robustness against A's variance.

### Caveats and limits

- **N=10 is too small.** Sonnet 4.6's published GPQA Diamond accuracy is
  ~90%; our sample produced 100% A-accuracy, suggesting a draw biased toward
  easier questions. Future iterations should expand N and explicitly target
  questions where A's accuracy is uncertain.
- **A and B are tied at ceiling.** With both at 10/10, B has no headroom to
  demonstrate value on this sample. The single hint of B's value is gpqa_006,
  where A has real variance (~80%) but B's consensus is robust. To
  distinguish signal from noise we need the variance check on more questions.
- **Only one question (gpqa_006) needed B's debate dynamics to engage at
  all.** 9 of 10 questions reached unanimous consensus at round 0 — meaning
  B added no debate value on those, just persona overhead.
- **Methodology iteration during the run.** The first-pass methodology had
  several leakage paths that we identified and patched mid-run:
  - Planner was solving the problem in its preamble (gpqa_004 cycle 2). Fixed
    by adding "you will not call any tools" to planner prompts.
  - Decomposer was embedding the answer letter in persona TRAITS (gpqa_002
    v1). Fixed by adding explicit anti-leakage guards to decomposer prompts.
  - Eval calls were running Python with numpy to verify arithmetic. Fixed by
    adding "Reason from your own knowledge only. Do not use external tools"
    to every eval prompt.
  - Condition A and B started with different no-tools rules, making them
    apples-to-oranges. Aligned mid-run.
  - First-pass result (10/10 A, 10/10 B with these issues) was discarded; the
    figures above come from the corrected protocol after re-running cleanly.
- **The synthesizer step never fired in this iteration** because every
  question reached unanimous consensus before round 4. Future protocol makes
  the synthesizer conditional (only when 4 rounds end without consensus).

### Methodology decisions worth carrying forward

- **Blinded planners do all orchestration.** Main session is dumb transport.
- **Eval model = sonnet.** Haiku was tried and produced off-script outputs +
  unreliable graduate-level reasoning.
- **No-tools rule** baked into every eval prompt.
- **Anti-leakage guards** baked into every decomposer prompt.
- **Planner forbidden from calling tools** to prevent pre-solving.
- **Question selection: ledger-based no-replacement** sampling
  (`csv_to_questions.py` v2 + `used_csv_indices.txt`).

See `eval/PROTOCOL_v2.md` for the full v2 protocol and prompts.

### Cost / wall-clock notes

- Iteration 1 burned roughly 200+ sub-agent calls across A, B, planners, and
  the methodology re-runs. Most permission prompts came from un-pre-approved
  `Agent` calls; for future runs, pre-approve `Agent` in `/permissions` for
  unattended execution.
- gpqa_006 alone consumed: 1 cycle-1 planner + 1 decomposer + 1 cycle-2 planner +
  4 round-0 + 1 cycle-3 planner + 4 round-1 + 4 round-2 = **16 sub-agent calls
  for one question** (vs. 1 for A). All to converge on the same answer A
  produced.

---

## Iteration 2 (N=10, sonnet eval, blinded planner architecture, opus auditor + opus blinded grader)

### Headline

**A: 10/10. B: 9/10. B's only contested question flipped to the wrong answer in debate.**

| Condition | Accuracy | Sub-agent calls per question (avg) |
|---|---|---|
| A (zero-shot CoT, single sonnet agent) | 10/10 | 1 (+3 variance trials on the contested case) |
| B (multi-persona debate, blinded planner pattern) | 9/10 | ~6.4 (decomposer + 4 round-0 + ~1.4 cycle planners; 1 question went to round 1) |

### Per-question outcome

| ID | Domain | A | B | Gold | B rounds used | Notes |
|---|---|---|---|---|---|---|
| gpqa_000 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_001 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_002 | Biology | C | C | C | 0 | unanimous round 0 |
| gpqa_003 | Physics | A | A | A | 0 | unanimous round 0 |
| gpqa_004 | Physics | A | A | A | 0 | unanimous round 0 |
| gpqa_005 | Chemistry | D | D | D | 0 | unanimous round 0 |
| gpqa_006 | Chemistry | B | B | B | 0 | unanimous round 0 |
| gpqa_007 | Chemistry | B | **A** | **B** | **1** | **FLIP. round 0 split 2-2; round 1 unanimous A; debate flattened the correct intuition** |
| gpqa_008 | Physics | D | D | D | 0 | unanimous round 0 |
| gpqa_009 | Physics | A | A | A | 0 | unanimous round 0 |

Sample composition: 6 Physics, 3 Chemistry, 1 Biology (seed=42 draw of 10 from the 188 not-yet-used GPQA Diamond rows after iter-1 ledger). Iter-1 vs iter-2 ledger total: 20/198 used; 178 remaining for future iterations.

### gpqa_007 — the flip case (aggregate dynamics)

Domain: Chemistry. Per-question content (verbatim text, choices, named entities, numerical values, persona reasoning, gold answer) stays in local-only transcripts under `eval/transcripts/gpqa_007_B_*.txt` and is NOT reproduced here.

The question admits two competing physical-interpretation framings (α and β). The two framings imply different answers among the four choices. The gold answer aligns with framing β.

Round-letter trajectory (anonymized: Lα and Lβ are the letters implied by framing α and β respectively):
- **Round 0**: 2 personas → Lα, 2 personas → Lβ. 2-2 split. The personas grouped by domain training: two from theoretical/spectroscopy backgrounds defaulted to framing α; two from applied/synthesis backgrounds defaulted to framing β.
- **Round 1** (odd → propose-refined): both β-camp personas flipped to α. The convince-mechanism (per opus auditor on round-1 transcripts, no quoted content here) was a rhetorically dominant reframing argument that recategorized the question's setup, making framing β look like a category error. The argument was sophisticated and internally consistent with framing α's premises — but it relied on a premise the question did not actually impose, so the recategorization was wrong.
- B-consensus: Lα (incorrect). B_rounds_used=1, B_consensus_reached=true, B_synthesizer_used=false.

### Variance check on gpqa_007

We re-ran A 4 times total on gpqa_007. The 4-trial distribution (using the same Lα/Lβ anonymization) was:

```
3/4 trials → Lβ (correct)
1/4 trials → Lα (incorrect)
```

A's modal answer (Lβ=correct) is robust at ~75%. B's debate happened to converge on the 25% framing — the *minority* intuition that A also occasionally produces. So on this question, B's debate dynamics did not add robustness over A; they pushed the consensus *away* from the correct dominant intuition.

This is the inverse of iteration-1's gpqa_006, where B's debate added robustness against A's variance. Here, it flattened a correct outlier under a sophisticated-sounding wrong argument.

### Caveats and limits

- **N=10 is still too small.** One flip case in a draw of 10 is consistent with noise, but the *direction* (debate flattens the correct intuition) is the worst-case scenario from README §5's interpretation table: "A beats B by ≥2 | Suggests debate flattens correct outliers. Investigate transcripts before scaling."
- **Sample composition shifted toward chemistry.** Iter-2's draw was 6 Physics / 3 Chemistry / 1 Biology vs. iter-1's 7P/2B/1C. Both sub-samples may overrepresent easy questions — A=10/10 on both iterations is well above sonnet's published GPQA Diamond accuracy (~90%).
- **B's debate engaged on only 1/10 questions.** Same low engagement rate as iter-1. The technique adds 6× the sub-agent cost of A on average but only does work in ~10% of cases.
- **Methodology corrections during iter-2 run:**
  - First cycle-1 planner wave hit transient API 500s on 4/10 calls; retried successfully.
  - First decomposer wave saved persona blocks to disk via Write inside each sub-agent (a new pattern this iteration: each eval sub-agent saves its own full reasoning to disk and returns only the parsed letter, so main-session context stays lean even at 40+ eval calls per run).
  - Cycle-3 planner for gpqa_007 round 1 introduced a benign 1-character typo when re-emitting one persona's prior transcript, and the main session also slipped in 1-sentence paraphrases of each persona's framing into the round-1 prompts. Both are main-session/planner contamination per PROTOCOL_v2 §1. Round 1 was re-run from scratch with cleaner prompts (round-1 sub-agents Read the round-0 files themselves rather than receiving any embedded transcripts or summaries from main). Both runs gave the same unanimous result, so the contamination did not affect the outcome — but the second run is the methodologically clean one.
  - Used opus quality-auditors after each round to spot-check transcripts on disk (the sub-agents-save-and-return-letter-only pattern means main can't directly verify reasoning quality). Auditors reported all clean across 40 round-0 + 4 round-1 transcripts.
  - Used an opus blinded grader at the end: it Read `pilot_questions.jsonl` itself and computed correctness, so the main session never ingested gold answers.

### Methodology decisions worth carrying forward

- **Sub-agent self-save pattern.** Each eval sub-agent calls Write to save its own full reasoning to disk and returns only `Answer: <letter>` to main. This collapses main-session context cost from ~1k tokens per eval to ~10. The "no external tools" rule for reasoning still holds — Write is permitted only after reasoning is complete.
- **Planner Read pattern.** Round-N+1 planners are given file paths and Read the round-N transcripts themselves, rather than main session re-reading and inline-embedding. Avoids main-session contamination of prior-round content.
- **Opus auditor after each round.** Reads a batch of saved transcripts and flags refusals, parse failures, leakage signals, persona drift, tool-misuse — returns a short summary to main. Replaces what main would otherwise have to do by re-reading transcripts itself.
- **Opus blinded grader.** The parent session stays blinded end-to-end; only the grader Reads gold and compares against a small dict the parent passes in. Strictly stronger blinding than "main reads gold at the very end" per the original protocol.

### Cost / wall-clock notes

- Total sub-agent calls: ~70 for B (10 cycle-1 planners w/ 4 retries + 10 decomposers + 10 cycle-2 planners + 40 round-0 evals + 1 cycle-3 planner + 4 round-1 evals + 1 audit cycle per round + 1 final grader), 13 for A (10 + 3 variance), plus 4 contaminated round-1 evals that were discarded.
- Wall-clock: substantial — round-0 batches and round-1 each took several minutes for parallel sonnet calls. The blinded planner architecture is high-overhead; 6× cost ratio vs. A on the median question, much higher on contested ones.
- Pre-approving `Agent` and `Bash` in `/permissions` before the run reduces permission-prompt friction; this iteration ran mostly unattended after the initial approvals.

---



Next iteration (iter-3) should:

1. **Use a fresh Claude Code session** (no carry-forward of gold knowledge).
2. **Run `csv_to_questions.py --update-ledger`** to sample 10 new questions
   (178 not-yet-used remain in the GPQA Diamond pool after iter-1 + iter-2).
3. **Apply v2 protocol** plus the iter-2 patches: sub-agent self-save +
   planner Read + opus auditor after each round + opus blinded grader.
4. **Run k=4 A-instances per question** (not just on contested ones) to
   estimate A's variance baseline more broadly. Iter-1 and iter-2 each had
   exactly 1 contested question — small denominator for variance comparison.
5. **Append iteration-3 results to this file** under a new section.

### Updated signal hunting after iter-2

We were hunting for: *questions where A is right less than 100% of the time,
and B's debate brings it above A's mean*. Iter-1's gpqa_006 was a candidate
(B converged unanimously on the correct answer; A was correct 8/10 across
trials). Iter-2's gpqa_007 is the **opposite signal**: A's modal answer (3/4
trials) was correct, and B's debate dynamics flipped the consensus to the
incorrect minority framing. The driving mechanism was a rhetorically dominant
but contextually-wrong reframing argument that recategorized the question
setup, overruling the framing the question actually intended.

Across the 2 contested cases observed so far, B's debate dynamics:
- Iter-1 gpqa_006: added robustness over A (B correct, A correct ~80%).
- Iter-2 gpqa_007: flattened correct outlier (B incorrect, A correct ~75%).

That's 1-1 — no directional signal yet. Iter-3 would help disambiguate, but
the more interesting question for iter-3+ is: *can we predict in advance
which way B will go?* If B's debate is just amplifying the most articulate
argument regardless of correctness, it's not a robustness layer; it's a
rhetoric amplifier. A study on prompt features that predict B's win/loss
direction would be more informative than another N=10 draw.
