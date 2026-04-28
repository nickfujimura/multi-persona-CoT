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

## Future iterations

Next iteration should:

1. **Use a fresh Claude Code session** (no carry-forward of gold knowledge).
2. **Run `csv_to_questions.py --update-ledger`** to sample 10 new questions
   (the 188 not yet used).
3. **Apply v2 protocol** as documented in `eval/PROTOCOL_v2.md`.
4. **Run k ≥ 3 A-instances per question** (or at least on questions where
   A's first answer disagrees with B's consensus) to estimate A's variance.
5. **Append iteration-2 results to this file** under a new section.

The signal we're hunting for: *questions where A is right less than 100%
of the time, and B's debate brings it above A's mean*. We didn't find that
signal in iteration 1; gpqa_006 is the only candidate and would need k≥3 of
B to confirm B's robustness across trials.
