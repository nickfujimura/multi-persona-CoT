# v5 — findings (probabilistic persona output)

Variant test: replace persona's discrete-letter output with a
probability vector over A/B/C/D + a modal-letter line. v5 tests
**representation depth** (probability vs single letter), orthogonal
to v_judge (which tested aggregation) and multipool (which tested
pool draws).

**The wager-framing arm (v5_v9, the v9 component) is deferred** —
see `wager_arm_spec.md` for the spec and rationale.

## Headline finding

**The probability-output prompt is not a neutral output format.**
Same persona descriptions, same question — different modal verdicts
when the prompt asks for probabilities + modal vs just a letter.

On Pool A (= v_judge fresh decomposer reused), 5 N-values:
- 1 modal verdict shifted toward gold (probabilistic prompt corrected
  a wrong letter-modal at one N).
- 1 modal verdict shifted away from gold (probabilistic prompt flipped
  a correct letter-modal at another N).
- 2 modal verdicts unchanged in direction (preserved correct).
- 1 modal verdict shifted from B-modal to tied at one N (toward A
  direction without crossing the modal boundary).

**Net Pool A accuracy: probably neutral or slightly worse with v5
than letter-only.** The probability prompt isn't a free improvement
either; it has its own failure modes.

## Experimental design

Two pools:
- **Pool A**: v_judge fresh decomposer outputs reused (same persona
  descriptions tested under both letter-only and v5 prompts).
- **Pool B**: 5 newly-generated decomposers for this experiment.

Conditions:
- **v5**: persona prompt asks for probability vector + modal letter
  (no wager framing).
- **v5_v9** (wager): same as v5 plus a wager-framing sentence
  ("you are wagering $100 on each letter at the probabilities you
  report"). **Not run** — see `wager_arm_spec.md`.

Sweep: N ∈ {4, 5, 6, 7, 8}, k=1 trial per persona, round 0 only.
Total: 5 new Pool B decomposers + 60 v5 r0 evals = 65 dispatches.

## Per-cell modal comparison (Pool A only)

| N | Letter-only modal | v5 modal | Δ direction |
|---|---|---|---|
| 4 | β | tie | toward α |
| 5 | α | α (sharpened) | same direction |
| 6 | β | α | flipped to α |
| 7 | α | β | flipped to β |
| 8 | β | β | same direction |

Two modal flips in opposite directions (N=6 toward α, N=7 toward β).
Probabilistic prompt is not uniform-direction in its effect.

## Why probability output flips modal direction

Two compounding factors:

1. **Asymmetric per-persona confidence by direction.** When a persona's
   modal is α, p_α typically lands 0.55–0.78 (moderate). When modal
   is β, p_β typically lands 0.70–0.88 (higher). The mass-vs-noise
   structure differs by direction. Asking for an explicit probability
   forces personas to articulate this asymmetry.

2. **Probability prompt biases personas toward their dominant
   reasoning frame.** Asking "what probability for each letter?"
   makes personas commit more strongly to whichever framework they
   started with — most often framework β on Q*. The letter-only
   prompt allows hedged single-letter answers from personas who
   would otherwise distribute mass; the probability prompt forces
   them to articulate their bias.

## Log-pool aggregation behavior

For one specific cell (N=6 Pool A v5), log-pooled probability
analysis:

- Geometric mean p_α across 6 personas ≈ 0.42 (post-renormalize).
- Geometric mean p_β across 6 personas ≈ 0.19.
- Log-pool gives modal α (margin ~2.2× over β).

So log-pool ALSO gives α at this cell — consistent with modal α from
counting. The single β-leaning persona's confidence (p_β ≈ 0.85) is
crushed by the geometric mean's sensitivity to low p_β values from
the α-leaning personas.

This confirms the dispersion-aware aggregation concern (see
`v_judge` discussion of dispersion vs uncertainty under log-pool):
log-pool conflates "5 personas confidently disagree with 1 confident
dissenter" with "everyone is uncertain." Both produce relatively
flat pooled distributions, but they mean very different things, and
log-pool can't distinguish.

## Cross-pool v5 ensemble

Pool A and Pool B v5 modal verdicts compared:

| N | Pool A v5 | Pool B v5 | Cross-pool modal | Match gold? |
|---|---|---|---|---|
| 4 | tie | β | β | yes |
| 5 | α | α | α | no |
| 6 | α | α | α | no |
| 7 | β | α | tie | no |
| 8 | β | β | β | yes |

**Cross-pool v5 ensemble: 2/5 correct.** Same accuracy as the
multi-pool letter-only ensemble at N=6 pools. Probability output
doesn't escape the persona-pool bottleneck.

## What v5 tells us

- Prompt format is a real degree of freedom for the methodology.
  Researchers should treat letter-vs-probability output as a
  methodology variable, not an implementation detail.
- Probability output preserves more information (confidence
  magnitude) but costs more (asymmetric calibration, log-pool
  fragility).
- For accuracy on Q*, neither letter-only nor v5 clearly wins.
  Both have failure modes, both fail to escape the persona-pool
  bottleneck.

## Limitations

- N=1 question. Cross-question generalization is the obvious test.
- Round 0 only. Round-1+ debate dynamics under probability output
  deferred.
- **No wager arm data** (`wager_arm_spec.md`). The v5 effect alone
  doesn't tell us whether wagering language reduces the
  asymmetric-confidence pattern observed in raw v5 calibration.
- 6 personas per cell (k=1 per persona). Sample is comfortable for
  trend detection, not precise effect-size estimation.

## Files

- Raw transcripts (gitignored): `eval/transcripts/experiment_v5_v9/`
- Full unredacted findings (gitignored):
  `eval/transcripts/experiment_v5_v9/findings.md`
- Design (gitignored):
  `eval/transcripts/experiment_v5_v9/DESIGN.md`
