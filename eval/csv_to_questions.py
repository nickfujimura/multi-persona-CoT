"""Convert a downloaded GPQA Diamond CSV into the pilot's JSONL format.

v2: Supports no-replacement sampling across iterations via a ledger file.

Usage:
    # 1. Download gpqa_diamond.csv from HuggingFace (Idavidrein/gpqa, gated)
    #    after accepting the terms, and place it at eval/GPQA/gpqa_diamond.csv.
    # 2. Run from repo root:
    python eval/csv_to_questions.py --csv eval/GPQA/gpqa_diamond.csv --n 10 --seed 42

Output: eval/pilot_questions.jsonl (gitignored)
Ledger: eval/used_csv_indices.txt (gitignored) — tracks which CSV row indices
have been used across iterations. Each run excludes already-used indices.
Pass --update-ledger to append the new run's indices to the ledger.

Each output line:
    {"id": "gpqa_000", "csv_index": 47, "question": "...",
     "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
     "answer": "C", "domain": "Physics"}

Choice ordering is shuffled deterministically (seeded) so the correct answer
is not always in the same position.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/gpqa_diamond.csv",
                    help="Path to the downloaded GPQA CSV.")
    ap.add_argument("--out", default="eval/pilot_questions.jsonl",
                    help="Output JSONL path (gitignored).")
    ap.add_argument("--n", type=int, default=10, help="How many questions to sample.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (for reproducibility).")
    ap.add_argument("--ledger", default="eval/used_csv_indices.txt",
                    help="File tracking CSV indices used in prior runs (one int per line). "
                         "These indices are excluded from sampling.")
    ap.add_argument("--update-ledger", action="store_true",
                    help="After sampling, append the new run's CSV indices to the ledger.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Download gpqa_diamond.csv from "
              f"https://huggingface.co/datasets/Idavidrein/gpqa (gated) and place it there.",
              file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {"Question", "Correct Answer",
                "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"}
    missing = required - set(rows[0].keys())
    if missing:
        print(f"ERROR: CSV is missing expected columns: {missing}\n"
              f"Got columns: {list(rows[0].keys())}", file=sys.stderr)
        return 1

    # Read ledger of used CSV indices
    ledger_path = Path(args.ledger)
    used_indices = set()
    if ledger_path.exists():
        with ledger_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    used_indices.add(int(line))

    # Build available pool
    available = [(i, row) for i, row in enumerate(rows) if i not in used_indices]
    if len(available) < args.n:
        print(f"ERROR: only {len(available)} unused rows remain; cannot sample {args.n}.",
              file=sys.stderr)
        return 1

    print(f"Pool: {len(rows)} total rows, {len(used_indices)} used, "
          f"{len(available)} available. Sampling {args.n}.", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = rng.sample(available, args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_indices = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, (csv_idx, row) in enumerate(sample):
            new_indices.append(csv_idx)
            correct = row["Correct Answer"].strip()
            choices = [
                correct,
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ]
            rng.shuffle(choices)
            answer_letter = "ABCD"[choices.index(correct)]
            record = {
                "id": f"gpqa_{i:03d}",
                "csv_index": csv_idx,
                "question": row["Question"].strip(),
                "choices": dict(zip("ABCD", choices)),
                "answer": answer_letter,
                "domain": row.get("High-level domain", "").strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} questions to {out_path}")
    print(f"NOTE: {out_path} is gitignored. Do NOT commit it.")

    if args.update_ledger:
        with ledger_path.open("a") as f:
            for idx in sorted(new_indices):
                f.write(f"{idx}\n")
        print(f"Appended {len(new_indices)} indices to {ledger_path}.")
    else:
        print(f"NOTE: ledger NOT updated (pass --update-ledger to record this run).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
