"""Convert a downloaded GPQA Diamond CSV into the pilot's JSONL format.

Usage:
    # 1. Download gpqa_diamond.csv from HuggingFace (Idavidrein/gpqa, gated)
    #    after accepting the terms, and place it at eval/gpqa_diamond.csv.
    # 2. Run from repo root:
    python eval/csv_to_questions.py --n 10 --seed 42

Output: eval/pilot_questions.jsonl (gitignored)

Each line:
    {"id": "gpqa_000", "question": "...",
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

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(sample):
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
                "question": row["Question"].strip(),
                "choices": dict(zip("ABCD", choices)),
                "answer": answer_letter,
                "domain": row.get("High-level domain", "").strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} questions to {out_path}")
    print(f"NOTE: {out_path} is gitignored. Do NOT commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
