"""Day 2 midday: lightweight MGSM data-quality check.

Samples 30 items across EN/ZH (and optionally JA/FR), displays parallel
problems plus gold answers for manual review, and records flagged items.
Numeric-gold consistency is automatically checked across languages.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import REPORT_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.io import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--flag_file", default=None,
                    help="optional path to a newline-separated list of item ids to mark as flagged")
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day2_mgsm_qa.json"))
    args = ap.parse_args()

    per_lang = {l: load_dataset("mgsm", l) for l in args.langs}
    n = min(len(v) for v in per_lang.values())
    rng = random.Random(args.seed)
    sample_idx = sorted(rng.sample(range(n), min(args.sample, n)))

    manual_flags = set()
    if args.flag_file and Path(args.flag_file).exists():
        for line in Path(args.flag_file).read_text().splitlines():
            line = line.strip()
            if line:
                manual_flags.add(line)

    items = []
    auto_flags = 0
    flag_lang = "en" if "en" in args.langs else args.langs[0]
    for i in sample_idx:
        per_lang_entries = {l: per_lang[l][i] for l in args.langs}
        golds = {l: per_lang_entries[l]["gold"] for l in args.langs}
        gold_consistent = len(set(golds.values())) == 1
        if not gold_consistent:
            auto_flags += 1
        rec = {
            "idx": i,
            "ids": {l: per_lang_entries[l]["id"] for l in args.langs},
            "questions": {l: per_lang_entries[l]["prompt_payload"]["question"] for l in args.langs},
            "golds": golds,
            "gold_consistent": gold_consistent,
            "manually_flagged": per_lang_entries[flag_lang]["id"] in manual_flags,
        }
        items.append(rec)

    result = {
        "langs": args.langs,
        "flag_language": flag_lang,
        "n_sampled": len(items),
        "n_auto_flagged_inconsistent_gold": auto_flags,
        "n_manually_flagged": sum(1 for r in items if r["manually_flagged"]),
        "items": items,
    }
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")
    print(f"auto-flagged (inconsistent gold): {auto_flags} / {len(items)}")


if __name__ == "__main__":
    main()
