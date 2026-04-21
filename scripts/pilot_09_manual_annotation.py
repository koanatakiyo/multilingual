"""Day 4 morning: prepare the manual hyperedge annotation sheet.

Selects 20 matched XCOPA items across 2 models, renders trajectories in EN/ZH
side-by-side, and emits a JSONL and a CSV template for two annotators.
Cohen κ computation given filled-in annotations is included as a helper.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, REPORT_DIR  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402


def build_sheet(dataset: str, models, n_items: int, seed: int) -> list:
    rng = random.Random(seed)
    records = []
    for m in models:
        en = read_jsonl(FEAT_DIR / f"{m}__{dataset}__en__cot.jsonl")
        zh = read_jsonl(FEAT_DIR / f"{m}__{dataset}__zh__cot.jsonl")
        en_by_id = {r["id"]: r for r in en}
        zh_by_id = {r["id"]: r for r in zh}
        matched = sorted(accuracy_matched_ids(
            {i: bool(r["correct"]) for i, r in en_by_id.items()},
            {i: bool(r["correct"]) for i, r in zh_by_id.items()},
        ))
        if not matched:
            continue
        picks = rng.sample(matched, min(n_items // max(len(models), 1), len(matched)))
        for iid in picks:
            e, z = en_by_id[iid], zh_by_id[iid]
            records.append({
                "task_id": f"{m}::{iid}",
                "item_id": iid,
                "model": m,
                "dataset": dataset,
                "en_steps": e["steps"],
                "zh_steps": z["steps"],
            })
    return records


def cohen_kappa(labels_a, labels_b) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 1.0
    categories = sorted(set(labels_a) | set(labels_b))
    agree = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    chance = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--models", nargs="+", default=["qwen3-8b", "llama3.1-8b"])
    ap.add_argument("--n_items", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_jsonl", default=str(REPORT_DIR / "pilot_day4_annotation_sheet.jsonl"))
    ap.add_argument("--out_csv", default=str(REPORT_DIR / "pilot_day4_annotation_sheet.csv"))
    args = ap.parse_args()

    recs = build_sheet(args.dataset, args.models, args.n_items, args.seed)
    write_jsonl(Path(args.out_jsonl), recs)

    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "lang", "candidate_hyperedge", "target_step",
            "premise_steps", "annotator_A_joint_necessary", "annotator_B_joint_necessary",
        ])
        for r in recs:
            for lang, steps in (("en", r["en_steps"]), ("zh", r["zh_steps"])):
                for i in range(1, len(steps)):
                    w.writerow([r["task_id"], lang, f"h{i}", steps[i],
                                " | ".join(steps[:i]), "", ""])

    write_json(Path(args.out_jsonl).with_suffix(".meta.json"),
               {"n_records": len(recs), "models": args.models, "dataset": args.dataset})
    print(f"wrote {args.out_jsonl} and {args.out_csv}")


if __name__ == "__main__":
    main()
