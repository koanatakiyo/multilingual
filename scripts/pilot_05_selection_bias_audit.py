"""Day 2 morning: selection bias audit.

Compare matched subset vs. full set on item difficulty (mean model accuracy),
question length, and optional category distribution.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, REPORT_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids, intersection_across_models  # noqa: E402
from utils.matching import bias_audit_summary  # noqa: E402


def _category_of(dataset: str, item: dict) -> str:
    if dataset == "xcopa":
        return item["prompt_payload"].get("question", "unknown")
    return "all"


def audit(dataset: str, lang_pair=("en", "zh")) -> dict:
    models = sorted({p.stem.split("__")[0] for p in FEAT_DIR.glob(f"*__{dataset}__en__cot.jsonl")})
    if not models:
        return {"error": "no features"}

    correctness_by_model_lang = {}
    for m in models:
        for lang in lang_pair:
            rows = read_jsonl(FEAT_DIR / f"{m}__{dataset}__{lang}__cot.jsonl")
            correctness_by_model_lang[(m, lang)] = {r["id"]: bool(r["correct"]) for r in rows}

    matched_per_model = {
        m: accuracy_matched_ids(
            correctness_by_model_lang[(m, lang_pair[0])],
            correctness_by_model_lang[(m, lang_pair[1])],
        )
        for m in models
    }
    matched_intersection = intersection_across_models(matched_per_model.values())

    en_full = load_dataset(dataset, lang_pair[0], limit=None)
    per_item_acc = defaultdict(list)
    for (m, _), correctness in correctness_by_model_lang.items():
        for iid, ok in correctness.items():
            per_item_acc[iid].append(int(ok))
    difficulty = {iid: (1 - mean(vs) if vs else 0.0) for iid, vs in per_item_acc.items()}

    full_cat = defaultdict(int)
    matched_cat = defaultdict(int)
    for it in en_full:
        full_cat[_category_of(dataset, it)] += 1
        if it["id"] in matched_intersection:
            matched_cat[_category_of(dataset, it)] += 1

    summary_by_model = {
        m: bias_audit_summary(matched_per_model[m], en_full, difficulty)
        for m in models
    }
    summary_intersection = bias_audit_summary(matched_intersection, en_full, difficulty)

    return {
        "dataset": dataset,
        "models": models,
        "matched_subset_sizes_per_model": {m: len(s) for m, s in matched_per_model.items()},
        "matched_intersection_size": len(matched_intersection),
        "bias_per_model": summary_by_model,
        "bias_intersection": summary_intersection,
        "category_distribution": {"full": dict(full_cat), "matched_intersection": dict(matched_cat)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day2_selection_bias.json"))
    args = ap.parse_args()

    result = {d: audit(d) for d in args.datasets}
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
