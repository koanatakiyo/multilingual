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


def _feature_path(model: str, dataset: str, lang: str) -> Path:
    return FEAT_DIR / f"{model}__{dataset}__{lang}__cot.jsonl"


def audit(dataset: str, lang_pair=("en", "zh"), exclude_models=None) -> dict:
    drop = set(exclude_models or [])
    candidates = {p.stem.split("__")[0] for p in FEAT_DIR.glob(f"*__{dataset}__en__cot.jsonl")}
    models = sorted(
        m for m in candidates
        if m not in drop and all(_feature_path(m, dataset, lang).exists() for lang in lang_pair)
    )
    if not models:
        return {"error": "no complete feature pairs"}

    correctness_by_model_lang = {}
    for m in models:
        for lang in lang_pair:
            rows = read_jsonl(_feature_path(m, dataset, lang))
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

    # Per-language difficulty. Mixing EN and ZH correctness into one score turns
    # "difficulty" into a blend of true difficulty and cross-lingual consistency,
    # which is exactly the variable we are trying to audit *for*. Compute them
    # separately and use the one that matches the item list being audited
    # (EN item list → EN difficulty).
    difficulty_per_lang = {}
    for lang in lang_pair:
        per_item_acc = defaultdict(list)
        for m in models:
            for iid, ok in correctness_by_model_lang[(m, lang)].items():
                per_item_acc[iid].append(int(ok))
        difficulty_per_lang[lang] = {
            iid: (1 - mean(vs) if vs else 0.0) for iid, vs in per_item_acc.items()
        }
    audit_lang = lang_pair[0]
    difficulty = difficulty_per_lang[audit_lang]

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
        "excluded_models": sorted(drop),
        "audit_language": audit_lang,
        "matched_subset_sizes_per_model": {m: len(s) for m, s in matched_per_model.items()},
        "matched_intersection_size": len(matched_intersection),
        "bias_per_model": summary_by_model,
        "bias_intersection": summary_intersection,
        "category_distribution": {"full": dict(full_cat), "matched_intersection": dict(matched_cat)},
        "mean_difficulty_per_lang_full": {
            lang: (mean(difficulty_per_lang[lang].values()) if difficulty_per_lang[lang] else 0.0)
            for lang in lang_pair
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--exclude_models", nargs="+", default=None,
                    help="model keys to drop before the audit")
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day2_selection_bias.json"))
    args = ap.parse_args()

    result = {d: audit(d, exclude_models=args.exclude_models) for d in args.datasets}
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
