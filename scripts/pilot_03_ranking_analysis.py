"""Day 1 afternoon: ranking comparison (survival test + domain contrast + multi-language).

Reads output/features/*.jsonl and produces output/rankings/pilot_day1_rankings.json
with per-dataset rankings, Kendall tau, inversion rate, CRSI, and EN vs ZH
per-feature divergence on accuracy-matched subsets.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.ranking import crsi, inversion_rate, kendall_tau, rank_models  # noqa: E402

FEATURE_NAMES = ["step_count", "verification_rate", "dependency_depth", "avg_step_tokens"]


def _load_feats(dataset: str, mode: str = "cot"):
    groups = defaultdict(list)
    for p in FEAT_DIR.glob(f"*__{dataset}__*__{mode}.jsonl"):
        parts = p.stem.split("__")
        model_key, _, lang, mode_ = parts
        for r in read_jsonl(p):
            groups[(model_key, lang)].append(r)
    return groups


def _per_item_correctness(rows):
    return {r["id"]: bool(r["correct"]) for r in rows}


def analyze_dataset(dataset: str):
    groups = _load_feats(dataset, mode="cot")
    models = sorted({k[0] for k in groups.keys()})
    langs = sorted({k[1] for k in groups.keys()})
    if "en" not in langs:
        return None

    accuracies = {m: {} for m in models}
    for (m, lang), rows in groups.items():
        accuracies[m][lang] = mean(r["correct"] for r in rows) if rows else 0.0

    en_rank_values = {m: accuracies[m].get("en", 0.0) for m in models}
    accuracy_ranks = rank_models(en_rank_values, higher_is_better=True)

    feature_values_per_lang = {lang: {f: {} for f in FEATURE_NAMES} for lang in langs}
    divergences = {f: {} for f in FEATURE_NAMES}
    matched_sizes = {}

    for m in models:
        rows_by_lang = {lang: groups.get((m, lang), []) for lang in langs}
        if "en" in rows_by_lang and "zh" in rows_by_lang:
            a = _per_item_correctness(rows_by_lang["en"])
            b = _per_item_correctness(rows_by_lang["zh"])
            matched = accuracy_matched_ids(a, b)
        else:
            matched = set()
        matched_sizes[m] = len(matched)

        for lang in langs:
            rows = [r for r in rows_by_lang.get(lang, []) if not matched or r["id"] in matched]
            for f in FEATURE_NAMES:
                vals = [r["features"][f] for r in rows]
                feature_values_per_lang[lang][f][m] = mean(vals) if vals else 0.0

        if "en" in langs and "zh" in langs:
            for f in FEATURE_NAMES:
                en = feature_values_per_lang["en"][f][m]
                zh = feature_values_per_lang["zh"][f][m]
                divergences[f][m] = abs(en - zh)

    feature_ranks_en = {
        f: rank_models(feature_values_per_lang["en"][f], higher_is_better=True)
        for f in FEATURE_NAMES
    }
    taus = {
        f: kendall_tau(
            [accuracy_ranks[m] for m in models],
            [feature_ranks_en[f][m] for m in models],
        )
        for f in FEATURE_NAMES
    }
    inversions = {
        f: inversion_rate(
            [accuracy_ranks[m] for m in models],
            [feature_ranks_en[f][m] for m in models],
        )
        for f in FEATURE_NAMES
    }
    crsi_value = crsi(accuracy_ranks, feature_ranks_en)

    return {
        "dataset": dataset,
        "models": models,
        "langs": langs,
        "accuracy_per_lang": accuracies,
        "accuracy_rank_en": accuracy_ranks,
        "feature_values_per_lang": feature_values_per_lang,
        "feature_ranks_en": feature_ranks_en,
        "kendall_tau_vs_accuracy": taus,
        "inversion_rate_vs_accuracy": inversions,
        "crsi": crsi_value,
        "mean_abs_divergence_per_feature": {
            f: (mean(divergences[f].values()) if divergences[f] else 0.0)
            for f in FEATURE_NAMES
        },
        "per_model_divergence": divergences,
        "matched_subset_sizes": matched_sizes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day1_rankings.json"))
    args = ap.parse_args()

    result = {d: analyze_dataset(d) for d in args.datasets}

    if "xcopa" in result and "mgsm" in result and result["xcopa"] and result["mgsm"]:
        x = result["xcopa"]["mean_abs_divergence_per_feature"]
        m = result["mgsm"]["mean_abs_divergence_per_feature"]
        result["domain_contrast"] = {
            f: {"xcopa": x.get(f, 0.0), "mgsm": m.get(f, 0.0),
                 "ratio_xcopa_over_mgsm": (x.get(f, 0.0) / m[f]) if m.get(f, 0.0) else None}
            for f in FEATURE_NAMES
        }

    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
