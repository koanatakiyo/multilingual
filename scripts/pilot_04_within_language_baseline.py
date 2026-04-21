"""Day 1 evening: within-language resampling baseline.

Split EN matched features into halves A/B, bootstrap 10 times, and compare
the resulting within-language feature divergence against the cross-lingual
divergence from pilot_03.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, PILOT, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, read_json, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402

FEATURE_NAMES = ["step_count", "verification_rate", "dependency_depth", "avg_step_tokens"]


def _load_pair(dataset: str, model: str):
    en = list((FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl").open() if False else read_jsonl(FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl"))
    zh_path = FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl"
    zh = read_jsonl(zh_path) if zh_path.exists() else []
    return en, zh


def within_language_baseline(dataset: str, models, iters: int = None) -> dict:
    iters = iters or PILOT["bootstrap_iters"]
    per_model = {}
    for m in models:
        en, zh = _load_pair(dataset, m)
        if not en or not zh:
            continue
        matched = accuracy_matched_ids(
            {r["id"]: bool(r["correct"]) for r in en},
            {r["id"]: bool(r["correct"]) for r in zh},
        )
        en_matched = [r for r in en if r["id"] in matched]
        if len(en_matched) < 4:
            continue
        feature_divergences = defaultdict(list)
        rng = random.Random(42)
        for _ in range(iters):
            shuffled = en_matched[:]
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            a, b = shuffled[:half], shuffled[half:]
            for f in FEATURE_NAMES:
                ma = mean(r["features"][f] for r in a)
                mb = mean(r["features"][f] for r in b)
                feature_divergences[f].append(abs(ma - mb))
        per_model[m] = {
            f: {"mean": mean(v), "sd": pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for f, v in feature_divergences.items()
        }
    return per_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa"])
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: infer from features dir")
    ap.add_argument("--rankings_file", default=str(RANK_DIR / "pilot_day1_rankings.json"))
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day1_within_language_baseline.json"))
    args = ap.parse_args()

    if args.models is None:
        models = sorted({p.stem.split("__")[0] for p in FEAT_DIR.glob("*__*__en__cot.jsonl")})
    else:
        models = args.models

    result = {}
    for d in args.datasets:
        result[d] = within_language_baseline(d, models)

    if Path(args.rankings_file).exists():
        rankings = read_json(Path(args.rankings_file))
        comparisons = {}
        for d in args.datasets:
            if d not in rankings or not rankings[d]:
                continue
            cross = rankings[d]["mean_abs_divergence_per_feature"]
            within = result.get(d, {})
            per_feat = {}
            for f in FEATURE_NAMES:
                within_vals = [within[m][f]["mean"] for m in within if f in within[m]]
                per_feat[f] = {
                    "cross_lingual_mean_abs_div": cross.get(f, 0.0),
                    "within_language_mean_abs_div": mean(within_vals) if within_vals else 0.0,
                }
            comparisons[d] = per_feat
        result["cross_vs_within_comparison"] = comparisons

    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
