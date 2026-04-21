"""Day 3: LaBSE cross-lingual step alignment → unmatched step ratio per item.

Adds unmatched_step_ratio as a fourth simple feature. Outputs per-item
alignment records plus a ranking comparison including the new feature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.alignment import align_steps  # noqa: E402
from utils.config import FEAT_DIR, PILOT, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.ranking import inversion_rate, kendall_tau, rank_models  # noqa: E402


def align_for(dataset: str, model: str, threshold: float) -> list:
    en = read_jsonl(FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl")
    zh_path = FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl"
    if not zh_path.exists():
        return []
    zh = read_jsonl(zh_path)
    en_by_id = {r["id"]: r for r in en}
    zh_by_id = {r["id"]: r for r in zh}
    matched = accuracy_matched_ids(
        {i: bool(r["correct"]) for i, r in en_by_id.items()},
        {i: bool(r["correct"]) for i, r in zh_by_id.items()},
    )
    out = []
    for iid in sorted(matched):
        e = en_by_id[iid]
        z = zh_by_id[iid]
        res = align_steps(e["steps"], z["steps"], threshold=threshold)
        out.append({
            "id": iid, "model": model, "dataset": dataset,
            "n_en": res["n_a"], "n_zh": res["n_b"],
            "unmatched_ratio": res["unmatched_ratio"],
            "mean_match_sim": res["mean_match_sim"],
            "matches": res["matches"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--threshold", type=float, default=PILOT["labse_threshold"])
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day3_alignment.json"))
    args = ap.parse_args()

    if args.models is None:
        models = sorted({p.stem.split("__")[0] for p in FEAT_DIR.glob("*__xcopa__en__cot.jsonl")})
    else:
        models = args.models

    summary = {}
    for dataset in args.datasets:
        per_model_values = {}
        for m in models:
            recs = align_for(dataset, m, args.threshold)
            if not recs:
                continue
            write_jsonl(RANK_DIR / f"alignment__{m}__{dataset}.jsonl", recs)
            per_model_values[m] = mean(r["unmatched_ratio"] for r in recs)

        if not per_model_values:
            continue

        acc_per_model = {}
        for m in per_model_values:
            en = read_jsonl(FEAT_DIR / f"{m}__{dataset}__en__cot.jsonl")
            acc_per_model[m] = mean(r["correct"] for r in en)

        acc_ranks = rank_models(acc_per_model, higher_is_better=True)
        unmatched_ranks = rank_models(per_model_values, higher_is_better=True)

        summary[dataset] = {
            "unmatched_mean_per_model": per_model_values,
            "ranks_unmatched": unmatched_ranks,
            "ranks_accuracy": acc_ranks,
            "kendall_tau_accuracy_vs_unmatched": kendall_tau(
                [acc_ranks[m] for m in sorted(per_model_values)],
                [unmatched_ranks[m] for m in sorted(per_model_values)],
            ),
            "inversion_rate_accuracy_vs_unmatched": inversion_rate(
                [acc_ranks[m] for m in sorted(per_model_values)],
                [unmatched_ranks[m] for m in sorted(per_model_values)],
            ),
        }
    write_json(Path(args.out), summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
