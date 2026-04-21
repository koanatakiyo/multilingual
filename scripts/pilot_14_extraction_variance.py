"""Day 7 morning: extraction variance check.

Runs the hyperedge extractor 5 times on 20 trajectories and reports the
pairwise variance in D_HG. Expected: V_ext < 0.2 * mean cross-lingual D_HG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, PILOT, REPORT_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402

from pilot_10_hyperedge_extraction import extract_hyperedges, stub_judge  # noqa: E402
from pilot_11_hypergraph_divergence import _match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--n_trajectories", type=int, default=20)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=PILOT["ensemble_k"])
    ap.add_argument("--tau", type=float, default=PILOT["judge_tau"])
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day7_extraction_variance.json"))
    args = ap.parse_args()

    en = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__en__cot.jsonl")
    zh = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__zh__cot.jsonl")
    en_by_id = {r["id"]: r for r in en}
    zh_by_id = {r["id"]: r for r in zh}
    matched = sorted(accuracy_matched_ids(
        {i: bool(r["correct"]) for i, r in en_by_id.items()},
        {i: bool(r["correct"]) for i, r in zh_by_id.items()},
    ))[: args.n_trajectories]

    per_item_runs = []
    for iid in matched:
        steps_en = en_by_id[iid]["steps"]
        steps_zh = zh_by_id[iid]["steps"]
        divs = []
        for _ in range(args.n_runs):
            cands_en = [c.__dict__ for c in extract_hyperedges(steps_en, stub_judge, args.k)]
            cands_zh = [c.__dict__ for c in extract_hyperedges(steps_zh, stub_judge, args.k)]
            ea = [{"target": c["target"], "premises": list(c["premises"]),
                   "confidence": min(c["c_forward"], c["c_backward"]), "retained": True}
                  for c in cands_en if min(c["c_forward"], c["c_backward"]) >= args.tau]
            eb = [{"target": c["target"], "premises": list(c["premises"]),
                   "confidence": min(c["c_forward"], c["c_backward"]), "retained": True}
                  for c in cands_zh if min(c["c_forward"], c["c_backward"]) >= args.tau]
            divs.append(1 - _match(ea, eb))
        per_item_runs.append({"id": iid, "runs": divs,
                              "mean": mean(divs), "sd": pstdev(divs) if len(divs) > 1 else 0.0})

    overall_mean = mean(r["mean"] for r in per_item_runs) if per_item_runs else 0.0
    overall_sd = mean(r["sd"] for r in per_item_runs) if per_item_runs else 0.0
    write_json(Path(args.out), {
        "model": args.model, "dataset": args.dataset,
        "n_trajectories": len(per_item_runs), "n_runs": args.n_runs,
        "overall_mean_D_HG": overall_mean,
        "overall_mean_sd_across_runs": overall_sd,
        "per_item": per_item_runs,
        "ratio_variance_to_mean": overall_sd / overall_mean if overall_mean else 0.0,
    })
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
