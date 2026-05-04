"""Day 7 morning: extraction variance check.

Runs the hyperedge extractor 5 times on 20 trajectories and reports the
pairwise variance in D_HG. Expected: V_ext < 0.2 * mean cross-lingual D_HG.

Valid variance measurements require a stochastic judge. The deterministic
stub returns the same score on every call and would under-report variance to
~0. This script therefore refuses to run with the stub unless `--allow-stub`
is explicitly set, and even then it threads a per-run seed into the stub so
the pipeline-test numbers are at least non-trivial.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, PILOT, REPORT_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402

from pilot_10_hyperedge_extraction import (  # noqa: E402
    anthropic_backward_factory, anthropic_forward_factory,
    extract_hyperedges,
    openai_backward_factory, openai_forward_factory,
    stub_backward, stub_forward,
)
from pilot_11_hypergraph_divergence import _match, _step_alignment  # noqa: E402


def _edges(cands, tau):
    return [
        {"target": c.target, "premises": list(c.premises),
         "confidence": min(c.c_forward, c.c_backward), "retained": True}
        for c in cands
        if min(c.c_forward, c.c_backward) >= tau
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--judge", choices=["stub", "openai", "anthropic"], default="openai")
    ap.add_argument("--judge_model", default="gpt-5.4-mini")
    ap.add_argument("--n_trajectories", type=int, default=20)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=PILOT["ensemble_k"])
    ap.add_argument("--tau", type=float, default=PILOT["judge_tau"])
    ap.add_argument("--allow-stub", action="store_true",
                    help="permit stub judge for pipeline tests only")
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day7_extraction_variance.json"))
    args = ap.parse_args()

    if args.judge == "openai":
        forward_judge = openai_forward_factory(args.judge_model)
        backward_judge = openai_backward_factory(args.judge_model)
    elif args.judge == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set")
        forward_judge = anthropic_forward_factory(args.judge_model)
        backward_judge = anthropic_backward_factory(args.judge_model)
    else:
        if not args.allow_stub:
            raise SystemExit(
                "Refusing to run variance check with stub judge (would under-report "
                "variance). Pass --allow-stub for pipeline testing, or --judge "
                "openai/anthropic for valid measurement."
            )
        forward_judge = stub_forward
        backward_judge = stub_backward

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
        en_to_zh, cos = _step_alignment(steps_en, steps_zh)
        divs = []
        for run_idx in range(args.n_runs):
            cands_en = extract_hyperedges(steps_en, forward_judge, backward_judge,
                                           args.k, run_seed=run_idx * 2)
            cands_zh = extract_hyperedges(steps_zh, forward_judge, backward_judge,
                                           args.k, run_seed=run_idx * 2 + 1)
            divs.append(1 - _match(_edges(cands_en, args.tau), _edges(cands_zh, args.tau),
                                    en_to_zh, cos))
        per_item_runs.append({
            "id": iid, "runs": divs,
            "mean": mean(divs), "sd": pstdev(divs) if len(divs) > 1 else 0.0,
        })

    overall_mean = mean(r["mean"] for r in per_item_runs) if per_item_runs else 0.0
    overall_sd = mean(r["sd"] for r in per_item_runs) if per_item_runs else 0.0
    write_json(Path(args.out), {
        "model": args.model, "dataset": args.dataset, "judge": args.judge,
        "n_trajectories": len(per_item_runs), "n_runs": args.n_runs,
        "overall_mean_D_HG": overall_mean,
        "overall_mean_sd_across_runs": overall_sd,
        "ratio_variance_to_mean": overall_sd / overall_mean if overall_mean else 0.0,
        "per_item": per_item_runs,
    })
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
