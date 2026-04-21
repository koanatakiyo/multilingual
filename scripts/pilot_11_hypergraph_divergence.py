"""Day 6 morning: hypergraph divergence D_HG vs. DAG-only divergence D_DAG.

For each item and model, computes confidence-weighted hyperedge matching
between EN and ZH graphs, then reports whether D_HG identifies ranking
inversions that simple features miss.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import HG_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.ranking import inversion_rate, kendall_tau, rank_models  # noqa: E402


def _hyperedge_key(e: dict, arity_only: bool) -> Tuple:
    if arity_only:
        return (len(e["premises"]),)
    return (e["target"], tuple(e["premises"]))


def _sim(e_a: dict, e_b: dict) -> float:
    pa = set(e_a["premises"])
    pb = set(e_b["premises"])
    if not pa and not pb:
        return 1.0
    jac = len(pa & pb) / len(pa | pb)
    arity_sim = 1 - abs(len(pa) - len(pb)) / max(1, max(len(pa), len(pb)))
    return 0.5 * (jac + arity_sim)


def _match(edges_a, edges_b) -> float:
    if not edges_a or not edges_b:
        return 0.0
    used_b = set()
    total = 0.0
    for ea in edges_a:
        best_j, best_s = -1, 0.0
        for j, eb in enumerate(edges_b):
            if j in used_b:
                continue
            s = _sim(ea, eb) * min(ea["confidence"], eb["confidence"])
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            used_b.add(best_j)
            total += best_s
    return total / max(len(edges_a), len(edges_b))


def divergence_for(records, arity_only: bool) -> Dict[str, float]:
    grouped = defaultdict(lambda: {"en": [], "zh": []})
    for r in records:
        grouped[r["id"]][r["lang"]] = [e for e in r["hyperedges"] if e["retained"]]
        if arity_only:
            grouped[r["id"]][r["lang"]] = [e for e in grouped[r["id"]][r["lang"]] if len(e["premises"]) >= 2]
    divs = {}
    for iid, pair in grouped.items():
        o = _match(pair["en"], pair["zh"])
        divs[iid] = 1 - o
    return divs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day6_hypergraph.json"))
    args = ap.parse_args()

    per_model_full = {}
    per_model_dag = {}
    for p in HG_DIR.glob(f"hg__*__{args.dataset}.jsonl"):
        model = p.stem.split("__")[1]
        records = read_jsonl(p)
        d_full = divergence_for(records, arity_only=False)
        d_dag = divergence_for(records, arity_only=True)
        per_model_full[model] = mean(d_full.values()) if d_full else 0.0
        per_model_dag[model] = mean(d_dag.values()) if d_dag else 0.0

    if not per_model_full:
        print("no hypergraph files found")
        return

    ranks_hg = rank_models(per_model_full, higher_is_better=True)
    ranks_dag = rank_models(per_model_dag, higher_is_better=True)
    models = sorted(per_model_full)
    result = {
        "dataset": args.dataset,
        "D_HG_per_model": per_model_full,
        "D_DAG_per_model": per_model_dag,
        "ranks_D_HG": ranks_hg,
        "ranks_D_DAG": ranks_dag,
        "kendall_tau_HG_vs_DAG": kendall_tau(
            [ranks_hg[m] for m in models], [ranks_dag[m] for m in models],
        ),
        "inversion_rate_HG_vs_DAG": inversion_rate(
            [ranks_hg[m] for m in models], [ranks_dag[m] for m in models],
        ),
    }
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
