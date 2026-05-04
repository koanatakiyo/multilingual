"""Day 6 morning: hypergraph divergence D_HG vs. DAG-only divergence D_DAG.

For each item and model, compares EN and ZH hyperedge sets after aligning
steps cross-lingually via LaBSE bipartite matching (chapter 3.2.5). Similarity
combines temperature-scaled target-text consistency and Jaccard of
alignment-mapped premise indices.

D_DAG filters to arity-1 (pairwise) hyperedges — the true pairwise-DAG
baseline — so that D_HG − D_DAG isolates the higher-order-structure signal.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.alignment import _optimal_bipartite, encode  # noqa: E402
from utils.config import HG_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.ranking import inversion_rate, kendall_tau, rank_models  # noqa: E402


_TARGET_TAU = 0.3  # temperature floor for soft target-sim (chapter 3.2.5)


def _soft_target_sim(cos: float) -> float:
    """Temperature-scaled soft target consistency (chapter 3.2.5)."""
    return max(0.0, (cos - _TARGET_TAU) / (1 - _TARGET_TAU))


def _step_alignment(steps_en: List[str], steps_zh: List[str], threshold: float = 0.5):
    """Map EN step index → best-matching ZH step index via LaBSE + Hungarian.

    Returns (en_to_zh: dict[int,int], cos_matrix: np.ndarray of shape (|en|,|zh|)).
    """
    if not steps_en or not steps_zh:
        return {}, np.zeros((len(steps_en), len(steps_zh)), dtype=np.float32)
    emb_en = encode(steps_en)
    emb_zh = encode(steps_zh)
    sim = emb_en @ emb_zh.T
    matches = _optimal_bipartite(sim, threshold)
    return {i: j for (i, j, _s) in matches}, sim


def _hyperedge_sim(
    e_en: dict, e_zh: dict, en_to_zh: Dict[int, int], cos_matrix: np.ndarray
) -> float:
    # Target text similarity with temperature scaling.
    t_en, t_zh = e_en["target"], e_zh["target"]
    if t_en < cos_matrix.shape[0] and t_zh < cos_matrix.shape[1]:
        target_sim = _soft_target_sim(float(cos_matrix[t_en, t_zh]))
    else:
        target_sim = 0.0
    if target_sim <= 0.0:
        return 0.0

    # Map EN premises through alignment; unaligned premises count as "misses".
    mapped = [en_to_zh.get(i) for i in e_en["premises"]]
    mapped_hits = {m for m in mapped if m is not None}
    unaligned = sum(1 for m in mapped if m is None)
    pb = set(e_zh["premises"])
    inter = len(mapped_hits & pb)
    union = len(mapped_hits | pb) + unaligned  # unaligned EN premises penalise sim
    premise_sim = inter / union if union else 1.0
    return target_sim * premise_sim


def _match(edges_en, edges_zh, en_to_zh, cos_matrix) -> float:
    """Confidence-weighted greedy best-match (formula from chapter 3.2.5)."""
    if not edges_en or not edges_zh:
        return 0.0
    used = set()
    total = 0.0
    for ea in edges_en:
        best_j, best_s = -1, 0.0
        for j, eb in enumerate(edges_zh):
            if j in used:
                continue
            s = _hyperedge_sim(ea, eb, en_to_zh, cos_matrix) * min(ea["confidence"], eb["confidence"])
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            used.add(best_j)
            total += best_s
    return total / max(len(edges_en), len(edges_zh))


def _per_lang(records):
    """Group records by item id and language. Each value has steps + hyperedges."""
    by = defaultdict(dict)
    for r in records:
        by[r["id"]][r["lang"]] = {
            "steps": r.get("steps", []),
            "edges": [e for e in r["hyperedges"] if e.get("retained")],
        }
    return by


def divergence_for(records, dag_only: bool) -> Dict[str, float]:
    """Per-item divergence. dag_only=True restricts to arity-1 pairwise edges."""
    grouped = _per_lang(records)
    divs: Dict[str, float] = {}
    for iid, pair in grouped.items():
        if "en" not in pair or "zh" not in pair:
            continue
        en = pair["en"]
        zh = pair["zh"]
        edges_en = [e for e in en["edges"] if (len(e["premises"]) == 1 if dag_only else True)]
        edges_zh = [e for e in zh["edges"] if (len(e["premises"]) == 1 if dag_only else True)]
        en_to_zh, cos = _step_alignment(en["steps"], zh["steps"])
        o = _match(edges_en, edges_zh, en_to_zh, cos)
        divs[iid] = 1 - o
    return divs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day6_hypergraph.json"))
    args = ap.parse_args()

    per_model_full: Dict[str, float] = {}
    per_model_dag: Dict[str, float] = {}
    per_item_full: Dict[str, Dict[str, float]] = {}
    for p in HG_DIR.glob(f"hg__*__{args.dataset}.jsonl"):
        model = p.stem.split("__")[1]
        records = read_jsonl(p)
        d_full = divergence_for(records, dag_only=False)
        d_dag = divergence_for(records, dag_only=True)
        per_model_full[model] = mean(d_full.values()) if d_full else 0.0
        per_model_dag[model] = mean(d_dag.values()) if d_dag else 0.0
        per_item_full[model] = d_full

    if not per_model_full:
        print("no hypergraph files found")
        return

    ranks_hg = rank_models(per_model_full, higher_is_better=True)
    ranks_dag = rank_models(per_model_dag, higher_is_better=True)
    models = sorted(per_model_full)
    hg_dag_gap = {m: per_model_full[m] - per_model_dag[m] for m in models}
    result = {
        "dataset": args.dataset,
        "D_HG_per_model": per_model_full,
        "D_DAG_per_model": per_model_dag,
        "D_HG_minus_D_DAG_per_model": hg_dag_gap,
        "mean_D_HG_minus_D_DAG": mean(hg_dag_gap.values()),
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
