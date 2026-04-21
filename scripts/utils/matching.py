"""Accuracy-matched subset construction and selection-bias audit helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Set


def accuracy_matched_ids(results_lang_a: Dict[str, bool], results_lang_b: Dict[str, bool]) -> Set[str]:
    """Items correct in both languages for a given model."""
    ids_a = {i for i, ok in results_lang_a.items() if ok}
    ids_b = {i for i, ok in results_lang_b.items() if ok}
    return ids_a & ids_b


def intersection_across_models(matched_sets: Iterable[Set[str]]) -> Set[str]:
    sets = list(matched_sets)
    if not sets:
        return set()
    acc = sets[0].copy()
    for s in sets[1:]:
        acc &= s
    return acc


def bias_audit_summary(matched_ids: Set[str], full_items: List[Dict],
                       difficulty: Dict[str, float]) -> Dict:
    """Compare matched subset vs. full set on length and difficulty.

    difficulty[item_id] is the mean accuracy across all models (0..1).
    """
    def _lens(items: List[Dict]) -> List[int]:
        out = []
        for it in items:
            pp = it["prompt_payload"]
            txt = " ".join(str(v) for v in pp.values())
            out.append(len(txt.split()))
        return out

    matched_items = [it for it in full_items if it["id"] in matched_ids]
    full_lens = _lens(full_items)
    matched_lens = _lens(matched_items) if matched_items else [0]
    full_diff = [difficulty.get(it["id"], 0.0) for it in full_items]
    matched_diff = [difficulty.get(it["id"], 0.0) for it in matched_items] if matched_items else [0.0]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_full": len(full_items),
        "n_matched": len(matched_items),
        "mean_len_full": _mean(full_lens),
        "mean_len_matched": _mean(matched_lens),
        "mean_difficulty_full": _mean(full_diff),
        "mean_difficulty_matched": _mean(matched_diff),
    }
