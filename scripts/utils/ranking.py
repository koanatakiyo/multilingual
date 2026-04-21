"""Ranking utilities: Kendall tau, inversion rate, CRSI."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence


def kendall_tau(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    n = len(ranks_a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i, j in combinations(range(n), 2):
        da = ranks_a[i] - ranks_a[j]
        db = ranks_b[i] - ranks_b[j]
        if da == 0 or db == 0:
            continue
        if (da > 0) == (db > 0):
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def inversion_rate(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    n = len(ranks_a)
    if n < 2:
        return 0.0
    inv = total = 0
    for i, j in combinations(range(n), 2):
        da = ranks_a[i] - ranks_a[j]
        db = ranks_b[i] - ranks_b[j]
        if da == 0 or db == 0:
            continue
        total += 1
        if (da > 0) != (db > 0):
            inv += 1
    return inv / total if total else 0.0


def rank_models(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, int]:
    """Return 1-indexed ranks by value. Higher value ⇒ rank 1 if higher_is_better."""
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    return {name: i + 1 for i, (name, _) in enumerate(items)}


def crsi(accuracy_ranks: Dict[str, int], feature_ranks_by_feat: Dict[str, Dict[str, int]]) -> float:
    """Mean Kendall tau between accuracy ranking and each feature ranking."""
    if not feature_ranks_by_feat:
        return 1.0
    models = sorted(accuracy_ranks.keys())
    acc = [accuracy_ranks[m] for m in models]
    taus: List[float] = []
    for _, ranks in feature_ranks_by_feat.items():
        fr = [ranks[m] for m in models]
        taus.append(kendall_tau(acc, fr))
    return sum(taus) / len(taus)
