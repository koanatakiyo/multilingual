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


def rank_models(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    """Return 1-indexed average ranks by value.

    Higher value receives rank 1 when higher_is_better. Tied values receive the
    same average rank so Kendall tau can treat them as true ties.
    """
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j][1] == items[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for name, _ in items[i:j]:
            ranks[name] = rank
        i = j
    return ranks


def crsi(accuracy_ranks: Dict[str, float], feature_ranks_by_feat: Dict[str, Dict[str, float]]) -> float:
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
