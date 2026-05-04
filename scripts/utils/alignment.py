"""LaBSE-based cross-lingual step alignment and unmatched-step ratio.

Uses optimal (Hungarian / linear_sum_assignment) maximum-weight bipartite
matching, not the greedy fallback, so the unmatched ratio is not inflated by
greedy miss-assignments.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import PILOT

_MODEL = None
_TOKENIZER = None


def _load_labse():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    import torch  # type: ignore
    name = "sentence-transformers/LaBSE"
    _TOKENIZER = AutoTokenizer.from_pretrained(name)
    _MODEL = AutoModel.from_pretrained(name)
    _MODEL.eval()
    if torch.cuda.is_available():
        _MODEL.cuda()
    return _MODEL, _TOKENIZER


def encode(sentences: List[str]) -> np.ndarray:
    import torch  # type: ignore
    model, tok = _load_labse()
    if not sentences:
        return np.zeros((0, 768), dtype=np.float32)
    enc = tok(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    embs = out.last_hidden_state[:, 0]
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy().astype(np.float32)


def _optimal_bipartite(sim: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """Maximum-weight bipartite matching via Hungarian assignment.

    Pairs with similarity below `threshold` are dropped from the final matching.
    """
    from scipy.optimize import linear_sum_assignment  # type: ignore

    m, n = sim.shape
    if m == 0 or n == 0:
        return []
    # linear_sum_assignment minimises cost; convert similarity → cost.
    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, float]] = []
    for i, j in zip(row_ind, col_ind):
        s = float(sim[i, j])
        if s >= threshold:
            matches.append((int(i), int(j), s))
    return matches


def align_steps(steps_a: List[str], steps_b: List[str], threshold: float = None) -> Dict:
    threshold = threshold if threshold is not None else PILOT["labse_threshold"]
    if not steps_a or not steps_b:
        return {
            "matches": [],
            "n_a": len(steps_a),
            "n_b": len(steps_b),
            "unmatched_ratio": 1.0,
            "mean_match_sim": 0.0,
        }
    emb_a = encode(steps_a)
    emb_b = encode(steps_b)
    sim = emb_a @ emb_b.T
    matches = _optimal_bipartite(sim, threshold)
    m, n = len(steps_a), len(steps_b)
    unmatched = 1 - (2 * len(matches)) / (m + n)
    return {
        "matches": matches,
        "n_a": m,
        "n_b": n,
        "unmatched_ratio": float(unmatched),
        "mean_match_sim": float(np.mean([s for _, _, s in matches])) if matches else 0.0,
    }
