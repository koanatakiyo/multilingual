"""LaBSE-based cross-lingual step alignment and unmatched-step ratio.

The encoder is loaded lazily so scripts that do not need LaBSE avoid the cost.
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


def _greedy_bipartite(sim: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    m, n = sim.shape
    matches: List[Tuple[int, int, float]] = []
    used_a, used_b = set(), set()
    flat = [(sim[i, j], i, j) for i in range(m) for j in range(n)]
    flat.sort(reverse=True)
    for s, i, j in flat:
        if s < threshold:
            break
        if i in used_a or j in used_b:
            continue
        matches.append((i, j, float(s)))
        used_a.add(i)
        used_b.add(j)
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
    matches = _greedy_bipartite(sim, threshold)
    m, n = len(steps_a), len(steps_b)
    unmatched = 1 - (2 * len(matches)) / (m + n)
    return {
        "matches": matches,
        "n_a": m,
        "n_b": n,
        "unmatched_ratio": float(unmatched),
        "mean_match_sim": float(np.mean([s for _, _, s in matches])) if matches else 0.0,
    }
