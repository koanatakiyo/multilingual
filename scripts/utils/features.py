"""Simple per-trajectory process features.

Feature list:
    - step_count
    - verification_count (keyword-based)
    - verification_rate = verification_count / step_count
    - dependency_depth (heuristic longest anaphora chain)
    - avg_step_tokens
"""
from __future__ import annotations

import re
from typing import Dict, List

from .config import PILOT

_ANAPHORA_CUES = {
    "en": ["this", "that", "these", "those", "it", "so", "then", "therefore", "thus", "hence"],
    "zh": ["这", "那", "此", "因此", "所以", "于是", "故", "由此", "进而"],
    "ja": ["これ", "それ", "その", "この", "よって", "したがって", "ゆえに"],
    "fr": ["cela", "ce", "donc", "ainsi", "par conséquent", "alors"],
}


def _count_verification_hits(steps: List[str], lang: str) -> int:
    kw = PILOT["verification_keywords"].get(lang, PILOT["verification_keywords"]["en"])
    hits = 0
    for s in steps:
        s_low = s.lower()
        for k in kw:
            if k.lower() in s_low:
                hits += 1
                break
    return hits


def _dependency_depth(steps: List[str], lang: str) -> int:
    """Longest run of consecutive steps that start with an anaphoric cue.

    Heuristic proxy for linear dependency depth without full parsing.
    """
    cues = _ANAPHORA_CUES.get(lang, _ANAPHORA_CUES["en"])
    run = best = 1 if steps else 0
    for i in range(1, len(steps)):
        s = steps[i].strip().lower()
        if any(s.startswith(c.lower()) for c in cues):
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _avg_step_tokens(steps: List[str]) -> float:
    if not steps:
        return 0.0
    lengths = [len(re.findall(r"\S+", s)) for s in steps]
    return sum(lengths) / len(lengths)


def extract_features(parsed: Dict, lang: str) -> Dict[str, float]:
    steps: List[str] = parsed["steps"]
    n = len(steps)
    v = _count_verification_hits(steps, lang)
    d = _dependency_depth(steps, lang)
    return {
        "step_count": float(n),
        "verification_count": float(v),
        "verification_rate": float(v / n) if n else 0.0,
        "dependency_depth": float(d),
        "avg_step_tokens": _avg_step_tokens(steps),
    }
