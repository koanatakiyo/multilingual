"""Simple per-trajectory process features.

Feature list:
    - step_count
    - procedural_count / procedural_rate   (forward-reasoning connectives)
    - epistemic_count  / epistemic_rate    (uncertainty / metacognitive control)
    - dependency_depth (language-neutral content-unit overlap chain)
    - avg_step_tokens

The marker split follows Chapter 3.2.3: reporting procedural and epistemic
separately lets us check whether cross-lingual divergence is concentrated in
uncertainty management, rather than in forward-reasoning style.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from .config import PILOT


_DEP_SIM_THRESHOLD = 0.10


_CJK_RE = re.compile(r"[㐀-鿿぀-ゟ゠-ヿ㇀-㇯]")
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


def _count_marker_hits(steps: List[str], lang: str, marker_type: str) -> int:
    """Count steps containing at least one marker of the given type."""
    table = PILOT[marker_type]
    kw = [k.lower() for k in table.get(lang, table["en"])]
    hits = 0
    for s in steps:
        s_low = s.lower()
        if any(k in s_low for k in kw):
            hits += 1
    return hits


def _content_units(s: str) -> Set[str]:
    """Language-neutral content units.

    - CJK characters are each treated as a content unit (Chinese/Japanese
      content words are typically 1–2 chars; single chars carry content).
    - Latin-script runs of ≥3 characters are treated as content words (this
      filters function words like "a"/"is" that create noise).
    All units are lowercased.
    """
    units: Set[str] = set()
    for ch in s:
        if _CJK_RE.match(ch):
            units.add(ch)
    for tok in _TOKEN_RE.findall(s.lower()):
        if len(tok) >= 3:
            units.add(tok)
    return units


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _dependency_depth(steps: List[str]) -> int:
    """Longest chain of consecutive steps that share content with the previous step.

    Language-fair: the content-unit set treats CJK characters and Latin words
    uniformly, so neither script is systematically advantaged.
    """
    if not steps:
        return 0
    units = [_content_units(s) for s in steps]
    run = best = 1
    for i in range(1, len(steps)):
        if _jaccard(units[i - 1], units[i]) >= _DEP_SIM_THRESHOLD:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _avg_step_tokens(steps: List[str]) -> float:
    if not steps:
        return 0.0
    lengths = []
    for s in steps:
        # Count whitespace tokens for EN/FR, fall back to char count for CJK.
        ws = re.findall(r"\S+", s)
        has_cjk = any("㐀" <= ch <= "鿿" for ch in s)
        lengths.append(len([c for c in s if not c.isspace()]) if has_cjk else len(ws))
    return sum(lengths) / len(lengths)


def extract_features(parsed: Dict, lang: str) -> Dict[str, float]:
    steps: List[str] = parsed["steps"]
    n = len(steps)
    proc = _count_marker_hits(steps, lang, "procedural_markers")
    epi = _count_marker_hits(steps, lang, "epistemic_markers")
    d = _dependency_depth(steps)
    return {
        "step_count": float(n),
        "procedural_count": float(proc),
        "procedural_rate": float(proc / n) if n else 0.0,
        "epistemic_count": float(epi),
        "epistemic_rate": float(epi / n) if n else 0.0,
        "dependency_depth": float(d),
        "avg_step_tokens": _avg_step_tokens(steps),
    }
