"""Parse a generated CoT trajectory into a sequence of reasoning steps.

Steps are identified by a variety of numbering cues (Step 1:, 步骤1：, 1., etc.)
with a fallback to sentence-level splits when no numbering is present.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_STEP_PATTERNS = [
    re.compile(r"^(?:step|Step|STEP)\s*(\d+)\s*[:.\-]?\s*(.*)$"),
    re.compile(r"^步骤\s*(\d+)\s*[：:.\-]?\s*(.*)$"),
    re.compile(r"^ステップ\s*(\d+)\s*[：:.\-]?\s*(.*)$"),
    re.compile(r"^[Ee]tape\s*(\d+)\s*[:.\-]?\s*(.*)$"),
    re.compile(r"^Étape\s*(\d+)\s*[:.\-]?\s*(.*)$"),
    re.compile(r"^(\d{1,2})[\.\)、]\s*(.*)$"),
    re.compile(r"^第\s*(\d+)\s*步[：:.\-]?\s*(.*)$"),
]

_ANSWER_MARKERS = [
    "answer:", "final answer:", "答案:", "答案：", "答え：", "答え:",
    "réponse :", "réponse:", "therefore the answer", "所以答案",
]


def _strip_answer_tail(text: str) -> Tuple[str, str]:
    """Split trajectory body from the final Answer: line."""
    lower = text.lower()
    cut = -1
    for marker in _ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx > cut:
            cut = idx
    if cut < 0:
        return text, ""
    return text[:cut].rstrip(), text[cut:].strip()


def _match_step(line: str) -> int:
    for pat in _STEP_PATTERNS:
        if pat.match(line.strip()):
            return 1
    return 0


def parse_steps(text: str) -> Dict:
    body, tail = _strip_answer_tail(text)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    steps: List[str] = []
    current: List[str] = []

    def flush():
        if current:
            steps.append(" ".join(current).strip())
            current.clear()

    numbered_hits = sum(1 for ln in lines if _match_step(ln))

    if numbered_hits >= 2:
        for ln in lines:
            if _match_step(ln):
                flush()
                current.append(ln)
            else:
                current.append(ln)
        flush()
    else:
        sentence_splitter = re.compile(r"(?<=[。．.!?！？])\s+")
        joined = " ".join(lines)
        sents = [s.strip() for s in sentence_splitter.split(joined) if s.strip()]
        steps = sents

    return {
        "raw": text,
        "body": body,
        "answer_line": tail,
        "steps": steps,
        "n_steps": len(steps),
    }


def extract_answer(text: str, valid_labels: List[str]) -> str:
    """Return the first label in valid_labels that appears after an answer marker."""
    _, tail = _strip_answer_tail(text)
    search_space = tail or text
    search_lower = search_space.lower()
    for lab in valid_labels:
        ll = lab.lower()
        for pattern in (f" {ll}", f":{ll}", f"：{ll}", f"'{ll}'", f'"{ll}"', f"({ll})", f" {ll}."):
            if pattern in f" {search_lower}":
                return lab
    return ""


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str:
    _, tail = _strip_answer_tail(text)
    search_space = tail or text
    nums = _NUMBER_RE.findall(search_space.replace(",", ""))
    return nums[-1] if nums else ""
