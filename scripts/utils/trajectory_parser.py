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
    "answer:", "answer is", "final answer:",
    "correct choice is", "correct answer is",
    "答案:", "答案：", "答案是",
    "答え：", "答え:", "答えは",
    "réponse :", "réponse:",
    "therefore the answer", "so the answer", "所以答案",
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
        # CJK text often has no whitespace after full-stop punctuation, so
        # requiring \s+ after [。.!?！？] collapses whole paragraphs into one
        # "step" for Chinese/Japanese. Split on the punctuation itself via a
        # capture group and reattach it to the preceding segment.
        joined = " ".join(lines)
        parts = re.split(r"([。．.!?！？])", joined)
        sents: List[str] = []
        for i in range(0, len(parts) - 1, 2):
            piece = (parts[i] + parts[i + 1]).strip()
            if piece:
                sents.append(piece)
        if len(parts) % 2 == 1:
            tail_piece = parts[-1].strip()
            if tail_piece:
                sents.append(tail_piece)
        steps = sents

    return {
        "raw": text,
        "body": body,
        "answer_line": tail,
        "steps": steps,
        "n_steps": len(steps),
    }


_LABEL_BOUNDARY_CHARS = set(
    " \t:：'\"()[]{}.,;，。、！？!?<>-—*\n\r"
    "（）「」『』"  # CJK fullwidth parens / brackets
)

# Strip "option" / "选项" / "choice" prefix that is glued to the label so the
# boundary check below sees a clean letter ("答案：选项A" → "答案： A").
_OPTION_PREFIX_RE = re.compile(
    r"(?:选\s*项|option|choice)\s*(?=[A-Za-z]\b|[A-Za-z][^A-Za-z])",
    re.IGNORECASE,
)


def extract_answer(text: str, valid_labels: List[str]) -> str:
    """Return the first valid label that appears in the Answer: tail.

    Requires an explicit answer marker. Short outputs (≤30 chars) with no
    marker are treated as direct answers and searched whole; longer outputs
    without a marker return "" rather than guessing from the reasoning body,
    where letters like "A" or "B" appear routinely in prose and would match
    spuriously.
    """
    _, tail = _strip_answer_tail(text)
    if tail:
        search_space = tail
    elif len(text.strip()) <= 30:
        search_space = text
    else:
        return ""
    search_space = _OPTION_PREFIX_RE.sub(" ", search_space)
    buf = f" {search_space.lower()} "
    for lab in valid_labels:
        ll = lab.lower()
        for k in range(1, len(buf) - len(ll)):
            if buf[k : k + len(ll)] != ll:
                continue
            left = buf[k - 1]
            right = buf[k + len(ll)]
            if left in _LABEL_BOUNDARY_CHARS and right in _LABEL_BOUNDARY_CHARS:
                return lab
    return ""


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A number immediately following an answer marker wins over numbers later
# in the same sentence ("The answer is 7, which is greater than 6." → 7, not 6).
_NUMBER_AFTER_MARKER_RE = re.compile(
    r"(?:answer|final answer|result|total|答案|结果|答え|réponse)"
    r"[^\d\-]{0,12}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Thousand-separator commas: comma followed by EXACTLY 3 digits not followed
# by another digit. "70,000" / "1,234,567" / "2,125" → strip.
_THOUSAND_SEP_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

# Decimal commas (French / Spanish / German convention): comma followed by
# 1-2 digits not followed by another digit. "20,00" / "1,5" / "75,99" →
# convert comma to dot so float() parses. Disambiguated from thousand-comma
# by digit count: thousand needs exactly 3 trailing digits, decimal 1-2.
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")

# Thousand-separator spaces (regular space, NBSP U+00A0, thin U+2009, narrow
# NBSP U+202F). French MGSM commonly writes "70 000"; without this strip the
# regex stops at the first space and returns "70".
_THOUSAND_SEP_SPACE_RE = re.compile(r"(?<=\d)[    ](?=\d{3}(?!\d))")

# CJK myriad unit (10000). "7万" → "70000", "7.5万" → "75000". 千 / 亿 and
# mixed compositions like "2万5千" are not handled — current data only has
# the plain <number>万 form.
_WAN_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*万")


def _strip_thousand_separators(s: str) -> str:
    s = _THOUSAND_SEP_COMMA_RE.sub("", s)         # "70,000" → "70000"
    s = _DECIMAL_COMMA_RE.sub(".", s)             # "20,00"  → "20.00"; "1,5" → "1.5"
    s = _THOUSAND_SEP_SPACE_RE.sub("", s)
    return s


def _expand_wan(s: str) -> str:
    def repl(m):
        val = float(m.group(1)) * 10000
        return str(int(val)) if val.is_integer() else repr(val)
    return _WAN_RE.sub(repl, s)


def normalize_numeric_string(s: str) -> str:
    """Normalize numeric text for comparison: strip thousand separators
    (commas + spaces) and expand the CJK 万 unit. Used by both extraction
    and gold-side comparison so a gold value like "2,125" matches a
    prediction "2125", and a prediction "7万" matches gold "70000"."""
    return _expand_wan(_strip_thousand_separators(s))


def extract_numeric_answer(text: str) -> str:
    """Extract the numeric answer.

    Preference order (the first two only apply when an answer marker exists):
      1. if the answer span contains an expression `=`, take the first
         number after the last `=` — handles "答案：120 + 67 = 187" → 187.
      2. first number immediately following the marker.
      3. last number in the final line (no-marker fallback).
    """
    _, tail = _strip_answer_tail(text)
    search_space = tail if tail else (text.splitlines()[-1] if text.splitlines() else "")
    cleaned = normalize_numeric_string(search_space)

    if tail:
        if "=" in cleaned:
            last_eq = cleaned.rfind("=")
            nums_after = _NUMBER_RE.findall(cleaned[last_eq + 1 :])
            if nums_after:
                return nums_after[0]
        m = _NUMBER_AFTER_MARKER_RE.search(cleaned)
        if m:
            return m.group(1)

    nums = _NUMBER_RE.findall(cleaned)
    return nums[-1] if nums else ""
