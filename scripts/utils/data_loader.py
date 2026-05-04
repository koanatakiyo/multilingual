"""Unified loaders for XCOPA, XStoryCloze, MGSM, Belebele, and anchor items.

Each loader returns a list of dicts with a consistent schema:
    {"id": str, "dataset": str, "lang": str, "prompt_payload": dict, "gold": str,
     "align_key": <dataset-specific key>}

align_key is used by load_parallel_pair to guarantee cross-lingual index
correspondence (same underlying item in each language). Position-based zipping
is unsafe because HF subsets are not guaranteed to preserve ordering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATASETS


def _read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_xcopa(lang: str, split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    path = DATASETS["xcopa"]["local_dir"] / f"{lang}_{split}.jsonl"
    raw = _read_jsonl(path)
    out: List[Dict] = []
    for i, r in enumerate(raw):
        label = int(r["label"])
        gold = "A" if label == 0 else "B" if label == 1 else ""
        align_key = int(r.get("idx", i))
        out.append({
            "id": f"xcopa-{split}-{align_key}",
            "dataset": "xcopa",
            "lang": lang,
            "prompt_payload": {
                "premise": r["premise"],
                "choice1": r["choice1"],
                "choice2": r["choice2"],
                "question": r.get("question", "cause"),
            },
            "gold": gold,
            "align_key": align_key,
        })
    return out[:limit] if limit else out


def load_xstorycloze(lang: str, split: str = "eval", limit: Optional[int] = None) -> List[Dict]:
    path = DATASETS["xstorycloze"]["local_dir"] / f"{lang}_{split}.jsonl"
    raw = _read_jsonl(path)
    out: List[Dict] = []
    for i, r in enumerate(raw):
        gold = "A" if int(r["answer_right_ending"]) == 1 else "B"
        align_key = str(r.get("story_id", i))
        out.append({
            "id": f"xstorycloze-{split}-{align_key}",
            "dataset": "xstorycloze",
            "lang": lang,
            "prompt_payload": {
                "input_sentence_1": r["input_sentence_1"],
                "input_sentence_2": r["input_sentence_2"],
                "input_sentence_3": r["input_sentence_3"],
                "input_sentence_4": r["input_sentence_4"],
                "sentence_quiz1": r["sentence_quiz1"],
                "sentence_quiz2": r["sentence_quiz2"],
            },
            "gold": gold,
            "align_key": align_key,
        })
    return out[:limit] if limit else out


def load_mgsm(lang: str, split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    path = DATASETS["mgsm"]["local_dir"] / f"{lang}_{split}.jsonl"
    raw = _read_jsonl(path)
    out: List[Dict] = []
    for i, r in enumerate(raw):
        out.append({
            "id": f"mgsm-{split}-{i}",
            "dataset": "mgsm",
            "lang": lang,
            "prompt_payload": {"question": r["question"]},
            "gold": str(r["answer_number"]),
            "align_key": i,
        })
    return out[:limit] if limit else out


def load_belebele(lang: str, split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    path = DATASETS["belebele"]["local_dir"] / f"{lang}_{split}.jsonl"
    raw = _read_jsonl(path)
    labels = ["A", "B", "C", "D"]
    out: List[Dict] = []
    for i, r in enumerate(raw):
        correct = int(r["correct_answer_num"]) - 1
        # question_number is 1 or 2 *within a passage*, so it's not unique on
        # its own. (link, question_number) identifies a problem globally.
        align_key = f"{r.get('link', '')}::{r.get('question_number', i)}"
        out.append({
            "id": f"belebele-{split}-{i}",
            "dataset": "belebele",
            "lang": lang,
            "prompt_payload": {
                "flores_passage": r["flores_passage"],
                "question": r["question"],
                "mc_answer1": r["mc_answer1"],
                "mc_answer2": r["mc_answer2"],
                "mc_answer3": r["mc_answer3"],
                "mc_answer4": r["mc_answer4"],
            },
            "gold": labels[correct],
            "align_key": align_key,
        })
    return out[:limit] if limit else out


LOADERS = {
    "xcopa": load_xcopa,
    "xstorycloze": load_xstorycloze,
    "mgsm": load_mgsm,
    "belebele": load_belebele,
}


def load_dataset(name: str, lang: str, split: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    loader = LOADERS[name]
    if split is None:
        return loader(lang, limit=limit)
    return loader(lang, split=split, limit=limit)


def load_parallel_pair(name: str, lang_a: str, lang_b: str, split: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict]:
    """Return paired items aligned by align_key (same underlying problem).

    Raises ValueError if the two languages do not share the same key universe
    so downstream callers can't silently compare mismatched items.
    """
    a = load_dataset(name, lang_a, split=split)
    b = load_dataset(name, lang_b, split=split)
    a_map = {it["align_key"]: it for it in a}
    b_map = {it["align_key"]: it for it in b}
    shared = sorted(a_map.keys() & b_map.keys(), key=lambda k: (isinstance(k, str), k))
    if not shared:
        raise ValueError(f"No shared align_key between {name}/{lang_a} and {name}/{lang_b}")
    only_a = set(a_map) - set(b_map)
    only_b = set(b_map) - set(a_map)
    if only_a or only_b:
        raise ValueError(
            f"align_key mismatch for {name}: {len(only_a)} in {lang_a} only, "
            f"{len(only_b)} in {lang_b} only. Fix the download before proceeding."
        )
    pairs = [{"a": a_map[k], "b": b_map[k]} for k in shared]
    return pairs[:limit] if limit else pairs
