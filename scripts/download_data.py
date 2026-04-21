"""Download all benchmark datasets into data/ as normalized JSONL files.

Sources (verified 2026-04):
  - XCOPA (EN): super_glue / copa  (validation: 100, test: 500)
  - XCOPA (ZH): xcopa / zh          (validation: 100, test: 500)
  - XStoryCloze: juletxara/xstory_cloze, configs {en, zh}, split=eval
  - MGSM: raw TSV from github.com/google-research/url-nlp/mgsm (en, zh, ja, fr)
  - Belebele: facebook/belebele, configs {eng_Latn, zho_Hans, jpn_Jpan, fra_Latn}

Output schema (JSONL, one row per item):
  XCOPA: {idx, premise, choice1, choice2, question, label}
  XStoryCloze: {story_id, input_sentence_1..4, sentence_quiz1, sentence_quiz2, answer_right_ending}
  MGSM: {question, answer_number}
  Belebele: {question_number, flores_passage, question, mc_answer1..4, correct_answer_num}
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import DATASETS  # noqa: E402


def _json_default(o):
    import datetime as _dt
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    return str(o)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def _hf_load(name: str, subset: str, split: str):
    from datasets import load_dataset  # type: ignore
    return load_dataset(name, subset, split=split)


def download_xcopa() -> None:
    spec = DATASETS["xcopa"]
    # English via SuperGLUE/COPA
    for split in spec["en"]["splits"]:
        print(f"[xcopa] en / {split}")
        ds = _hf_load(spec["en"]["hf_id"], spec["en"]["subset"], split)
        out = spec["local_dir"] / f"en_{split}.jsonl"
        rows = [dict(r) for r in ds]
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} -> {out}")
    # Chinese via XCOPA
    for split in spec["zh"]["splits"]:
        print(f"[xcopa] zh / {split}")
        ds = _hf_load(spec["zh"]["hf_id"], spec["zh"]["subset"], split)
        out = spec["local_dir"] / f"zh_{split}.jsonl"
        rows = [dict(r) for r in ds]
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} -> {out}")


def download_xstorycloze() -> None:
    spec = DATASETS["xstorycloze"]
    for lang, subset in spec["subsets"].items():
        for split in spec["splits"]:
            print(f"[xstorycloze] {lang} / {split} (subset={subset})")
            ds = _hf_load(spec["hf_id"], subset, split)
            out = spec["local_dir"] / f"{lang}_{split}.jsonl"
            n = _write_jsonl(out, (dict(r) for r in ds))
            print(f"  wrote {n} -> {out}")


def download_mgsm() -> None:
    spec = DATASETS["mgsm"]
    for lang in spec["subsets"].keys():
        url = spec["url_tpl"].format(lang=lang)
        print(f"[mgsm] {lang}: {url}")
        resp = urllib.request.urlopen(url, timeout=60)
        text = resp.read().decode("utf-8")
        rows = []
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            if len(parts) == 1:
                rows.append({"question": parts[0], "answer_number": ""})
            else:
                question, gold = parts[0], parts[-1]
                rows.append({"question": question, "answer_number": gold})
        out = spec["local_dir"] / f"{lang}_test.jsonl"
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} -> {out}")


def download_belebele() -> None:
    spec = DATASETS["belebele"]
    for lang, subset in spec["subsets"].items():
        for split in spec["splits"]:
            print(f"[belebele] {lang} / {split} (subset={subset})")
            ds = _hf_load(spec["hf_id"], subset, split)
            out = spec["local_dir"] / f"{lang}_{split}.jsonl"
            n = _write_jsonl(out, (dict(r) for r in ds))
            print(f"  wrote {n} -> {out}")


DOWNLOADERS = {
    "xcopa": download_xcopa,
    "xstorycloze": download_xstorycloze,
    "mgsm": download_mgsm,
    "belebele": download_belebele,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DOWNLOADERS.keys()),
                    choices=list(DOWNLOADERS.keys()))
    args = ap.parse_args()
    for name in args.datasets:
        try:
            DOWNLOADERS[name]()
        except Exception as e:
            print(f"[error] {name}: {e}")


if __name__ == "__main__":
    main()
