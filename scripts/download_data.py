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
    """Download XCOPA EN+ZH and backfill EN test labels from ZH.

    super_glue/copa hides test labels (label = -1). XCOPA/zh is a professional
    translation of the same COPA items and does expose labels. Because the
    items share idx (verified equal key universes on test and validation),
    the correct choice is invariant across languages and we can backfill the
    EN gold from ZH.
    """
    spec = DATASETS["xcopa"]
    zh_by_idx_by_split: Dict[str, Dict[int, Dict]] = {}
    # Chinese via XCOPA first so we have labels available for backfill.
    for split in spec["zh"]["splits"]:
        print(f"[xcopa] zh / {split}")
        ds = _hf_load(spec["zh"]["hf_id"], spec["zh"]["subset"], split)
        rows = [dict(r) for r in ds]
        out = spec["local_dir"] / f"zh_{split}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} -> {out}")
        zh_by_idx_by_split[split] = {int(r["idx"]): r for r in rows}

    for split in spec["en"]["splits"]:
        print(f"[xcopa] en / {split}")
        ds = _hf_load(spec["en"]["hf_id"], spec["en"]["subset"], split)
        rows = [dict(r) for r in ds]
        backfilled = 0
        for r in rows:
            if int(r.get("label", -1)) < 0 and split in zh_by_idx_by_split:
                zh_r = zh_by_idx_by_split[split].get(int(r["idx"]))
                if zh_r is not None:
                    r["label"] = int(zh_r["label"])
                    r["label_source"] = "xcopa_zh_backfill"
                    backfilled += 1
        out = spec["local_dir"] / f"en_{split}.jsonl"
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} (backfilled {backfilled}) -> {out}")


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
        numeric_gold = 0
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            if len(parts) == 1:
                rows.append({"question": parts[0], "answer_number": ""})
                continue
            question, gold = parts[0], parts[-1]
            try:
                float(gold.replace(",", ""))
                numeric_gold += 1
            except ValueError:
                pass
            rows.append({"question": question, "answer_number": gold})
        out = spec["local_dir"] / f"{lang}_test.jsonl"
        n = _write_jsonl(out, rows)
        print(f"  wrote {n} (numeric gold: {numeric_gold}/{n}) -> {out}")
        if n and numeric_gold / n < 0.9:
            print(f"  [warn] only {numeric_gold}/{n} rows had numeric gold — "
                  "MGSM TSV format may have changed; inspect the file.")


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
