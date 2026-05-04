"""Day 2 afternoon: no-CoT ranking stability comparison.

Requires pilot_01 to have been run with --mode nocot for the target datasets.
Compares ranking stability across EN–ZH under no-CoT vs. CoT.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.ranking import kendall_tau, rank_models  # noqa: E402


def _accuracy(dataset: str, mode: str):
    groups = defaultdict(list)
    for p in FEAT_DIR.glob(f"*__{dataset}__*__{mode}.jsonl"):
        model, _, lang, mode_ = p.stem.split("__")
        for r in read_jsonl(p):
            groups[(model, lang)].append(r)
    acc = defaultdict(dict)
    for (m, lang), rows in groups.items():
        acc[m][lang] = mean(r["correct"] for r in rows) if rows else 0.0
    return dict(acc)


def stability(accuracy: dict, lang_pair=("en", "zh")) -> dict:
    models = sorted(
        m for m, per_lang in accuracy.items()
        if all(lang in per_lang for lang in lang_pair)
    )
    if not models:
        return {"error": "no complete EN/ZH accuracy pairs"}
    langs = set(lang_pair)
    results = {}
    for lang in langs:
        vals = {m: accuracy[m][lang] for m in models}
        results[lang] = {"values": vals, "ranks": rank_models(vals)}
    if "en" in results and "zh" in results:
        en = [results["en"]["ranks"][m] for m in models]
        zh = [results["zh"]["ranks"][m] for m in models]
        results["kendall_tau_en_zh"] = kendall_tau(en, zh)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day2_nocot_comparison.json"))
    args = ap.parse_args()

    cot_acc = _accuracy(args.dataset, "cot")
    nocot_acc = _accuracy(args.dataset, "nocot")

    result = {
        "dataset": args.dataset,
        "cot": stability(cot_acc),
        "nocot": stability(nocot_acc),
    }
    if "kendall_tau_en_zh" in result["cot"] and "kendall_tau_en_zh" in result["nocot"]:
        result["delta_tau_cot_minus_nocot"] = (
            result["cot"]["kendall_tau_en_zh"] - result["nocot"]["kendall_tau_en_zh"]
        )
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
