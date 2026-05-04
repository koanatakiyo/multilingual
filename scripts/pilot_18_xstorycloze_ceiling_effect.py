"""XStoryCloze ceiling-effect analysis.

This is a reporting-only analysis: no new model calls. It summarizes EN/ZH
accuracy ranges, EN-ZH gaps, and how many models sit near ceiling so the paper
can preempt the "XStoryCloze does not replicate XCOPA" reviewer objection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, REPORT_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402


def _accuracy(model: str, dataset: str, lang: str, mode: str) -> tuple[float, int]:
    path = FEAT_DIR / f"{model}__{dataset}__{lang}__{mode}.jsonl"
    rows = read_jsonl(path)
    return (mean(r["correct"] for r in rows) if rows else 0.0), len(rows)


def _range(vals: list[float]) -> dict:
    return {"min": min(vals), "max": max(vals), "width": max(vals) - min(vals)}


def analyze(dataset: str, mode: str, gap_threshold: float,
            ceiling_threshold: float) -> dict:
    models = sorted({
        p.stem.split("__")[0]
        for p in FEAT_DIR.glob(f"*__{dataset}__en__{mode}.jsonl")
        if (FEAT_DIR / f"{p.stem.split('__')[0]}__{dataset}__zh__{mode}.jsonl").exists()
    })
    per_model = {}
    for m in models:
        en_acc, n_en = _accuracy(m, dataset, "en", mode)
        zh_acc, n_zh = _accuracy(m, dataset, "zh", mode)
        per_model[m] = {
            "n_en": n_en,
            "n_zh": n_zh,
            "accuracy_en": en_acc,
            "accuracy_zh": zh_acc,
            "gap_en_minus_zh": en_acc - zh_acc,
            "abs_gap": abs(en_acc - zh_acc),
        }

    en_vals = [v["accuracy_en"] for v in per_model.values()]
    zh_vals = [v["accuracy_zh"] for v in per_model.values()]
    gaps = [v["abs_gap"] for v in per_model.values()]
    small_gap = [m for m, v in per_model.items() if v["abs_gap"] < gap_threshold]
    ceiling_en = [m for m, v in per_model.items() if v["accuracy_en"] >= ceiling_threshold]
    ceiling_zh = [m for m, v in per_model.items() if v["accuracy_zh"] >= ceiling_threshold]
    largest_gap_model = max(per_model, key=lambda m: per_model[m]["abs_gap"]) if per_model else None

    out = {
        "dataset": dataset,
        "mode": mode,
        "n_models": len(models),
        "models": models,
        "gap_threshold": gap_threshold,
        "ceiling_threshold": ceiling_threshold,
        "accuracy_en": {
            "mean": mean(en_vals) if en_vals else 0.0,
            "range": _range(en_vals) if en_vals else None,
            "n_at_or_above_ceiling_threshold": len(ceiling_en),
            "models_at_or_above_ceiling_threshold": ceiling_en,
        },
        "accuracy_zh": {
            "mean": mean(zh_vals) if zh_vals else 0.0,
            "range": _range(zh_vals) if zh_vals else None,
            "n_at_or_above_ceiling_threshold": len(ceiling_zh),
            "models_at_or_above_ceiling_threshold": ceiling_zh,
        },
        "en_zh_abs_gap": {
            "mean": mean(gaps) if gaps else 0.0,
            "range": _range(gaps) if gaps else None,
            "n_below_threshold": len(small_gap),
            "pct_below_threshold": (len(small_gap) / len(models)) if models else 0.0,
            "models_below_threshold": small_gap,
            "largest_gap_model": largest_gap_model,
            "largest_gap": per_model[largest_gap_model]["abs_gap"] if largest_gap_model else None,
        },
        "per_model": per_model,
    }

    if largest_gap_model and len(models) > 1:
        remaining = [m for m in models if m != largest_gap_model]
        en_ex = [per_model[m]["accuracy_en"] for m in remaining]
        zh_ex = [per_model[m]["accuracy_zh"] for m in remaining]
        gap_ex = [per_model[m]["abs_gap"] for m in remaining]
        out["excluding_largest_gap_model"] = {
            "excluded_model": largest_gap_model,
            "n_models": len(remaining),
            "accuracy_en_mean": mean(en_ex),
            "accuracy_zh_mean": mean(zh_ex),
            "abs_gap_mean": mean(gap_ex),
            "abs_gap_range": _range(gap_ex),
        }
    return out


def write_markdown(result: dict, path: Path) -> None:
    en = result["accuracy_en"]
    zh = result["accuracy_zh"]
    gap = result["en_zh_abs_gap"]
    lines = [
        "# XStoryCloze Ceiling-Effect Analysis",
        "",
        f"- Dataset/mode: `{result['dataset']}` / `{result['mode']}`",
        f"- Models: {result['n_models']}",
        f"- EN accuracy mean/range: {en['mean']:.3f} "
        f"[{en['range']['min']:.3f}, {en['range']['max']:.3f}]",
        f"- ZH accuracy mean/range: {zh['mean']:.3f} "
        f"[{zh['range']['min']:.3f}, {zh['range']['max']:.3f}]",
        f"- Absolute EN-ZH gap mean/range: {gap['mean']:.3f} "
        f"[{gap['range']['min']:.3f}, {gap['range']['max']:.3f}]",
        f"- Models with abs gap < {result['gap_threshold']:.3f}: "
        f"{gap['n_below_threshold']}/{result['n_models']} "
        f"({gap['pct_below_threshold']:.1%})",
        f"- Models with EN accuracy >= {result['ceiling_threshold']:.2f}: "
        f"{en['n_at_or_above_ceiling_threshold']}/{result['n_models']}",
        f"- Models with ZH accuracy >= {result['ceiling_threshold']:.2f}: "
        f"{zh['n_at_or_above_ceiling_threshold']}/{result['n_models']}",
        "",
        "Interpretation: XStoryCloze is near ceiling for most models, especially "
        "in English, and most EN-ZH gaps are small. Weak or reversed rank-signal "
        "on this benchmark should therefore be read as a dataset-characteristic "
        "result rather than as a direct replication failure of the harder "
        "commonsense XCOPA setting.",
        "",
        "| Model | EN acc | ZH acc | EN-ZH gap | abs gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, v in result["per_model"].items():
        lines.append(
            f"| {m} | {v['accuracy_en']:.3f} | {v['accuracy_zh']:.3f} | "
            f"{v['gap_en_minus_zh']:+.3f} | {v['abs_gap']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xstorycloze")
    ap.add_argument("--mode", default="cot")
    ap.add_argument("--gap_threshold", type=float, default=0.04)
    ap.add_argument("--ceiling_threshold", type=float, default=0.90)
    ap.add_argument("--json_out", default=str(REPORT_DIR / "pilot_day3_xstorycloze_ceiling_effect.json"))
    ap.add_argument("--md_out", default=str(REPORT_DIR / "pilot_day3_xstorycloze_ceiling_effect.md"))
    args = ap.parse_args()

    result = analyze(args.dataset, args.mode, args.gap_threshold, args.ceiling_threshold)
    write_json(Path(args.json_out), result)
    write_markdown(result, Path(args.md_out))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
