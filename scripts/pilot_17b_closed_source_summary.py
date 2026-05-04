"""Summarize closed-source XCOPA spot checks into an appendix table."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import OUTPUT_DIR, RANK_DIR, REPORT_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.ranking import kendall_tau, rank_models  # noqa: E402

MAIN_FEAT_DIR = OUTPUT_DIR / "features"
CLOSED_FEAT_DIR = OUTPUT_DIR / "closed_source" / "features"


def _accuracy_from_dirs(dataset: str, mode: str, dirs: list[Path],
                        include_models: set[str] | None = None,
                        exclude_models: set[str] | None = None) -> dict:
    groups = defaultdict(list)
    include_models = include_models or set()
    exclude_models = exclude_models or set()
    for root in dirs:
        for p in root.glob(f"*__{dataset}__*__{mode}.jsonl"):
            parts = p.stem.split("__")
            if len(parts) != 4:
                continue
            model, _, lang, _ = parts
            if include_models and model not in include_models:
                continue
            if model in exclude_models:
                continue
            for r in read_jsonl(p):
                groups[(model, lang)].append(r)
    acc = defaultdict(dict)
    for (model, lang), rows in groups.items():
        acc[model][lang] = mean(r["correct"] for r in rows) if rows else 0.0
    return dict(acc)


def _stability(accuracy: dict, lang_pair=("en", "zh")) -> dict:
    models = sorted(m for m, langs in accuracy.items() if all(l in langs for l in lang_pair))
    if len(models) < 2:
        return {"error": f"need at least two complete models, found {len(models)}"}
    ranks = {}
    for lang in lang_pair:
        ranks[lang] = rank_models({m: accuracy[m][lang] for m in models}, higher_is_better=True)
    return {
        "models": models,
        "n_models": len(models),
        "kendall_tau_en_zh": kendall_tau(
            [ranks[lang_pair[0]][m] for m in models],
            [ranks[lang_pair[1]][m] for m in models],
        ),
    }


def _complete_closed_models(dataset: str) -> list[str]:
    models = set()
    for p in CLOSED_FEAT_DIR.glob(f"*__{dataset}__en__cot.jsonl"):
        model = p.stem.split("__")[0]
        needed = [
            CLOSED_FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl",
            CLOSED_FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl",
            CLOSED_FEAT_DIR / f"{model}__{dataset}__en__nocot.jsonl",
            CLOSED_FEAT_DIR / f"{model}__{dataset}__zh__nocot.jsonl",
        ]
        if all(path.exists() for path in needed):
            models.add(model)
    return sorted(models)


def summarize(dataset: str) -> dict:
    closed_models = _complete_closed_models(dataset)
    open_acc = {
        mode: _accuracy_from_dirs(dataset, mode, [MAIN_FEAT_DIR], exclude_models=set(closed_models))
        for mode in ("cot", "nocot")
    }
    open_stability = {mode: _stability(open_acc[mode]) for mode in ("cot", "nocot")}
    if all("kendall_tau_en_zh" in open_stability[m] for m in ("cot", "nocot")):
        open_stability["delta_tau_cot_minus_nocot"] = (
            open_stability["cot"]["kendall_tau_en_zh"]
            - open_stability["nocot"]["kendall_tau_en_zh"]
        )

    rows = []
    for model in closed_models:
        closed_acc = {
            mode: _accuracy_from_dirs(dataset, mode, [CLOSED_FEAT_DIR], include_models={model}).get(model, {})
            for mode in ("cot", "nocot")
        }
        with_closed = {}
        for mode in ("cot", "nocot"):
            acc = _accuracy_from_dirs(
                dataset, mode, [MAIN_FEAT_DIR, CLOSED_FEAT_DIR], include_models=set()
            )
            # Keep all open-weight models plus this one closed-source model.
            for other in closed_models:
                if other != model:
                    acc.pop(other, None)
            with_closed[mode] = _stability(acc)
        delta = None
        if all("kendall_tau_en_zh" in with_closed[m] for m in ("cot", "nocot")):
            delta = with_closed["cot"]["kendall_tau_en_zh"] - with_closed["nocot"]["kendall_tau_en_zh"]
        rows.append({
            "model": model,
            "cot_accuracy_en": closed_acc["cot"].get("en"),
            "cot_accuracy_zh": closed_acc["cot"].get("zh"),
            "nocot_accuracy_en": closed_acc["nocot"].get("en"),
            "nocot_accuracy_zh": closed_acc["nocot"].get("zh"),
            "cot_tau_with_open_models": with_closed["cot"].get("kendall_tau_en_zh"),
            "nocot_tau_with_open_models": with_closed["nocot"].get("kendall_tau_en_zh"),
            "delta_tau_cot_minus_nocot": delta,
            "cot_less_stable_than_nocot": (delta < 0) if delta is not None else None,
        })

    return {
        "dataset": dataset,
        "closed_models": closed_models,
        "open_weight_baseline": open_stability,
        "closed_source_rows": rows,
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Closed-Source XCOPA Spot Check",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Closed-source models: {len(result['closed_models'])}",
        "",
        "| Model | CoT EN | CoT ZH | no-CoT EN | no-CoT ZH | CoT tau | no-CoT tau | Delta tau | CoT less stable? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["closed_source_rows"]:
        def fmt(x):
            return "" if x is None else f"{x:.3f}"
        lines.append(
            f"| {row['model']} | {fmt(row['cot_accuracy_en'])} | {fmt(row['cot_accuracy_zh'])} | "
            f"{fmt(row['nocot_accuracy_en'])} | {fmt(row['nocot_accuracy_zh'])} | "
            f"{fmt(row['cot_tau_with_open_models'])} | {fmt(row['nocot_tau_with_open_models'])} | "
            f"{fmt(row['delta_tau_cot_minus_nocot'])} | {row['cot_less_stable_than_nocot']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--json_out", default=str(RANK_DIR / "pilot_day3_closed_source_spotcheck_summary.json"))
    ap.add_argument("--md_out", default=str(REPORT_DIR / "pilot_day3_closed_source_spotcheck_summary.md"))
    args = ap.parse_args()

    result = summarize(args.dataset)
    write_json(Path(args.json_out), result)
    write_markdown(result, Path(args.md_out))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
