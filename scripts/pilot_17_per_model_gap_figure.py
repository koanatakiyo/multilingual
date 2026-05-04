"""Figure 1: per-model EN-ZH accuracy gap under CoT vs no-CoT (XCOPA).

Shows the 10 open-weight models + 3 closed-source spot-check models. Each
model gets two bars (CoT gap, no-CoT gap). The 3 CoT-fragile models
(|gap shift| >= 0.10) are starred. Closed-source models are visually
distinguished by hatching.

Output: output/figures/per_model_gap.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FIG_DIR, RANK_DIR  # noqa: E402

FRAGILITY_THRESHOLD = 0.10


def load_open_weight_gaps(nocot_path: Path):
    d = json.load(open(nocot_path))
    cot_en = d["cot"]["en"]["values"]
    cot_zh = d["cot"]["zh"]["values"]
    nc_en = d["nocot"]["en"]["values"]
    nc_zh = d["nocot"]["zh"]["values"]
    rows = []
    for m in sorted(cot_en):
        rows.append({
            "model": m, "kind": "open",
            "cot_gap": cot_en[m] - cot_zh[m],
            "nocot_gap": nc_en[m] - nc_zh[m],
        })
    return rows


def load_closed_source_gaps(rank_dir: Path):
    rows = []
    for p in sorted(rank_dir.glob("pilot_day3_closed_source_spotcheck_*.json")):
        if "summary" in p.name:
            continue
        d = json.load(open(p))
        acc = d["closed_model_accuracy"]
        m = d["closed_model_key"]
        rows.append({
            "model": m, "kind": "closed",
            "cot_gap": acc["cot"]["en"] - acc["cot"]["zh"],
            "nocot_gap": acc["nocot"]["en"] - acc["nocot"]["zh"],
        })
    return rows


def short_name(m: str) -> str:
    table = {
        "llama3.1-8b": "Llama-3.1-8B",
        "Ministral-8B-Instruct-2410": "Ministral-8B",
        "c4ai-command-r7b-12-2024": "Command-R-7B",
        "Yi-1.5-9B-Chat": "Yi-1.5-9B",
        "qwen3-8b": "Qwen3-8B",
        "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
        "Phi-4": "Phi-4",
        "Gemma-3-4B-Instruct": "Gemma-3-4B",
        "aya-expanse-8b": "Aya-Expanse-8B",
        "DeepSeek-V2-Lite-Chat": "DeepSeek-V2-Lite",
        "anthropic-claude-sonnet-4-6": "Claude-Sonnet-4.6",
        "openai-gpt-4o-mini": "GPT-4o-mini",
        "grok-grok-4.3": "Grok-4.3",
    }
    return table.get(m, m)


def render(rows, out_pdf: Path, out_png: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

    rows = sorted(rows, key=lambda r: -r["cot_gap"])
    n = len(rows)
    x = list(range(n))
    cot = [r["cot_gap"] for r in rows]
    nocot = [r["nocot_gap"] for r in rows]
    fragile = [abs(r["cot_gap"] - r["nocot_gap"]) >= FRAGILITY_THRESHOLD for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.38

    cot_color = "#d35454"     # red-ish for CoT
    nocot_color = "#4a90e2"   # blue for no-CoT
    cot_bars = ax.bar([i - w/2 for i in x], cot, width=w, label="CoT gap (EN − ZH)",
                      color=cot_color, edgecolor="#333", linewidth=0.6)
    nocot_bars = ax.bar([i + w/2 for i in x], nocot, width=w,
                        label="no-CoT gap (EN − ZH)",
                        color=nocot_color, edgecolor="#333", linewidth=0.6)

    # Hatch for closed-source models
    for r, b1, b2 in zip(rows, cot_bars, nocot_bars):
        if r["kind"] == "closed":
            b1.set_hatch("///"); b2.set_hatch("///")

    # Star fragile models
    for i, (r, frag) in enumerate(zip(rows, fragile)):
        if frag:
            top = max(cot[i], nocot[i], 0) + 0.025
            ax.annotate("★", xy=(i, top), ha="center", fontsize=15,
                        color="#222", fontweight="bold")

    # Threshold band
    ax.axhspan(-FRAGILITY_THRESHOLD, FRAGILITY_THRESHOLD, color="#ddd", alpha=0.35,
               zorder=0, label=f"±{FRAGILITY_THRESHOLD} fragility band")
    ax.axhline(0, color="#333", linewidth=0.7, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(r["model"]) for r in rows], rotation=35,
                       ha="right", fontsize=9)
    ax.set_ylabel("EN − ZH accuracy gap on XCOPA", fontsize=10)
    ax.set_title("CoT-induced cross-lingual fragility per model "
                 "(★ = |Δgap| ≥ 0.10; hatched = closed-source spot check)",
                 fontsize=11, pad=10)
    ax.set_ylim(min(min(cot), min(nocot)) - 0.05, max(cot + nocot) + 0.10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocot_json",
                    default=str(RANK_DIR / "pilot_day2_nocot_comparison.json"))
    ap.add_argument("--include_closed_source", action="store_true", default=True)
    ap.add_argument("--out_pdf", default=str(FIG_DIR / "per_model_gap.pdf"))
    ap.add_argument("--out_png", default=str(FIG_DIR / "per_model_gap.png"))
    args = ap.parse_args()

    rows = load_open_weight_gaps(Path(args.nocot_json))
    if args.include_closed_source:
        rows.extend(load_closed_source_gaps(Path(RANK_DIR)))
    print(f"loaded {len(rows)} models "
          f"({sum(1 for r in rows if r['kind']=='closed')} closed-source)")
    fragile = [r for r in rows if abs(r["cot_gap"] - r["nocot_gap"]) >= FRAGILITY_THRESHOLD]
    print(f"fragile (|Δgap| >= {FRAGILITY_THRESHOLD}): "
          f"{', '.join(short_name(r['model']) for r in fragile)}")
    render(rows, Path(args.out_pdf), Path(args.out_png))
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
