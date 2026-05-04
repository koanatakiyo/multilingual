"""Figure 4: per-dataset scatter of (EN-ZH accuracy gap, LaBSE unmatched
ratio) across 10 open-weight models. Three subplots — XCOPA, MGSM,
XStoryCloze. Pearson r annotated per panel; the cross-dataset sign reversal
on XStoryCloze is the C3 honest finding.

Output: output/figures/gap_unmatched_scatter.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FIG_DIR, RANK_DIR  # noqa: E402

DATASET_ORDER = ["xcopa", "xstorycloze", "mgsm"]
DATASET_LABELS = {
    "xcopa": "XCOPA (causal commonsense)",
    "xstorycloze": "XStoryCloze (narrative commonsense)",
    "mgsm": "MGSM (math)",
}


def short_name(m: str) -> str:
    table = {
        "llama3.1-8b": "Llama-3.1",
        "Ministral-8B-Instruct-2410": "Ministral-8B",
        "c4ai-command-r7b-12-2024": "Command-R",
        "Yi-1.5-9B-Chat": "Yi-1.5-9B",
        "qwen3-8b": "Qwen3-8B",
        "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
        "Phi-4": "Phi-4",
        "Gemma-3-4B-Instruct": "Gemma-3",
        "aya-expanse-8b": "Aya",
        "DeepSeek-V2-Lite-Chat": "DeepSeek-V2",
    }
    return table.get(m, m)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs))
    sy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(sx*sy) if sx and sy else 0.0


def render(alignment: dict, out_pdf: Path, out_png: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panels = [(ds, alignment[ds]) for ds in DATASET_ORDER if ds in alignment]
    # Highlight models where structural divergence > accuracy gap dimension —
    # i.e., they are flagged by unmatched but not by accuracy gap
    structural_outliers = {"c4ai-command-r7b-12-2024", "Yi-1.5-9B-Chat", "Phi-4"}

    for ax, (ds, d) in zip(axes, panels):
        models = sorted(set(d["accuracy_en_per_model"])
                        & set(d["accuracy_zh_per_model"])
                        & set(d["unmatched_mean_per_model"]))
        gaps = [d["accuracy_en_per_model"][m] - d["accuracy_zh_per_model"][m]
                for m in models]
        unm = [d["unmatched_mean_per_model"][m] for m in models]
        r = pearson(gaps, unm)

        for m, g, u in zip(models, gaps, unm):
            colour = "#d35454" if m in structural_outliers else "#4a90e2"
            ax.scatter(g, u, s=60, color=colour, edgecolor="#222",
                       linewidth=0.7, zorder=3)
            ax.annotate(short_name(m), xy=(g, u), xytext=(4, 4),
                        textcoords="offset points", fontsize=7.5,
                        color="#333")

        ax.set_xlabel("EN − ZH accuracy gap", fontsize=10)
        ax.set_ylabel("LaBSE unmatched-step ratio" if ax is axes[0] else "",
                      fontsize=10)
        ax.set_title(f"{DATASET_LABELS[ds]}\nPearson r = {r:+.3f}  (n = {len(models)})",
                     fontsize=10.5, pad=8)
        ax.axhline(0, color="#888", linewidth=0.4, linestyle=":")
        ax.axvline(0, color="#888", linewidth=0.4, linestyle=":")
        ax.grid(linestyle=":", alpha=0.3)

        # Add a brief reading note
        sign = "+" if r >= 0 else "−"
        if abs(r) < 0.20:
            note = f"weak {sign}corr"
        elif abs(r) < 0.5:
            note = f"moderate {sign}corr"
        else:
            note = f"strong {sign}corr"
        ax.text(0.02, 0.97, note, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff",
                          edgecolor="#888", alpha=0.85))

    fig.suptitle("Accuracy gap vs LaBSE structural divergence — partly independent dimensions, "
                 "domain-direction-dependent  (red ★ = structural outlier without accuracy gap)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment_json",
                    default=str(RANK_DIR / "pilot_day3_alignment.json"))
    ap.add_argument("--out_pdf", default=str(FIG_DIR / "gap_unmatched_scatter.pdf"))
    ap.add_argument("--out_png", default=str(FIG_DIR / "gap_unmatched_scatter.png"))
    args = ap.parse_args()

    alignment = json.load(open(args.alignment_json))
    render(alignment, Path(args.out_pdf), Path(args.out_png))
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
