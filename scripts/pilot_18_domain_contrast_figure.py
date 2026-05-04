"""Figure 3: domain contrast bar chart — XCOPA vs MGSM cross-lingual
divergence per process feature, with the XCOPA/MGSM ratio annotated above
each pair.

Output: output/figures/domain_contrast.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FIG_DIR, RANK_DIR  # noqa: E402

FEATURE_NAMES = ["step_count", "procedural_rate", "epistemic_rate", "dependency_depth"]
FEATURE_LABELS = {
    "step_count": "step count",
    "procedural_rate": "procedural rate",
    "epistemic_rate": "epistemic rate",
    "dependency_depth": "dependency depth",
}


def render(domain_contrast: dict, out_pdf: Path, out_png: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

    feats = FEATURE_NAMES
    xcopa_vals = [domain_contrast[f]["xcopa"] for f in feats]
    mgsm_vals = [domain_contrast[f]["mgsm"] for f in feats]
    ratios = [domain_contrast[f]["ratio_xcopa_over_mgsm"] for f in feats]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(feats)))
    w = 0.38
    b1 = ax.bar([i - w/2 for i in x], xcopa_vals, width=w,
                label="XCOPA (commonsense)", color="#d35454",
                edgecolor="#333", linewidth=0.6)
    b2 = ax.bar([i + w/2 for i in x], mgsm_vals, width=w,
                label="MGSM (math)", color="#4a90e2",
                edgecolor="#333", linewidth=0.6)

    # Annotate XCOPA / MGSM ratio above each pair
    for i, ratio in enumerate(ratios):
        if ratio is None:
            continue
        top = max(xcopa_vals[i], mgsm_vals[i])
        if ratio >= 1:
            color = "#c33"
            text = f"{ratio:.2f}×"
        else:
            color = "#36a"
            text = f"{ratio:.2f}× (M>C)"
        ax.annotate(text, xy=(i, top), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in feats], fontsize=10)
    ax.set_ylabel("cross-lingual mean absolute divergence (EN−ZH)", fontsize=10)
    ax.set_title("Domain contrast: XCOPA vs MGSM cross-lingual divergence by feature\n"
                 "(epistemic-rate ratio 37.53× is the main C2 effect)",
                 fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ymax = max(max(xcopa_vals), max(mgsm_vals)) * 1.25
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings_json",
                    default=str(RANK_DIR / "pilot_day1_rankings.json"))
    ap.add_argument("--out_pdf", default=str(FIG_DIR / "domain_contrast.pdf"))
    ap.add_argument("--out_png", default=str(FIG_DIR / "domain_contrast.png"))
    args = ap.parse_args()

    d = json.load(open(args.rankings_json))
    if "domain_contrast" not in d:
        sys.exit("no domain_contrast field in rankings_json")
    render(d["domain_contrast"], Path(args.out_pdf), Path(args.out_png))
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
