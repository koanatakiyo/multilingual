"""Llama one-look figure (chapter 4 §4.2.3): the paper's primary visual.

Selects one XCOPA item where Llama 3.1 answered correctly in BOTH EN and ZH
yet the LaBSE step alignment shows large structural divergence. Renders:

  - top:    paired EN / ZH CoT trajectories side by side, with LaBSE
            alignment lines connecting matched steps; unmatched steps
            highlighted on each side.
  - bottom: per-feature bar chart EN vs ZH on this single item (step count,
            procedural rate, epistemic rate, dependency depth).

The narrative caption: "Both EN and ZH receive correct answers, but the
process trajectory and per-feature rankings diverge — accuracy-only
evaluation cannot detect this."

Requires pilot_08 to have run (`alignment__llama3.1-8b__xcopa.jsonl`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, FIG_DIR, RANK_DIR  # noqa: E402
from utils.data_loader import load_xcopa  # noqa: E402
from utils.io import read_jsonl  # noqa: E402

MODEL = "llama3.1-8b"
DATASET = "xcopa"
FEATURE_NAMES = ["step_count", "procedural_rate", "epistemic_rate", "dependency_depth"]


def select_item(alignment_records, en_feat_by_id, zh_feat_by_id,
                min_unmatched=0.20, prefer_balanced_steps=True):
    """Pick an XCOPA item with: both EN+ZH correct, high unmatched ratio,
    sensible step counts (≥3 each side, ≤8 each side for figure readability)."""
    candidates = []
    for r in alignment_records:
        iid = r["id"]
        if iid not in en_feat_by_id or iid not in zh_feat_by_id:
            continue
        e, z = en_feat_by_id[iid], zh_feat_by_id[iid]
        if not (e["correct"] and z["correct"]):
            continue
        if r["unmatched_ratio"] < min_unmatched:
            continue
        n_e, n_z = e["n_steps"], z["n_steps"]
        if not (3 <= n_e <= 8 and 3 <= n_z <= 8):
            continue
        candidates.append({
            "id": iid, "unmatched": r["unmatched_ratio"],
            "mean_sim": r["mean_match_sim"], "n_en": n_e, "n_zh": n_z,
            "abs_step_diff": abs(n_e - n_z),
            "matches": r["matches"],
        })
    if not candidates:
        return None
    if prefer_balanced_steps:
        # secondary sort: prefer items where step counts are roughly comparable
        # so alignment lines are visually clean
        candidates.sort(key=lambda c: (-c["unmatched"], c["abs_step_diff"]))
    else:
        candidates.sort(key=lambda c: -c["unmatched"])
    return candidates[0]


def render_figure(out_pdf: Path, out_png: Path, item_id: str, item_data: dict,
                  en_steps, zh_steps, alignment_matches,
                  en_feat, zh_feat, problem_payload, gold):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import patches
    # Use a CJK-capable font so Chinese trajectory text renders correctly.
    matplotlib.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.35)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # === TOP: paired CoT with alignment lines ===
    ax_top.set_xlim(0, 10)
    n_max = max(len(en_steps), len(zh_steps))
    ax_top.set_ylim(-0.5, n_max + 0.5)
    ax_top.invert_yaxis()
    ax_top.axis("off")

    matched_a = {m[0] for m in alignment_matches}
    matched_b = {m[1] for m in alignment_matches}

    def _wrap(text, w=42):
        words = text.split()
        lines, cur = [], ""
        for w_ in words:
            if len(cur) + len(w_) + 1 > w:
                lines.append(cur); cur = w_
            else:
                cur = (cur + " " + w_).strip()
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    for i, s in enumerate(en_steps):
        bg = "#f0f0f0" if i in matched_a else "#fde0dc"
        ax_top.add_patch(patches.FancyBboxPatch(
            (0.2, i - 0.35), 4.0, 0.70, boxstyle="round,pad=0.04",
            linewidth=0.8, facecolor=bg, edgecolor="#888"))
        ax_top.text(0.32, i, _wrap(s, 50), va="center", ha="left", fontsize=8)

    for j, s in enumerate(zh_steps):
        bg = "#f0f0f0" if j in matched_b else "#fde0dc"
        ax_top.add_patch(patches.FancyBboxPatch(
            (5.8, j - 0.35), 4.0, 0.70, boxstyle="round,pad=0.04",
            linewidth=0.8, facecolor=bg, edgecolor="#888"))
        ax_top.text(5.92, j, _wrap(s, 50), va="center", ha="left", fontsize=8)

    for m in alignment_matches:
        ax_top.plot([4.2, 5.8], [m[0], m[1]],
                    color="#2c7fb8", linewidth=0.9, alpha=0.6)

    ax_top.text(2.2, -0.3, "EN trajectory", ha="center", fontsize=10,
                fontweight="bold", color="#333")
    ax_top.text(7.8, -0.3, "ZH trajectory", ha="center", fontsize=10,
                fontweight="bold", color="#333")
    ax_top.text(5.0, n_max + 0.0,
                f"unmatched ratio: {item_data['unmatched']:.3f}   "
                f"matched-step similarity: {item_data['mean_sim']:.3f}",
                ha="center", fontsize=9, color="#555")

    # === BOTTOM: per-feature EN vs ZH bar chart ===
    en_vals = [en_feat["features"][f] for f in FEATURE_NAMES]
    zh_vals = [zh_feat["features"][f] for f in FEATURE_NAMES]
    x = list(range(len(FEATURE_NAMES)))
    ax_bot.bar([i - 0.18 for i in x], en_vals, width=0.36, label="EN",
               color="#4a90e2")
    ax_bot.bar([i + 0.18 for i in x], zh_vals, width=0.36, label="ZH",
               color="#e94e77")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(FEATURE_NAMES, rotation=10, fontsize=9)
    ax_bot.set_ylabel("feature value", fontsize=9)
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
    ax_bot.set_title("Per-feature comparison on this item (EN vs ZH)",
                     fontsize=10, pad=6)

    # Title
    premise = problem_payload.get("premise", "")
    qtype = problem_payload.get("question", "")
    ca = problem_payload.get("choice1", "")
    cb = problem_payload.get("choice2", "")
    head = (f"Llama 3.1 8B on XCOPA item {item_id} — "
            f"both EN and ZH correct (gold {gold}); "
            f"process diverges.\n"
            f"Premise: {premise[:120]} (q: {qtype})  "
            f"A: {ca[:50]}  B: {cb[:50]}")
    fig.suptitle(head, fontsize=11, x=0.05, ha="left", y=0.98)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--min_unmatched", type=float, default=0.20,
                    help="lowest unmatched_ratio acceptable for a candidate")
    ap.add_argument("--item_id", default=None,
                    help="override candidate selection")
    ap.add_argument("--out_pdf", default=str(FIG_DIR / "llama_one_look.pdf"))
    ap.add_argument("--out_png", default=str(FIG_DIR / "llama_one_look.png"))
    args = ap.parse_args()

    align_path = RANK_DIR / f"alignment__{args.model}__{args.dataset}.jsonl"
    if not align_path.exists():
        sys.exit(f"missing {align_path} — run pilot_08 first")
    align_records = read_jsonl(align_path)
    en_rows = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__en__cot.jsonl")
    zh_rows = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__zh__cot.jsonl")
    en_by_id = {r["id"]: r for r in en_rows}
    zh_by_id = {r["id"]: r for r in zh_rows}

    if args.item_id:
        rec = next((r for r in align_records if r["id"] == args.item_id), None)
        if rec is None:
            sys.exit(f"item {args.item_id} not in alignment records")
        chosen = {
            "id": rec["id"], "unmatched": rec["unmatched_ratio"],
            "mean_sim": rec["mean_match_sim"],
            "n_en": rec["n_en"], "n_zh": rec["n_zh"],
            "abs_step_diff": abs(rec["n_en"] - rec["n_zh"]),
            "matches": rec["matches"],
        }
    else:
        chosen = select_item(align_records, en_by_id, zh_by_id,
                             min_unmatched=args.min_unmatched)
        if chosen is None:
            sys.exit("no qualifying item; lower --min_unmatched or pass --item_id")

    e, z = en_by_id[chosen["id"]], zh_by_id[chosen["id"]]
    print(f"selected: {chosen['id']}  unmatched={chosen['unmatched']:.3f}  "
          f"sim={chosen['mean_sim']:.3f}  n_en={chosen['n_en']}  n_zh={chosen['n_zh']}")

    # Need original XCOPA payload (premise / choices / gold)
    en_data = {it["id"]: it for it in load_xcopa("en", split="test")}
    if chosen["id"] not in en_data:
        sys.exit(f"item {chosen['id']} not found in xcopa test set")
    payload = en_data[chosen["id"]]["prompt_payload"]
    gold = en_data[chosen["id"]]["gold"]

    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    render_figure(Path(args.out_pdf), Path(args.out_png),
                  chosen["id"], chosen, e["steps"], z["steps"],
                  chosen["matches"], e, z, payload, gold)
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
