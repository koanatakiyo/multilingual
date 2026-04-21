"""Day 6: anchor validation.

Runs the full pipeline (generate → parse → features → hyperedge extraction →
D_HG) on a small set of anchor items (syllogisms, simple arithmetic, clear
causal chains). Expected: anchor D_HG << main-data D_HG.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import DATA_DIR, HG_DIR, REPORT_DIR  # noqa: E402
from utils.features import extract_features  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.models import generate  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402
from utils.trajectory_parser import parse_steps  # noqa: E402


ANCHOR_PATH = DATA_DIR / "anchor" / "anchor_items.jsonl"


def _default_anchor_items() -> list:
    return [
        {"id": f"anchor-{i}", "en": en, "zh": zh}
        for i, (en, zh) in enumerate([
            ("All humans are mortal. Socrates is a human. Therefore Socrates is ___.",
             "所有人都是会死的。苏格拉底是人。因此苏格拉底是___。"),
            ("If it rains, the ground gets wet. It is raining. Therefore the ground is ___.",
             "如果下雨，地面就会湿。现在在下雨。因此地面___。"),
            ("7 + 5 = ?", "7 + 5 = ?"),
            ("12 * 11 = ?", "12 × 11 = ?"),
            ("A is north of B. B is north of C. Where is A relative to C?",
             "甲在乙的北边。乙在丙的北边。甲相对于丙在哪个方向？"),
            ("If today is Monday, what day is it 10 days later?",
             "如果今天是星期一，10天后是星期几？"),
            ("John has 3 apples. He eats 1 and buys 4 more. How many does he have?",
             "约翰有3个苹果。他吃掉1个又买了4个，他现在有多少个？"),
            ("All primes greater than 2 are odd. 7 is a prime greater than 2. Is 7 odd?",
             "所有大于2的素数都是奇数。7是大于2的素数。7是奇数吗？"),
            ("A > B and B > C. What is the relationship between A and C?",
             "A > B 且 B > C。A 与 C 的关系是什么？"),
            ("If x + 3 = 10, what is x?", "若 x + 3 = 10，x 是多少？"),
        ])
    ]


def ensure_anchor_items() -> list:
    if ANCHOR_PATH.exists():
        return read_jsonl(ANCHOR_PATH)
    items = _default_anchor_items()
    write_jsonl(ANCHOR_PATH, items)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-8b", "llama3.1-8b"])
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day6_anchor.json"))
    args = ap.parse_args()

    anchor_items = ensure_anchor_items()
    summary = {"models": args.models, "n_items": len(anchor_items), "per_model": {}}

    for model in args.models:
        en_feats, zh_feats = [], []
        for it in anchor_items:
            for lang, body in (("en", it["en"]), ("zh", it["zh"])):
                prompt_payload = {"premise": body, "choice1": "", "choice2": "", "question": "cause"}
                prompt = f"{body}\n\n" + ("Think step by step and give the final answer." if lang == "en" else "请逐步思考并给出最终答案。")
                output = generate(model, [prompt], temperature=0.0, max_new_tokens=256)[0]
                parsed = parse_steps(output)
                feats = extract_features(parsed, lang)
                (en_feats if lang == "en" else zh_feats).append(feats["step_count"])
        summary["per_model"][model] = {
            "mean_step_count_en": mean(en_feats) if en_feats else 0.0,
            "mean_step_count_zh": mean(zh_feats) if zh_feats else 0.0,
            "abs_diff_step_count": abs((mean(en_feats) if en_feats else 0.0)
                                       - (mean(zh_feats) if zh_feats else 0.0)),
        }

    write_json(Path(args.out), summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
