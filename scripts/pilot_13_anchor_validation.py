"""Day 6: anchor validation — the full extraction pipeline on anchor items.

Anchor items are language-invariant (syllogisms, simple arithmetic, clear
causal chains). If the extraction pipeline is clean, anchor D_HG should be
substantially lower than main-data D_HG (target: ≥ 2× gap, chapter 4.3.6).

Pipeline executed here:
    generate CoT (EN & ZH) → parse → features
    → forward+backward ensemble hyperedge extraction
    → LaBSE-aligned D_HG between EN and ZH
    → compare against main-data D_HG from pilot_11's JSON, if available.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import DATA_DIR, PILOT, RANK_DIR, REPORT_DIR  # noqa: E402
from utils.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.models import generate  # noqa: E402
from utils.trajectory_parser import parse_steps  # noqa: E402

from pilot_10_hyperedge_extraction import (  # noqa: E402
    anthropic_backward_factory, anthropic_forward_factory,
    extract_hyperedges,
    openai_backward_factory, openai_forward_factory,
    stub_backward, stub_forward,
)
from pilot_11_hypergraph_divergence import _match, _step_alignment  # noqa: E402


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


def _edges(cands, tau):
    return [
        {"target": c.target, "premises": list(c.premises),
         "confidence": min(c.c_forward, c.c_backward), "retained": True}
        for c in cands
        if min(c.c_forward, c.c_backward) >= tau
    ]


def _anchor_trajectories(model: str, items: list, seed: int) -> List[dict]:
    rows = []
    for it in items:
        for lang, body in (("en", it["en"]), ("zh", it["zh"])):
            instruction = ("Think step by step. Number each reasoning step (Step 1:, Step 2:, ...). "
                           "Give the final answer on the last line."
                           if lang == "en"
                           else "请逐步思考。依次为每个推理步骤编号（步骤1：、步骤2：、...）。最后一行给出最终答案。")
            prompt = f"{body}\n\n{instruction}"
            output = generate(model, [prompt], temperature=0.0, max_new_tokens=256, seed=seed)[0]
            rows.append({"id": it["id"], "lang": lang, "output": output})
    return rows


def run_one(model: str, items: list, forward_judge, backward_judge,
            k: int, tau: float, seed: int):
    trajs = _anchor_trajectories(model, items, seed)
    by_id = {}
    for r in trajs:
        steps = parse_steps(r["output"])["steps"]
        by_id.setdefault(r["id"], {})[r["lang"]] = steps

    per_item = []
    for iid, pair in by_id.items():
        steps_en = pair.get("en", [])
        steps_zh = pair.get("zh", [])
        if len(steps_en) < 2 or len(steps_zh) < 2:
            per_item.append({"id": iid, "n_en": len(steps_en), "n_zh": len(steps_zh),
                              "D_HG": None, "skipped": "fewer than 2 steps"})
            continue
        cands_en = extract_hyperedges(steps_en, forward_judge, backward_judge, k)
        cands_zh = extract_hyperedges(steps_zh, forward_judge, backward_judge, k)
        en_to_zh, cos = _step_alignment(steps_en, steps_zh)
        o = _match(_edges(cands_en, tau), _edges(cands_zh, tau), en_to_zh, cos)
        per_item.append({"id": iid, "n_en": len(steps_en), "n_zh": len(steps_zh),
                         "D_HG": 1 - o})
    return per_item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-8b", "llama3.1-8b"])
    ap.add_argument("--judge", choices=["stub", "openai", "anthropic"], default="openai")
    ap.add_argument("--judge_model", default="gpt-5.4-mini")
    ap.add_argument("--k", type=int, default=PILOT["ensemble_k"])
    ap.add_argument("--tau", type=float, default=PILOT["judge_tau"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--main_data_rankings",
                    default=str(RANK_DIR / "pilot_day6_hypergraph.json"),
                    help="pilot_11 output used for main-data D_HG comparison")
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day6_anchor.json"))
    args = ap.parse_args()

    if args.judge == "openai":
        forward_judge = openai_forward_factory(args.judge_model)
        backward_judge = openai_backward_factory(args.judge_model)
    elif args.judge == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set")
        forward_judge = anthropic_forward_factory(args.judge_model)
        backward_judge = anthropic_backward_factory(args.judge_model)
    else:
        if not args.allow_stub:
            raise SystemExit(
                "Refusing anchor validation with stub judge. "
                "Pass --allow-stub for pipeline tests, or --judge openai/anthropic."
            )
        forward_judge = stub_forward
        backward_judge = stub_backward

    items = ensure_anchor_items()
    main_path = Path(args.main_data_rankings)
    main_D_HG_per_model = {}
    if main_path.exists():
        rankings = read_json(main_path)
        main_D_HG_per_model = rankings.get("D_HG_per_model", {})

    summary = {"models": args.models, "n_items": len(items), "judge": args.judge,
               "per_model": {}}
    for model in args.models:
        per_item = run_one(model, items, forward_judge, backward_judge,
                            args.k, args.tau, args.seed)
        scored = [r["D_HG"] for r in per_item if r["D_HG"] is not None]
        anchor_mean = mean(scored) if scored else 0.0
        main_mean = main_D_HG_per_model.get(model)
        summary["per_model"][model] = {
            "anchor_D_HG_mean": anchor_mean,
            "main_D_HG_mean": main_mean,
            "anchor_over_main_ratio": (anchor_mean / main_mean) if main_mean else None,
            "per_item": per_item,
        }

    write_json(Path(args.out), summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
