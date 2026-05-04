"""Day 1 afternoon: parse trajectories, extract simple features, and score correctness.

For each trajectory JSONL under output/trajectories/, emit a sibling file in
output/features/ with parsed steps and per-item features + correctness.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, TRAJ_DIR  # noqa: E402
from utils.features import extract_features  # noqa: E402
from utils.io import read_jsonl, write_jsonl  # noqa: E402
from utils.trajectory_parser import (  # noqa: E402
    extract_answer, extract_numeric_answer, normalize_numeric_string, parse_steps,
)


def _score(row: dict) -> int:
    dataset = row["dataset"]
    out = row["output"]
    gold = str(row["gold"])
    if dataset == "mgsm":
        pred = extract_numeric_answer(out)
        gold_clean = normalize_numeric_string(gold)
        try:
            return int(abs(float(pred) - float(gold_clean)) < 1e-6)
        except (TypeError, ValueError):
            return 0
    if dataset == "belebele":
        return int(extract_answer(out, ["A", "B", "C", "D"]).upper() == gold.upper())
    return int(extract_answer(out, ["A", "B"]).upper() == gold.upper())


def process(traj_path: Path) -> Path:
    rows = read_jsonl(traj_path)
    out_rows = []
    for r in rows:
        parsed = parse_steps(r["output"])
        feats = extract_features(parsed, r["lang"])
        correct = _score(r)
        out_rows.append({
            "id": r["id"], "model": r["model"], "dataset": r["dataset"],
            "lang": r["lang"], "mode": r["mode"],
            "steps": parsed["steps"], "n_steps": parsed["n_steps"],
            "n_output_tokens": r.get("n_output_tokens"),
            "features": feats, "correct": correct, "gold": r["gold"],
        })
    out_path = FEAT_DIR / traj_path.name
    write_jsonl(out_path, out_rows)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()
    for p in sorted(TRAJ_DIR.glob(args.pattern)):
        # prompt-flip files have a different schema (content_lang / instr_lang
        # instead of lang) and are processed by pilot_07b with per-row output-
        # language detection.
        if "__flip_" in p.name:
            print(f"[skip] {p.name}: prompt-flip file (handled by pilot_07b)")
            continue
        out = process(p)
        print(f"[feat] {p.name} -> {out}")


if __name__ == "__main__":
    main()
