"""Day 1 Key test 4: tokenization control.

Correlate per-item token-count differences with step-count differences across
languages on the accuracy-matched subset. If |Δtokens| strongly predicts
|Δsteps|, then step-count divergence is partially confounded by tokenization
granularity and the signal must be discounted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _spearman(xs, ys):
    def _rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        pos = 0
        while pos < len(order):
            end = pos + 1
            while end < len(order) and vs[order[end]] == vs[order[pos]]:
                end += 1
            rank = (pos + 1 + end) / 2.0
            for i in order[pos:end]:
                ranks[i] = rank
            pos = end
        return ranks
    return _pearson(_rank(xs), _rank(ys))


def correlate(dataset: str, model: str):
    en = read_jsonl(FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl")
    zh = read_jsonl(FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl")
    en_by_id = {r["id"]: r for r in en}
    zh_by_id = {r["id"]: r for r in zh}
    matched = sorted(accuracy_matched_ids(
        {i: bool(r["correct"]) for i, r in en_by_id.items()},
        {i: bool(r["correct"]) for i, r in zh_by_id.items()},
    ))
    dtok, dstep = [], []
    missing_tok = 0
    for iid in matched:
        e = en_by_id[iid]
        z = zh_by_id[iid]
        if e.get("n_output_tokens") is None or z.get("n_output_tokens") is None:
            missing_tok += 1
            continue
        dtok.append(int(e["n_output_tokens"]) - int(z["n_output_tokens"]))
        dstep.append(int(e["features"]["step_count"]) - int(z["features"]["step_count"]))
    return {
        "model": model,
        "dataset": dataset,
        "n_matched": len(matched),
        "n_used": len(dtok),
        "n_missing_token_count": missing_tok,
        "pearson_dtok_dstep": _pearson(dtok, dstep),
        "spearman_dtok_dstep": _spearman(dtok, dstep),
        "pearson_abs_dtok_abs_dstep": _pearson([abs(x) for x in dtok], [abs(x) for x in dstep]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--exclude_models", nargs="+", default=None,
                    help="model keys to drop before analysis (e.g. Phi-4-reasoning)")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day1_tokenization_control.json"))
    args = ap.parse_args()

    if args.models is None:
        models = sorted({
            p.stem.split("__")[0]
            for d in args.datasets
            for p in FEAT_DIR.glob(f"*__{d}__en__cot.jsonl")
        })
    else:
        models = args.models
    if args.exclude_models:
        drop = set(args.exclude_models)
        models = [m for m in models if m not in drop]

    result = {d: [correlate(d, m) for m in models
                  if (FEAT_DIR / f"{m}__{d}__en__cot.jsonl").exists()
                  and (FEAT_DIR / f"{m}__{d}__zh__cot.jsonl").exists()]
              for d in args.datasets}
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
