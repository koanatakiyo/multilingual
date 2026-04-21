"""Day 6 afternoon: layer-wise CKA between EN and ZH hidden states.

For one CKA model (default qwen3-8b), extract per-layer mean-pooled
representations on ~50 matched items and compute linear CKA per layer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import CKA_DIR, FEAT_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.models import hidden_states  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--n_items", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    en_feat = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__en__cot.jsonl")
    zh_feat = read_jsonl(FEAT_DIR / f"{args.model}__{args.dataset}__zh__cot.jsonl")
    matched = sorted(accuracy_matched_ids(
        {r["id"]: bool(r["correct"]) for r in en_feat},
        {r["id"]: bool(r["correct"]) for r in zh_feat},
    ))[: args.n_items]
    if not matched:
        raise SystemExit("no matched items")

    items_en = {it["id"]: it for it in load_dataset(args.dataset, "en", limit=None)}
    items_zh = {it["id"]: it for it in load_dataset(args.dataset, "zh", limit=None)}

    layers_en: Dict[int, List[np.ndarray]] = {}
    layers_zh: Dict[int, List[np.ndarray]] = {}

    for iid in matched:
        en_prompt = build_prompt(args.dataset, items_en[iid]["prompt_payload"], "en", cot=False)
        zh_prompt = build_prompt(args.dataset, items_zh[iid]["prompt_payload"], "zh", cot=False)
        hs_en = hidden_states(args.model, en_prompt)
        hs_zh = hidden_states(args.model, zh_prompt)
        for li, (he, hz) in enumerate(zip(hs_en, hs_zh)):
            layers_en.setdefault(li, []).append(he)
            layers_zh.setdefault(li, []).append(hz)

    cka_per_layer = {}
    for li in sorted(layers_en):
        X = np.stack(layers_en[li])
        Y = np.stack(layers_zh[li])
        cka_per_layer[li] = linear_cka(X, Y)

    out_path = Path(args.out) if args.out else (CKA_DIR / f"cka__{args.model}__{args.dataset}.json")
    write_json(out_path, {"model": args.model, "dataset": args.dataset,
                          "n_items": len(matched), "cka_per_layer": cka_per_layer})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
