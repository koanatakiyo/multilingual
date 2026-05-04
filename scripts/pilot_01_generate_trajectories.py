"""Day 1 morning: generate CoT (and optionally no-CoT) trajectories.

Writes one JSONL per (model, dataset, lang, mode) under output/trajectories/.
Row schema:
    {id, model, dataset, lang, mode, prompt, output, gold, n_output_tokens}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import MODELS, PILOT, TRAJ_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.io import write_jsonl  # noqa: E402
from utils.models import generate, get_tokenizer  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402


def _task_spec(dataset: str, mode: str):
    if mode == "cot":
        return {
            "xcopa": PILOT["xcopa_items"],
            "xstorycloze": None,
            "mgsm": PILOT["mgsm_items"],
            "belebele": PILOT["belebele_items"],
        }[dataset], PILOT["max_new_tokens"]
    return {
        "xcopa": PILOT["xcopa_items"],
        "mgsm": PILOT["mgsm_items"],
    }.get(dataset, PILOT["xcopa_items"]), PILOT["nocot_max_new_tokens"]


def run(model_keys, datasets, langs, mode: str, seed: int, batch_size: int) -> None:
    for model_key in model_keys:
        for dataset in datasets:
            limit, max_new = _task_spec(dataset, mode)
            for lang in langs:
                items = load_dataset(dataset, lang, limit=limit)
                prompts = [build_prompt(dataset, it["prompt_payload"], lang, cot=(mode == "cot")) for it in items]
                print(f"[gen] {model_key} {dataset} {lang} {mode}: {len(prompts)} prompts")
                outputs = generate(
                    model_key, prompts,
                    temperature=PILOT["temperature"],
                    max_new_tokens=max_new,
                    seed=seed, batch_size=batch_size,
                )
                if len(outputs) != len(prompts):
                    raise RuntimeError(
                        f"{model_key} {dataset} {lang} {mode}: generate returned "
                        f"{len(outputs)} outputs for {len(prompts)} prompts"
                    )
                # Token counts via the generating tokenizer (chapter 3.2.3
                # tokenization control: record raw token count per trajectory).
                tok = get_tokenizer(model_key)
                token_counts = [len(tok(o, add_special_tokens=False)["input_ids"]) for o in outputs]
                rows = []
                for it, p, o, n_tok in zip(items, prompts, outputs, token_counts):
                    rows.append({
                        "id": it["id"], "model": model_key, "dataset": dataset,
                        "lang": lang, "mode": mode, "prompt": p, "output": o,
                        "gold": it["gold"], "n_output_tokens": n_tok,
                    })
                out_path = TRAJ_DIR / f"{model_key}__{dataset}__{lang}__{mode}.jsonl"
                n = write_jsonl(out_path, rows)
                print(f"  wrote {n} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--mode", choices=["cot", "nocot"], default="cot")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for sampling; required for reproducible trajectories")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    run(args.models, args.datasets, args.langs, args.mode, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
