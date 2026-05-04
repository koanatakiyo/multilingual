"""Day 2 afternoon: prompt language flip.

On a subset of accuracy-matched XCOPA items, generate CoT trajectories with
the *instruction language flipped* relative to the content language: English
instructions on Chinese-content questions, and Chinese instructions on
English-content questions. Extract simple features and compare mean |EN-ZH|
divergence under flipped vs. matched prompt language.

If divergence tracks content language (flipped ≈ matched), the effect is
content-driven; if it tracks instruction language, the effect is generation-
mode-driven. See Chapter 4.3.1 / Pilot Day 2 afternoon.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, PILOT, RANK_DIR, TRAJ_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.features import extract_features  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.models import generate, get_tokenizer  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402
from utils.trajectory_parser import parse_steps  # noqa: E402

FEATURES = ["step_count", "procedural_rate", "epistemic_rate", "dependency_depth"]

# When prompts are flipped, models commonly switch their *output* language to
# match the instruction language rather than the content language. Applying a
# content-language keyword list to a switched output collapses marker counts
# to zero — a parser bug, not a behavioural finding. Detect output language
# per row and use the matching keyword list.
_CJK_RE = re.compile(r"[㐀-鿿぀-ゟ゠-ヿ㇀-㇯]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")


def _detect_output_lang(text: str) -> str:
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = cjk + latin
    if total == 0:
        return "en"
    return "zh" if cjk / total > 0.20 else "en"


def _matched_ids(model: str, dataset: str):
    en_path = FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl"
    zh_path = FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl"
    if not en_path.exists() or not zh_path.exists():
        raise FileNotFoundError(f"missing EN/ZH CoT feature files for {model}/{dataset}")
    en = read_jsonl(en_path)
    zh = read_jsonl(zh_path)
    return sorted(accuracy_matched_ids(
        {r["id"]: bool(r["correct"]) for r in en},
        {r["id"]: bool(r["correct"]) for r in zh},
    ))


def _generate_flip(model: str, items, content_lang: str, instr_lang: str,
                    seed: int, batch_size: int):
    prompts = [build_prompt("xcopa", it["prompt_payload"], content_lang,
                             cot=True, instr_lang=instr_lang) for it in items]
    outputs = generate(
        model, prompts,
        temperature=PILOT["temperature"],
        max_new_tokens=PILOT["max_new_tokens"],
        seed=seed, batch_size=batch_size,
    )
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"{model} {content_lang}->{instr_lang}: generate returned "
            f"{len(outputs)} outputs for {len(prompts)} prompts"
        )
    tok = get_tokenizer(model)
    token_counts = [len(tok(o, add_special_tokens=False)["input_ids"]) for o in outputs]
    rows = []
    for it, p, o, n_tok in zip(items, prompts, outputs, token_counts):
        rows.append({
            "id": it["id"], "model": model, "dataset": "xcopa",
            "content_lang": content_lang, "instr_lang": instr_lang,
            "prompt": p, "output": o, "gold": it["gold"],
            "n_output_tokens": n_tok,
        })
    return rows


def _features_from_rows(rows):
    """Extract features per row, detecting output language per trajectory.

    The earlier signature took `parse_lang` (set to the *content* language).
    That was wrong for flipped prompts: 4 of 5 models follow the instruction
    language for output, so applying the content-language keyword list to a
    switched output zeroed out the marker counts. Per-row detection is the
    principled fix and a no-op for matched runs (where output language always
    equals content language).
    """
    out = []
    for r in rows:
        parsed = parse_steps(r["output"])
        out_lang = _detect_output_lang(r["output"])
        feats = extract_features(parsed, out_lang)
        out.append({
            "id": r["id"],
            "out_lang": out_lang,
            "n_output_tokens": r["n_output_tokens"],
            "features": feats,
        })
    return out


def _mean_abs_div(en_feats, zh_feats):
    en_by_id = {r["id"]: r for r in en_feats}
    zh_by_id = {r["id"]: r for r in zh_feats}
    shared = sorted(en_by_id.keys() & zh_by_id.keys())
    per_feat = {}
    for f in FEATURES:
        diffs = [abs(en_by_id[i]["features"][f] - zh_by_id[i]["features"][f]) for i in shared]
        per_feat[f] = mean(diffs) if diffs else 0.0
    return per_feat, len(shared)


def run(model: str, n_items: int, seed: int, batch_size: int):
    matched = set(_matched_ids(model, "xcopa"))
    ids = [it["id"] for it in load_dataset("xcopa", "en") if it["id"] in matched][:n_items]
    if not ids:
        raise ValueError(f"no accuracy-matched XCOPA EN/ZH items for {model}")
    id_set = set(ids)
    en_items = [it for it in load_dataset("xcopa", "en") if it["id"] in id_set]
    zh_items = [it for it in load_dataset("xcopa", "zh") if it["id"] in id_set]
    order_en = {it["id"]: it for it in en_items}
    order_zh = {it["id"]: it for it in zh_items}
    en_items = [order_en[i] for i in ids]
    zh_items = [order_zh[i] for i in ids]

    # Flipped: EN content + ZH instruction; ZH content + EN instruction.
    flip_en = _generate_flip(model, en_items, "en", "zh", seed, batch_size)
    flip_zh = _generate_flip(model, zh_items, "zh", "en", seed, batch_size)
    write_jsonl(TRAJ_DIR / f"{model}__xcopa__en__flip_zh_instr.jsonl", flip_en)
    write_jsonl(TRAJ_DIR / f"{model}__xcopa__zh__flip_en_instr.jsonl", flip_zh)

    # Detect output language per row (handles instruction-language-following).
    en_feats_flip = _features_from_rows(flip_en)
    zh_feats_flip = _features_from_rows(flip_zh)
    flip_div, n_flip = _mean_abs_div(en_feats_flip, zh_feats_flip)

    # Output-language-following profile — itself a finding worth keeping.
    out_lang_following = {
        "en_content_zh_instr_to_zh_output_pct": (
            sum(1 for r in en_feats_flip if r["out_lang"] == "zh") / max(1, len(en_feats_flip))
        ),
        "zh_content_en_instr_to_en_output_pct": (
            sum(1 for r in zh_feats_flip if r["out_lang"] == "en") / max(1, len(zh_feats_flip))
        ),
    }

    # Baseline: matched-prompt runs from pilot_01/02 on the same 50 ids.
    base_en = [r for r in read_jsonl(FEAT_DIR / f"{model}__xcopa__en__cot.jsonl") if r["id"] in id_set]
    base_zh = [r for r in read_jsonl(FEAT_DIR / f"{model}__xcopa__zh__cot.jsonl") if r["id"] in id_set]
    base_div, n_base = _mean_abs_div(base_en, base_zh)

    return {
        "model": model,
        "n_items": len(ids),
        "matched_baseline_n": n_base,
        "flipped_n": n_flip,
        "matched_baseline_divergence": base_div,
        "flipped_divergence": flip_div,
        "delta_flip_minus_matched": {f: flip_div[f] - base_div[f] for f in FEATURES},
        "output_language_following": out_lang_following,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-8b", "llama3.1-8b"])
    ap.add_argument("--n_items", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day2_prompt_flip.json"))
    args = ap.parse_args()

    result = {m: run(m, args.n_items, args.seed, args.batch_size) for m in args.models}
    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
