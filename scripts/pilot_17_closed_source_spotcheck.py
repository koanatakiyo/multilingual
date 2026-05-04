"""Closed-source XCOPA spot check.

Generates EN/ZH CoT and no-CoT trajectories for a closed-source model, extracts
the same simple process features as pilot_02, and reports whether adding the
closed-source model preserves the CoT-vs-no-CoT cross-lingual ranking direction.

Outputs are isolated under output/closed_source/ so this appendix check cannot
accidentally contaminate the main open-weight feature/ranking tables.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import API_KEY_DIR, OUTPUT_DIR, PILOT, load_api_key  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.features import extract_features  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402
from utils.ranking import kendall_tau, rank_models  # noqa: E402
from utils.trajectory_parser import extract_answer, parse_steps  # noqa: E402

FEATURE_DIR_MAIN = OUTPUT_DIR / "features"
CLOSED_DIR = OUTPUT_DIR / "closed_source"
CLOSED_TRAJ_DIR = CLOSED_DIR / "trajectories"
CLOSED_FEAT_DIR = CLOSED_DIR / "features"
RANK_DIR = OUTPUT_DIR / "rankings"


def _model_key(provider: str, model: str) -> str:
    key = f"{provider}-{model}".replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", key).strip("-")


def _score_xcopa(row: dict) -> int:
    gold = str(row["gold"]).upper()
    pred = extract_answer(row["output"], ["A", "B"]).upper()
    return int(pred == gold)


def _openai_generate(model: str, prompts: List[str], temperature: float,
                     max_tokens: int, seed: int | None, sleep_s: float) -> List[str]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=load_api_key("openai"))
    out = []
    use_default_temperature = model.lower().startswith("gpt-5") and temperature != 1
    if use_default_temperature:
        print(f"[openai] {model} supports only default temperature; using temperature=1")
    for i, prompt in enumerate(prompts, 1):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        kwargs["temperature"] = 1 if use_default_temperature else temperature
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(5):
            try:
                try:
                    resp = client.chat.completions.create(**kwargs)
                except TypeError:
                    # Older OpenAI SDKs use max_tokens in Chat Completions.
                    fallback = dict(kwargs)
                    fallback["max_tokens"] = fallback.pop("max_completion_tokens")
                    resp = client.chat.completions.create(**fallback)
                text = resp.choices[0].message.content or ""
                out.append(text)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if sleep_s:
            time.sleep(sleep_s)
        if i % 25 == 0:
            print(f"  generated {i}/{len(prompts)}")
    return out


def _gemini_key() -> str:
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(env_name):
            return os.environ[env_name]
    return load_api_key("gemini")


def _gemini_generate(model: str, prompts: List[str], temperature: float,
                     max_tokens: int, sleep_s: float) -> List[str]:
    api_key = _gemini_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    out = []
    for i, prompt in enumerate(prompts, 1):
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if model.lower().startswith("gemini-2.5"):
            # Gemini 2.5 models may spend output budget on hidden thinking.
            # This spot check compares visible CoT/no-CoT traces, so keep
            # the budget on the answer text and avoid no-CoT 400s with tiny
            # maxOutputTokens.
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                out.append(text)
                break
            except urllib.error.HTTPError as e:
                if attempt == 4:
                    body = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Gemini API HTTP {e.code} for model={model}. Response body: {body}"
                    ) from e
                time.sleep(min(2 ** attempt, 16))
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if sleep_s:
            time.sleep(sleep_s)
        if i % 25 == 0:
            print(f"  generated {i}/{len(prompts)}")
    return out


def _anthropic_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for name in ("api_key_anthropic.txt", "api_key_claude.txt", "api_claude.txt"):
        path = API_KEY_DIR / name
        if path.exists():
            return path.read_text().strip().strip('"').strip("'").strip()
    return load_api_key("anthropic")


def _anthropic_generate(model: str, prompts: List[str], temperature: float,
                        max_tokens: int, sleep_s: float) -> List[str]:
    from anthropic import Anthropic  # type: ignore

    client = Anthropic(api_key=_anthropic_key())
    out = []
    for i, prompt in enumerate(prompts, 1):
        for attempt in range(5):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                pieces = [getattr(block, "text", "") for block in resp.content]
                out.append("".join(pieces))
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if sleep_s:
            time.sleep(sleep_s)
        if i % 25 == 0:
            print(f"  generated {i}/{len(prompts)}")
    return out


def _xai_key() -> str:
    for env_name in ("XAI_API_KEY", "GROK_API_KEY"):
        if os.environ.get(env_name):
            return os.environ[env_name]
    return load_api_key("grok")


def _grok_generate(model: str, prompts: List[str], temperature: float,
                   max_tokens: int, seed: int | None, sleep_s: float) -> List[str]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=_xai_key(), base_url="https://api.x.ai/v1")
    out = []
    for i, prompt in enumerate(prompts, 1):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                out.append(text)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
        if sleep_s:
            time.sleep(sleep_s)
        if i % 25 == 0:
            print(f"  generated {i}/{len(prompts)}")
    return out


def _generate(provider: str, model: str, prompts: List[str], temperature: float,
              max_tokens: int, seed: int | None, sleep_s: float) -> List[str]:
    if provider == "openai":
        return _openai_generate(model, prompts, temperature, max_tokens, seed, sleep_s)
    if provider == "gemini":
        return _gemini_generate(model, prompts, temperature, max_tokens, sleep_s)
    if provider == "anthropic":
        return _anthropic_generate(model, prompts, temperature, max_tokens, sleep_s)
    if provider == "grok":
        return _grok_generate(model, prompts, temperature, max_tokens, seed, sleep_s)
    raise ValueError(f"unknown provider: {provider}")


def generate_one(provider: str, model: str, model_key: str, dataset: str, lang: str,
                 mode: str, limit: int, temperature: float, seed: int | None,
                 sleep_s: float, force: bool) -> Path:
    out_path = CLOSED_TRAJ_DIR / f"{model_key}__{dataset}__{lang}__{mode}.jsonl"
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists")
        return out_path

    items = load_dataset(dataset, lang, limit=limit)
    prompts = [
        build_prompt(dataset, it["prompt_payload"], lang, cot=(mode == "cot"))
        for it in items
    ]
    max_tokens = PILOT["max_new_tokens"] if mode == "cot" else PILOT["nocot_max_new_tokens"]
    print(f"[closed] {provider}:{model} {dataset} {lang} {mode}: {len(prompts)} prompts")
    outputs = _generate(provider, model, prompts, temperature, max_tokens, seed, sleep_s)
    rows = []
    for it, prompt, output in zip(items, prompts, outputs):
        rows.append({
            "id": it["id"],
            "model": model_key,
            "provider": provider,
            "provider_model": model,
            "dataset": dataset,
            "lang": lang,
            "mode": mode,
            "prompt": prompt,
            "output": output,
            "gold": it["gold"],
        })
    write_jsonl(out_path, rows)
    print(f"  wrote {len(rows)} -> {out_path}")
    return out_path


def extract_one(traj_path: Path) -> Path:
    rows = read_jsonl(traj_path)
    out_rows = []
    for r in rows:
        parsed = parse_steps(r["output"])
        feats = extract_features(parsed, r["lang"])
        out_rows.append({
            "id": r["id"],
            "model": r["model"],
            "provider": r.get("provider"),
            "provider_model": r.get("provider_model"),
            "dataset": r["dataset"],
            "lang": r["lang"],
            "mode": r["mode"],
            "steps": parsed["steps"],
            "n_steps": parsed["n_steps"],
            "n_output_tokens": None,
            "features": feats,
            "correct": _score_xcopa(r),
            "gold": r["gold"],
        })
    out_path = CLOSED_FEAT_DIR / traj_path.name
    write_jsonl(out_path, out_rows)
    print(f"[feat] {traj_path.name} -> {out_path}")
    return out_path


def _accuracy_from_dirs(dataset: str, mode: str, dirs: Iterable[Path],
                        exclude_models: set[str] | None = None) -> dict:
    groups = defaultdict(list)
    exclude_models = exclude_models or set()
    for root in dirs:
        for p in root.glob(f"*__{dataset}__*__{mode}.jsonl"):
            parts = p.stem.split("__")
            if len(parts) != 4:
                continue
            model, _, lang, _ = parts
            if model in exclude_models:
                continue
            for r in read_jsonl(p):
                groups[(model, lang)].append(r)
    acc = defaultdict(dict)
    for (m, lang), rows in groups.items():
        acc[m][lang] = mean(r["correct"] for r in rows) if rows else 0.0
    return dict(acc)


def _closed_feature_models(dataset: str) -> set[str]:
    models = set()
    for p in CLOSED_FEAT_DIR.glob(f"*__{dataset}__*__*.jsonl"):
        parts = p.stem.split("__")
        if len(parts) == 4:
            models.add(parts[0])
    return models


def _stability(accuracy: dict, lang_pair=("en", "zh")) -> dict:
    models = sorted(
        m for m, per_lang in accuracy.items()
        if all(lang in per_lang for lang in lang_pair)
    )
    if len(models) < 2:
        return {"error": f"need at least two complete models, found {len(models)}"}
    ranks = {}
    values = {}
    for lang in lang_pair:
        values[lang] = {m: accuracy[m][lang] for m in models}
        ranks[lang] = rank_models(values[lang], higher_is_better=True)
    tau = kendall_tau([ranks[lang_pair[0]][m] for m in models],
                      [ranks[lang_pair[1]][m] for m in models])
    return {
        "models": models,
        "n_models": len(models),
        "values": values,
        "ranks": ranks,
        "kendall_tau_en_zh": tau,
    }


def summarize(model_key: str, dataset: str) -> dict:
    open_dirs = [FEATURE_DIR_MAIN]
    with_closed_dirs = [FEATURE_DIR_MAIN, CLOSED_FEAT_DIR]
    closed_models = _closed_feature_models(dataset)
    open_exclude = set(closed_models)
    other_closed = closed_models - {model_key}

    open_cot = _stability(_accuracy_from_dirs(dataset, "cot", open_dirs,
                                             exclude_models=open_exclude))
    open_nocot = _stability(_accuracy_from_dirs(dataset, "nocot", open_dirs,
                                               exclude_models=open_exclude))
    with_cot = _stability(_accuracy_from_dirs(dataset, "cot", with_closed_dirs,
                                             exclude_models=other_closed))
    with_nocot = _stability(_accuracy_from_dirs(dataset, "nocot", with_closed_dirs,
                                               exclude_models=other_closed))

    closed_acc = {}
    for mode in ("cot", "nocot"):
        acc = _accuracy_from_dirs(dataset, mode, [CLOSED_FEAT_DIR])
        closed_acc[mode] = acc.get(model_key, {})

    result = {
        "dataset": dataset,
        "closed_model_key": model_key,
        "closed_model_accuracy": closed_acc,
        "open_weight_rank_stability": {"cot": open_cot, "nocot": open_nocot},
        "with_closed_source_rank_stability": {"cot": with_cot, "nocot": with_nocot},
    }
    for label, block in [
        ("open_weight", result["open_weight_rank_stability"]),
        ("with_closed_source", result["with_closed_source_rank_stability"]),
    ]:
        if "kendall_tau_en_zh" in block["cot"] and "kendall_tau_en_zh" in block["nocot"]:
            block["delta_tau_cot_minus_nocot"] = (
                block["cot"]["kendall_tau_en_zh"] - block["nocot"]["kendall_tau_en_zh"]
            )
            block["cot_less_stable_than_nocot"] = block["delta_tau_cot_minus_nocot"] < 0
    if all(lang in closed_acc["cot"] for lang in ("en", "zh")):
        result["closed_model_accuracy"]["cot_en_minus_zh_gap"] = (
            closed_acc["cot"]["en"] - closed_acc["cot"]["zh"]
        )
    if all(lang in closed_acc["nocot"] for lang in ("en", "zh")):
        result["closed_model_accuracy"]["nocot_en_minus_zh_gap"] = (
            closed_acc["nocot"]["en"] - closed_acc["nocot"]["zh"]
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini", "anthropic", "grok"], default="openai")
    ap.add_argument("--model", default=None,
                    help="Defaults by provider: openai=gpt-4o-mini, "
                         "gemini=gemini-2.5-flash, "
                         "anthropic=claude-sonnet-4-6, "
                         "grok=grok-4.3.")
    ap.add_argument("--model_key", default=None)
    ap.add_argument("--dataset", default="xcopa", choices=["xcopa"])
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--modes", nargs="+", choices=["cot", "nocot"], default=["cot", "nocot"])
    ap.add_argument("--limit", type=int, default=PILOT["xcopa_items"])
    ap.add_argument("--temperature", type=float, default=PILOT["temperature"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--generate_only", action="store_true",
                    help="Generate/extract files but skip rank-stability summary.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    CLOSED_TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    CLOSED_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    RANK_DIR.mkdir(parents=True, exist_ok=True)

    default_models = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "anthropic": "claude-sonnet-4-6",
        "grok": "grok-4.3",
    }
    model = args.model or default_models[args.provider]
    model_key = args.model_key or _model_key(args.provider, model)
    traj_paths = []
    for mode in args.modes:
        for lang in args.langs:
            traj_paths.append(generate_one(
                args.provider,
                model,
                model_key,
                args.dataset,
                lang,
                mode,
                args.limit,
                args.temperature,
                args.seed,
                args.sleep_s,
                args.force,
            ))
    for p in traj_paths:
        extract_one(p)

    if args.generate_only:
        return
    result = summarize(model_key, args.dataset)
    out = Path(args.out) if args.out else (
        RANK_DIR / f"pilot_day3_closed_source_spotcheck_{model_key}.json"
    )
    write_json(out, result)
    print(f"wrote {out}")
    for label in ("open_weight_rank_stability", "with_closed_source_rank_stability"):
        block = result[label]
        if "delta_tau_cot_minus_nocot" in block:
            print(f"{label}: delta_tau_cot_minus_nocot={block['delta_tau_cot_minus_nocot']:+.3f}")


if __name__ == "__main__":
    main()
