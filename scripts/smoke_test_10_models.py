"""Smoke-test the 10-model set before full trajectory generation.

This intentionally does only the first gate:
  - can vLLM load each model?
  - can the tokenizer chat template render EN/ZH prompts?
  - does generation produce parseable, numbered CoT steps?
  - what output language does the model actually use?

It does not write trajectory files and does not orchestrate the full run.
Use this before investing GPU time in XCOPA/MGSM/XStoryCloze generation.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import MODELS, PILOT, REPORT_DIR  # noqa: E402
from utils.data_loader import load_dataset  # noqa: E402
from utils.io import write_json  # noqa: E402
from utils.models import apply_vllm_env  # noqa: E402
from utils.prompts import build_prompt  # noqa: E402
from utils.trajectory_parser import _match_step, extract_answer, parse_steps  # noqa: E402


_CJK_RE = re.compile(r"[㐀-鿿぀-ゟ゠-ヿ㇀-㇯]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")


def _detect_output_lang(text: str) -> str:
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = cjk + latin
    if total == 0:
        return "unknown"
    return "zh" if cjk / total > 0.20 else "en"


def _lang_ok(expected: str, detected: str) -> bool:
    if expected in {"zh", "ja"}:
        return detected == "zh"
    if expected in {"en", "fr"}:
        return detected == "en"
    return detected != "unknown"


def _parse_overrides(raw: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"bad override '{item}', expected MODEL=INT")
        key, val = item.split("=", 1)
        out[key.strip()] = int(val.strip())
    return out


def _unload_llm(llm) -> None:
    if llm is not None:
        del llm
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


def _make_prompts(dataset: str, langs: List[str], n_items: int, start_index: int):
    prompts = []
    rows = []
    for lang in langs:
        items = load_dataset(dataset, lang, limit=start_index + n_items)
        picked = items[start_index : start_index + n_items]
        if len(picked) < n_items:
            raise ValueError(
                f"{dataset}/{lang}: requested {n_items} items from index "
                f"{start_index}, found {len(picked)}"
            )
        for it in picked:
            prompt = build_prompt(dataset, it["prompt_payload"], lang, cot=True)
            prompts.append(prompt)
            rows.append({
                "id": it["id"],
                "dataset": dataset,
                "lang": lang,
                "gold": it["gold"],
                "prompt": prompt,
            })
    return prompts, rows


def _render_chat(tokenizer, prompts: List[str], template_kwargs: dict | None):
    rendered = []
    errors = []
    for idx, prompt in enumerate(prompts):
        try:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if template_kwargs:
                kwargs.update(template_kwargs)
            rendered.append(tokenizer.apply_chat_template(messages, **kwargs))
        except Exception as exc:
            rendered.append(prompt)
            errors.append({
                "prompt_index": idx,
                "type": type(exc).__name__,
                "message": str(exc),
            })
    return rendered, errors


def _row_result(row: dict, output: str, min_steps: int, min_numbered: int,
                require_output_lang: bool) -> dict:
    parsed = parse_steps(output)
    lines = [ln.strip() for ln in parsed["body"].splitlines() if ln.strip()]
    numbered_hits = sum(1 for ln in lines if _match_step(ln))
    out_lang = _detect_output_lang(output)
    lang_ok = _lang_ok(row["lang"], out_lang)
    answer = extract_answer(output, ["A", "B"])
    parseable = parsed["n_steps"] >= min_steps
    numbered_ok = numbered_hits >= min_numbered
    answer_present = bool(answer)
    passed = parseable and numbered_ok and answer_present
    if require_output_lang:
        passed = passed and lang_ok
    return {
        "id": row["id"],
        "lang": row["lang"],
        "gold": row["gold"],
        "passed": passed,
        "parseable": parseable,
        "step_numbered": numbered_ok,
        "answer_present": answer_present,
        "answer": answer,
        "n_steps": parsed["n_steps"],
        "numbered_hits": numbered_hits,
        "output_lang": out_lang,
        "output_lang_ok": lang_ok,
        "answer_line": parsed["answer_line"],
        "output_preview": output[:1200],
    }


def _smoke_one_model(model_key: str, args, tp_overrides: Dict[str, int]) -> dict:
    apply_vllm_env(model_key)
    from vllm import LLM, SamplingParams  # type: ignore

    cfg = MODELS[model_key]
    model_path = cfg["hf_id"]
    tp = tp_overrides.get(model_key)
    if tp is None:
        tp = (
            args.tensor_parallel_size
            if args.tensor_parallel_size is not None
            else int(cfg.get("tensor_parallel_size", 1))
        )
    max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else int(cfg.get("max_model_len", 5120))
    )
    started = time.time()
    llm = None
    result = {
        "model": model_key,
        "model_path": model_path,
        "tensor_parallel_size": tp,
        "max_model_len": max_model_len,
        "dtype": args.dtype,
        "vllm_env": cfg.get("vllm_env", {}),
        "loaded": False,
        "chat_template_ok": False,
        "chat_template_errors": [],
        "passed": False,
        "elapsed_sec": None,
        "rows": [],
        "error": None,
    }
    try:
        llm = LLM(
            model=model_path,
            dtype=args.dtype,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        result["loaded"] = True
        tok = llm.get_tokenizer()
        prompts, row_meta = _make_prompts(
            args.dataset, args.langs, args.n_items, args.start_index
        )
        tpl_kw = cfg.get("chat_template_kwargs")
        rendered, template_errors = _render_chat(tok, prompts, tpl_kw)
        result["chat_template_errors"] = template_errors
        result["chat_template_ok"] = not template_errors
        sampling = SamplingParams(
            temperature=max(float(args.temperature), 0.0),
            max_tokens=int(args.max_new_tokens),
            seed=args.seed,
        )
        outputs = llm.generate(rendered, sampling)
        texts = [r.outputs[0].text for r in outputs]
        rows = [
            _row_result(
                row, text,
                min_steps=args.min_steps,
                min_numbered=args.min_numbered,
                require_output_lang=args.require_output_lang,
            )
            for row, text in zip(row_meta, texts)
        ]
        result["rows"] = rows
        result["passed"] = all(r["passed"] for r in rows)
        if template_errors and not args.allow_raw_prompt_fallback:
            result["passed"] = False
    except Exception as exc:  # smoke report should capture and continue.
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        _unload_llm(llm)
        result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--n_items", type=int, default=1)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=PILOT["temperature"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float16",
                    help="A6000-safe default; use bfloat16/auto if desired.")
    ap.add_argument("--tensor_parallel_size", type=int, default=None)
    ap.add_argument(
        "--tp_override", action="append", default=[],
        help="Per-model TP override, e.g. DeepSeek-V2-Lite-Chat=2. "
             "May be supplied multiple times.",
    )
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--min_numbered", type=int, default=2)
    ap.add_argument("--require_output_lang", action="store_true",
                    help="Fail a row if detected output language differs from prompt language.")
    ap.add_argument("--allow_raw_prompt_fallback", action="store_true",
                    help="Do not fail a model when apply_chat_template falls back to raw prompts.")
    ap.add_argument("--out", default=str(REPORT_DIR / "smoke_test_10_models.json"))
    args = ap.parse_args()

    missing = [m for m in args.models if m not in MODELS]
    if missing:
        raise SystemExit(f"unknown model key(s): {missing}")

    tp_overrides = _parse_overrides(args.tp_override)
    report = {
        "args": {
            "models": args.models,
            "dataset": args.dataset,
            "langs": args.langs,
            "n_items": args.n_items,
            "start_index": args.start_index,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "tp_overrides": tp_overrides,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "min_steps": args.min_steps,
            "min_numbered": args.min_numbered,
            "require_output_lang": args.require_output_lang,
            "allow_raw_prompt_fallback": args.allow_raw_prompt_fallback,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "results": [],
    }

    for model_key in args.models:
        print(f"[smoke] {model_key} ...", flush=True)
        res = _smoke_one_model(model_key, args, tp_overrides)
        report["results"].append(res)
        status = "PASS" if res["passed"] else "FAIL"
        if res["error"]:
            detail = f"{res['error']['type']}: {res['error']['message']}"
        else:
            template = "tpl=ok" if res["chat_template_ok"] else "tpl=FAIL"
            detail = ", ".join(
                f"{r['lang']} steps={r['n_steps']} numbered={r['numbered_hits']} "
                f"out={r['output_lang']} ans={r['answer'] or '-'}"
                for r in res["rows"]
            )
            detail = f"{template}; {detail}"
        print(f"  {status} ({res['elapsed_sec']}s) {detail}", flush=True)

    report["n_models"] = len(report["results"])
    report["n_passed"] = sum(1 for r in report["results"] if r["passed"])
    report["all_passed"] = report["n_passed"] == report["n_models"]
    write_json(Path(args.out), report)
    print(f"\nWrote {args.out}")
    print(f"Summary: {report['n_passed']}/{report['n_models']} passed")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
