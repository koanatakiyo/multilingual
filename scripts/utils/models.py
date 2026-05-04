"""Model wrappers: vLLM for batch generation, HuggingFace for hidden states.

vLLM's continuous batching is ~5-10x faster than HF `model.generate()` on
the scale we need for trajectory generation. But vLLM does not expose
per-layer hidden states, so `hidden_states()` (used by pilot_12 CKA) stays
on the HF path. The two backends share the GPU: loading one evicts the other.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .config import MODELS

# HF backend — used by hidden_states() and as a tokenizer fallback.
_ACTIVE = {"key": None, "model": None, "tokenizer": None}

# vLLM backend — used by generate().
_VLLM = {"key": None, "llm": None}
_VLLM_ENV_KEYS = ("VLLM_USE_V1", "VLLM_MLA_DISABLE", "VLLM_ATTENTION_BACKEND")
_VLLM_ENV_ORIGINAL = {name: os.environ.get(name) for name in _VLLM_ENV_KEYS}


def apply_vllm_env(key: str) -> None:
    """Apply per-model vLLM environment overrides before importing/creating LLM."""
    overrides = MODELS[key].get("vllm_env", {})
    for name in _VLLM_ENV_KEYS:
        if name in overrides:
            os.environ[name] = str(overrides[name])
        elif _VLLM_ENV_ORIGINAL[name] is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = _VLLM_ENV_ORIGINAL[name]


def load_model(key: str):
    """Load HF model + tokenizer. Used only by hidden_states() now."""
    if _ACTIVE["key"] == key and _ACTIVE["model"] is not None:
        return _ACTIVE["model"], _ACTIVE["tokenizer"]
    _unload()
    _unload_vllm()  # free vLLM GPU before loading HF
    from transformers import (  # type: ignore
        AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
        AutoTokenizer,
    )
    import torch  # type: ignore

    hf_id = MODELS[key]["hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Left-padding is required for correct batched causal generation.
    tok.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = {"": 0} if torch.cuda.is_available() else None

    # Ministral-3-8B-Instruct ships as a VLM (Mistral3ForConditionalGeneration,
    # text backbone + pixtral vision tower + lm_head). AutoModelForCausalLM
    # refuses the outer `mistral3` config, so load via the image-text-to-text
    # class.
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    if getattr(cfg, "model_type", "") == "mistral3":
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
        )
    model.eval()
    _ACTIVE.update({"key": key, "model": model, "tokenizer": tok})
    return model, tok


def _unload():
    if _ACTIVE["model"] is None:
        return
    import torch  # type: ignore
    del _ACTIVE["model"]
    del _ACTIVE["tokenizer"]
    _ACTIVE["model"] = None
    _ACTIVE["tokenizer"] = None
    _ACTIVE["key"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_vllm():
    if _VLLM["llm"] is None:
        return
    import gc
    import torch  # type: ignore
    del _VLLM["llm"]
    _VLLM["llm"] = None
    _VLLM["key"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # vLLM leaves distributed / parallel state allocated; tear it down so the
    # next LLM() call (possibly a different model) can re-initialise cleanly.
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment, destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


def _load_vllm(key: str):
    if _VLLM["key"] == key and _VLLM["llm"] is not None:
        return _VLLM["llm"]
    _unload_vllm()
    _unload()  # vLLM wants the whole GPU
    apply_vllm_env(key)
    from vllm import LLM  # type: ignore
    hf_id = MODELS[key]["hf_id"]
    dtype = MODELS[key].get("dtype", "float16")
    max_model_len = int(MODELS[key].get("max_model_len", 5120))
    tensor_parallel_size = int(MODELS[key].get("tensor_parallel_size", 1))
    # max_model_len = prompt cap (4096) + max_new (768) with a small buffer.
    # gpu_memory_utilization 0.85 leaves headroom so model swaps don't OOM on
    # fragmented memory.
    extra_kwargs = MODELS[key].get("vllm_kwargs", {}) or {}
    llm = LLM(
        model=hf_id,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        **extra_kwargs,
    )
    _VLLM.update({"key": key, "llm": llm})
    return llm


def get_tokenizer(key: str):
    """Return the HF tokenizer for `key`, reusing whichever backend is live."""
    if _VLLM["key"] == key and _VLLM["llm"] is not None:
        return _VLLM["llm"].get_tokenizer()
    if _ACTIVE["key"] == key and _ACTIVE["tokenizer"] is not None:
        return _ACTIVE["tokenizer"]
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(MODELS[key]["hf_id"], trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_format(tokenizer, prompt: str, template_kwargs: Optional[dict] = None) -> str:
    """Render as chat if a template is available, otherwise raw.

    template_kwargs: extra kwargs for apply_chat_template (e.g. Qwen3's
    enable_thinking=False). See MODELS[key]["chat_template_kwargs"].
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if template_kwargs:
            kwargs.update(template_kwargs)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return prompt


def generate(
    model_key: str,
    prompts: List[str],
    temperature: float = 0.7,
    max_new_tokens: int = 768,
    seed: Optional[int] = 42,
    batch_size: int = 8,  # kept for signature compat; vLLM does continuous batching
) -> List[str]:
    """Batch generation via vLLM. Returns completions aligned to `prompts`."""
    apply_vllm_env(model_key)
    from vllm import SamplingParams  # type: ignore
    llm = _load_vllm(model_key)
    tok = llm.get_tokenizer()
    tpl_kw = MODELS[model_key].get("chat_template_kwargs")
    rendered = [chat_format(tok, p, tpl_kw) for p in prompts]

    sampling = SamplingParams(
        temperature=max(float(temperature), 0.0),
        max_tokens=int(max_new_tokens),
        seed=seed,
    )
    results = llm.generate(rendered, sampling)
    return [r.outputs[0].text for r in results]


def hidden_states(model_key: str, prompt: str, pooling: str = "last"):
    """Return per-layer hidden states for a single prompt (CKA use).

    pooling:
        - "last":  use the final non-pad token's hidden state per layer.
                   This is token-count-invariant across EN/ZH prompts, which
                   differ in length, and so avoids a length confound in CKA.
        - "mean":  mean over all non-pad tokens (length-dependent; avoid for CKA).
    """
    import torch  # type: ignore
    model, tok = load_model(model_key)
    tpl_kw = MODELS[model_key].get("chat_template_kwargs")
    rendered = chat_format(tok, prompt, tpl_kw)
    enc = tok(rendered, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # tuple (L+1,) of (1, T, D)
    mask = enc["attention_mask"][0].bool()
    pooled = []
    for h in hs:
        tokens = h[0][mask]
        if pooling == "last":
            v = tokens[-1]
        else:
            v = tokens.mean(dim=0)
        pooled.append(v.cpu().float().numpy())
    return pooled
