"""HuggingFace model loading + generation wrapper.

Keeps one model resident at a time to limit GPU memory pressure during pilot.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import MODELS

_ACTIVE = {"key": None, "model": None, "tokenizer": None}


def load_model(key: str):
    if _ACTIVE["key"] == key and _ACTIVE["model"] is not None:
        return _ACTIVE["model"], _ACTIVE["tokenizer"]
    _unload()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    hf_id = MODELS[key]["hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
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


def chat_format(tokenizer, prompt: str) -> str:
    """Render as chat if a template is available, otherwise raw."""
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def generate(model_key: str, prompts: List[str], temperature: float = 0.7,
             max_new_tokens: int = 768, seed: Optional[int] = None) -> List[str]:
    import torch  # type: ignore
    model, tok = load_model(model_key)
    if seed is not None:
        torch.manual_seed(seed)
    outs: List[str] = []
    for p in prompts:
        rendered = chat_format(tok, p)
        enc = tok(rendered, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        outs.append(text)
    return outs


def hidden_states(model_key: str, prompt: str, layers: Optional[List[int]] = None):
    """Return per-layer mean-pooled hidden states for a single prompt (CKA use)."""
    import torch  # type: ignore
    model, tok = load_model(model_key)
    rendered = chat_format(tok, prompt)
    enc = tok(rendered, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # tuple of (L+1, 1, T, D)
    mask = enc["attention_mask"][0].bool()
    pooled = []
    for i, h in enumerate(hs):
        if layers is not None and i not in layers:
            continue
        v = h[0][mask].mean(dim=0)
        pooled.append(v.cpu().float().numpy())
    return pooled
