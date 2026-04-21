"""Central configuration: paths, dataset configs, model configs."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAJ_DIR = OUTPUT_DIR / "trajectories"
FEAT_DIR = OUTPUT_DIR / "features"
RANK_DIR = OUTPUT_DIR / "rankings"
REPORT_DIR = OUTPUT_DIR / "reports"
FIG_DIR = OUTPUT_DIR / "figures"
HG_DIR = OUTPUT_DIR / "hypergraph"
CKA_DIR = OUTPUT_DIR / "cka"

for _d in (TRAJ_DIR, FEAT_DIR, RANK_DIR, REPORT_DIR, FIG_DIR, HG_DIR, CKA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    # XCOPA: English from SuperGLUE/COPA; Chinese from XCOPA/zh.
    "xcopa": {
        "en": {"hf_id": "super_glue", "subset": "copa", "splits": ["validation", "test"]},
        "zh": {"hf_id": "xcopa", "subset": "zh", "splits": ["validation", "test"]},
        "local_dir": DATA_DIR / "xcopa",
    },
    "xstorycloze": {
        "hf_id": "juletxara/xstory_cloze",
        "subsets": {"en": "en", "zh": "zh"},
        "splits": ["eval"],
        "local_dir": DATA_DIR / "xstorycloze",
    },
    # MGSM: canonical TSVs from google-research/url-nlp (tab: question\tanswer_number).
    "mgsm": {
        "source": "github_tsv",
        "url_tpl": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_{lang}.tsv",
        "subsets": {"en": "en", "zh": "zh", "ja": "ja", "fr": "fr"},
        "splits": ["test"],
        "local_dir": DATA_DIR / "mgsm",
    },
    "belebele": {
        "hf_id": "facebook/belebele",
        "subsets": {
            "en": "eng_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "fr": "fra_Latn",
        },
        "splits": ["test"],
        "local_dir": DATA_DIR / "belebele",
    },
}

MODELS = {
    "qwen3-8b": {"hf_id": "Qwen/Qwen3-8B", "scale_b": 8, "primary": True, "hypergraph": True, "cka": True},
    "llama3.1-8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "scale_b": 8, "primary": True, "hypergraph": True, "cka": True},
    "ministral3-8b": {"hf_id": "mistralai/Ministral-8B-Instruct-2410", "scale_b": 8, "primary": True, "hypergraph": True, "cka": False},
    "gemma4-e4b": {"hf_id": "google/gemma-2-2b-it", "scale_b": 4, "primary": True, "hypergraph": False, "cka": False},
    "phi4-14b": {"hf_id": "microsoft/Phi-4", "scale_b": 14, "primary": True, "hypergraph": False, "cka": False},
}

LANGUAGES = ["en", "zh", "ja", "fr"]
PRIMARY_LANGS = ["en", "zh"]

PILOT = {
    "xcopa_items": 100,
    "mgsm_items": 250,
    "belebele_items": 900,
    "anchor_items": 10,
    "temperature": 0.7,
    "max_new_tokens": 768,
    "nocot_max_new_tokens": 32,
    "bootstrap_iters": 10,
    "ensemble_k": 3,
    "ensemble_k_fallback": 5,
    "judge_tau": 0.5,
    "labse_threshold": 0.55,
    "verification_keywords": {
        "en": [
            "verify", "verification", "check", "double-check", "recheck",
            "confirm", "ensure", "wait", "hmm", "let me reconsider",
            "actually", "on second thought", "but wait", "re-examine",
        ],
        "zh": [
            "验证", "检查", "再检查", "确认", "确保", "等等",
            "嗯", "让我重新考虑", "实际上", "再想想", "不过",
            "重新审视", "再看看", "其实",
        ],
        "ja": ["確認", "検証", "もう一度", "実は", "ちょっと待って"],
        "fr": ["vérifier", "vérification", "confirmer", "attendez", "en fait"],
    },
}


def get_env_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
