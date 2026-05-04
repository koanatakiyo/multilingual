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

_LOCAL_MODEL_CACHE = Path("/data/yandan/transformers_cache")


def _resolve_local(wrapper: str) -> str:
    """Turn a HF-cache wrapper dir (models--*/snapshots/<hash>/) into a path
    from_pretrained can consume directly. If the wrapper already holds a flat
    model dir (config.json at root), return it as-is.
    """
    root = _LOCAL_MODEL_CACHE / wrapper
    if (root / "config.json").exists():
        return str(root)
    snapshots = list(root.glob("models--*/snapshots/*"))
    configured = [s for s in snapshots if (s / "config.json").exists()]
    if configured:
        return str(configured[0])
    if snapshots:
        return str(snapshots[0])
    return str(root)  # fall back; from_pretrained will raise a clear error


# Keep plan-facing keys stable; each hf_id resolves to the snapshot containing
# config.json inside the user's transformers_cache.
#
# chat_template_kwargs: extra kwargs for tokenizer.apply_chat_template.
#   Qwen3-8B runs in no_thinking mode (enable_thinking=False) so the template
#   injects an empty <think></think> and the model produces only the visible,
#   structured "Step 1 / 第一步 …" output. Rationale: (1) the trajectory parser
#   consumes the visible stepwise output, not an internal scratchpad; (2) other
#   models have no thinking/output split, so using <think> content would create
#   a cross-model confound in the CoT-as-observation-window analysis.
MODELS = {
    "qwen3-8b":     {"hf_id": _resolve_local("qwen3-8b"),          "scale_b": 8,  "primary": True, "hypergraph": True, "cka": True,
                     "chat_template_kwargs": {"enable_thinking": False}},
    "llama3.1-8b":  {"hf_id": _resolve_local("llama31-8b"),        "scale_b": 8,  "primary": True, "hypergraph": True, "cka": True},
    # Pure Mistral arch. The newer Ministral-8B-Instruct-2512 ships as FP8
    # Mistral3 VLM with bleeding-edge config that transformers 4.55 / vLLM
    # 0.8.5 don't yet handle.
    "Ministral-8B-Instruct-2410": {"hf_id": _resolve_local("ministral-8b-instruct-2410"), "scale_b": 8, "primary": True, "hypergraph": True, "cka": False},
    # Pure text arch. The Gemma-4 checkpoint ships as a VLM whose model_type
    # is not yet supported by transformers 4.55 / vLLM 0.8.5 — parallel to
    # the Ministral-2512 case.
    # Gemma-3 is trained in bfloat16 and emits pure <pad> output under the
    # default fp16 downcast (sampler saturates to pad id 0). A6000 supports
    # bfloat16 natively; pin it here.
    "Gemma-3-4B-Instruct":        {"hf_id": _resolve_local("gemma-3-4b-it"),    "scale_b": 4,  "primary": True, "hypergraph": False, "cka": False, "dtype": "bfloat16"},
    # Vanilla Phi-4 (instruct). Replaced Phi-4-reasoning after the latter's
    # extended <think> traces (26-33 visible steps vs 4-5 for instruct models)
    # were found to dominate group-level averages and obscure the domain
    # contrast pattern. Old reasoning data archived under
    # output/trajectories_phi_reasoning_backup/.
    "Phi-4":                      {"hf_id": _resolve_local("phi-4"),            "scale_b": 14, "primary": True, "hypergraph": False, "cka": False},
    "Qwen2.5-14B-Instruct":        {"hf_id": _resolve_local("qwen2.5-14B"),      "scale_b": 14, "primary": True, "hypergraph": False, "cka": False},
    "aya-expanse-8b":              {"hf_id": _resolve_local("aya-expanse"),      "scale_b": 8,  "primary": True, "hypergraph": False, "cka": False},
    "c4ai-command-r7b-12-2024":    {"hf_id": _resolve_local("c4ai-command-r7b-12-2024"), "scale_b": 7, "primary": True, "hypergraph": False, "cka": False},
    "Yi-1.5-9B-Chat":              {"hf_id": _resolve_local("yi-1.5-9b-chat"),   "scale_b": 9,  "primary": True, "hypergraph": False, "cka": False,
                                     "max_model_len": 4096},
    # vLLM 0.8.5.post1's V1 MLA backend can crash on this checkpoint during
    # batched generation (`MLACommonMetadataBuilder.page_size`). V0 uses the
    # older Triton MLA path and passes repeated generate() calls.
    "DeepSeek-V2-Lite-Chat":       {"hf_id": _resolve_local("deepseek-v2-lite-chat"), "scale_b": 15.7, "active_b": 2.4, "primary": True, "hypergraph": False, "cka": False,
                                     "tensor_parallel_size": 2,
                                     # vLLM 0.8.5 v1 engine + DeepSeek MLA hits a cuda-graph capture bug
                                     # ('MLACommonMetadataBuilder' has no attribute 'page_size'). enforce_eager
                                     # disables graph capture and is reliable. VLLM_USE_V1=0 was attempted
                                     # but ignored by 0.8.5 — kept for documentation.
                                     "vllm_env": {"VLLM_USE_V1": "0"},
                                     "vllm_kwargs": {"enforce_eager": True}},
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
    # Chapter 3.2.3 splits the old "verification" list into two functionally
    # distinct categories. We report them separately so that if divergence is
    # concentrated in epistemic behavior, this says something about
    # uncertainty management rather than forward-reasoning style.
    #
    # - procedural_markers: logical connectives that advance the chain
    # - epistemic_markers : uncertainty / metacognitive control
    "procedural_markers": {
        "en": ["therefore", "hence", "thus", " so ", "so,", "so.", "as a result",
               "it follows that", "consequently"],
        "zh": ["因此", "所以", "故", "由此", "进而", "于是"],
        "ja": ["よって", "したがって", "ゆえに", "それゆえ"],
        "fr": ["donc", "ainsi", "par conséquent", "alors"],
    },
    # Each language's list covers the same two functional sub-categories so
    # that cross-lingual counts are comparable:
    #   (a) hedging / uncertainty externalisation
    #   (b) metacognitive re-evaluation (pause, re-check, reconsider)
    # Keyword set rebuilt after inspecting 500 ZH XCOPA trajectories — the
    # earlier ZH list (等等/其实/不过/嗯) fired in < 1% of traces because models
    # don't use colloquial markers in formal reasoning; "可能" alone covered
    # 293/500 trajectories.
    "epistemic_markers": {
        "en": [
            # hedging
            "maybe", "perhaps", "probably", "possibly", "might", "likely",
            "seems", "appears to", "i think", "i believe",
            "not sure", "uncertain",
            # re-evaluation
            "wait", "hmm", "actually", "on second thought",
            "let me check", "let me verify", "let me reconsider",
            "let me re-examine", "but wait", "re-examine",
            "double-check", "double check",
        ],
        "zh": [
            # hedging
            "可能", "也许", "或许", "大概", "大约", "估计",
            "应该", "似乎", "看起来", "看来", "不确定",
            "我认为", "我觉得", "需要考虑",
            # re-evaluation
            "让我想想", "让我重新考虑", "让我重新检查", "让我核实",
            "重新考虑", "重新审视", "重新思考", "等等",
        ],
        "ja": [
            # hedging
            "かもしれない", "たぶん", "おそらく", "多分", "あるいは",
            "のようだ", "と思う",
            # re-evaluation
            "ちょっと待って", "もう一度", "やっぱり", "実は", "再確認",
        ],
        "fr": [
            # hedging
            "peut-être", "probablement", "sans doute", "semble",
            "apparemment", "je pense", "je crois",
            # re-evaluation
            "en fait", "attendez", "en réalité", "à bien y penser",
        ],
    },
}


def get_env_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


API_KEY_DIR = PROJECT_ROOT / "APIkeys"


def load_api_key(provider: str) -> str:
    """Read a one-line API key from APIkeys/api_key_<provider>.txt.

    Tolerates surrounding quotes and trailing whitespace.
    """
    path = API_KEY_DIR / f"api_key_{provider}.txt"
    raw = path.read_text().strip()
    return raw.strip('"').strip("'").strip()
