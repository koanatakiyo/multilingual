"""Day 1 afternoon: ranking comparison (survival test + domain contrast + multi-language).

The paper's main hypothesis is that rankings shift when we compare accuracy in
one language to process features in the *other* language (chapter 3.2.7 CRSI).
We therefore report three distinct tau families:

    - within-lang:   tau(acc-rank-L,  feat-rank-L)
    - cross-lingual: tau(acc-rank-EN, feat-rank-ZH) and the ZH→EN variant
    - feature-only:  tau(feat-rank-EN, feat-rank-ZH)     (pure cross-lingual
                     stability of the process metric itself)

and CRSI is computed on the cross-lingual family.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.ranking import crsi, inversion_rate, kendall_tau, rank_models  # noqa: E402

# Per chapter 3.2.3 / 4.3.1 the Day-1 ranking feature set is: step count,
# procedural marker rate, epistemic marker rate, dependency depth (and the
# alignment-based unmatched step ratio, which pilot_08 adds later). avg_step_tokens
# is a tokenization-control measurement (pilot_03b), not a process feature, and
# is kept out of the ranking set so it does not pollute CRSI.
FEATURE_NAMES = [
    "step_count",
    "procedural_rate",
    "epistemic_rate",
    "dependency_depth",
]


def _load_feats(dataset: str, mode: str = "cot"):
    groups = defaultdict(list)
    for p in FEAT_DIR.glob(f"*__{dataset}__*__{mode}.jsonl"):
        parts = p.stem.split("__")
        model_key, _, lang, mode_ = parts
        for r in read_jsonl(p):
            groups[(model_key, lang)].append(r)
    return groups


def _per_item_correctness(rows):
    return {r["id"]: bool(r["correct"]) for r in rows}


def analyze_dataset(dataset: str, lang_pair=("en", "zh"), exclude_models=None):
    """Analyse one dataset for a single EN↔X language pair. Returns None if
    either language has no trajectories under that dataset.

    exclude_models: iterable of model keys to drop before analysis. Use this
    to produce a "primary" run without reasoning-tuned outliers (e.g.
    Phi-4-reasoning) while keeping their data for the sensitivity analysis.
    """
    la, lb = lang_pair
    groups = _load_feats(dataset, mode="cot")
    drop = set(exclude_models or [])
    all_seen = {m for m, _ in groups.keys() if m not in drop}
    models = sorted({
        m for m in all_seen
        if all(groups.get((m, lang)) for lang in lang_pair)
    })
    excluded = sorted(all_seen - set(models))
    if excluded:
        print(f"[pilot_03] {dataset} {lang_pair}: dropped {len(excluded)} model(s) "
              f"missing one or both langs: {excluded}", file=sys.stderr)
    if not models:
        return None

    accuracies = {m: {} for m in models}
    for m in models:
        for lang in lang_pair:
            rows = groups.get((m, lang), [])
            accuracies[m][lang] = mean(r["correct"] for r in rows) if rows else 0.0

    acc_ranks_per_lang = {
        lang: rank_models({m: accuracies[m].get(lang, 0.0) for m in models}, higher_is_better=True)
        for lang in lang_pair
    }

    feature_values_per_lang = {lang: {f: {} for f in FEATURE_NAMES} for lang in lang_pair}
    divergences = {f: {} for f in FEATURE_NAMES}
    matched_sizes = {}

    for m in models:
        rows_by_lang = {lang: groups.get((m, lang), []) for lang in lang_pair}
        a = _per_item_correctness(rows_by_lang[la])
        b = _per_item_correctness(rows_by_lang[lb])
        matched = accuracy_matched_ids(a, b)
        matched_sizes[m] = len(matched)

        for lang in lang_pair:
            rows = [r for r in rows_by_lang[lang] if not matched or r["id"] in matched]
            for f in FEATURE_NAMES:
                vals = [r["features"][f] for r in rows]
                feature_values_per_lang[lang][f][m] = mean(vals) if vals else 0.0

        for f in FEATURE_NAMES:
            divergences[f][m] = abs(
                feature_values_per_lang[la][f][m] - feature_values_per_lang[lb][f][m]
            )

    feature_ranks_per_lang = {
        lang: {
            f: rank_models(feature_values_per_lang[lang][f], higher_is_better=True)
            for f in FEATURE_NAMES
        }
        for lang in lang_pair
    }

    def _tau(rank_a: dict, rank_b: dict) -> float:
        return kendall_tau([rank_a[m] for m in models], [rank_b[m] for m in models])

    def _inv(rank_a: dict, rank_b: dict) -> float:
        return inversion_rate([rank_a[m] for m in models], [rank_b[m] for m in models])

    # Within-language: tau(acc-L, feat-L). Named as a baseline/sanity check;
    # the *primary* instability metric is the cross-lingual family below.
    tau_within = {
        lang: {f: _tau(acc_ranks_per_lang[lang], feature_ranks_per_lang[lang][f])
               for f in FEATURE_NAMES}
        for lang in lang_pair
    }
    # Cross-lingual: tau(acc-L_a, feat-L_b) and reverse — the real test of
    # ranking comparability across languages (chapter 3.2.7 CRSI).
    key_ab = f"acc_{la}_vs_feat_{lb}"
    key_ba = f"acc_{lb}_vs_feat_{la}"
    tau_cross = {
        key_ab: {f: _tau(acc_ranks_per_lang[la], feature_ranks_per_lang[lb][f])
                  for f in FEATURE_NAMES},
        key_ba: {f: _tau(acc_ranks_per_lang[lb], feature_ranks_per_lang[la][f])
                  for f in FEATURE_NAMES},
    }
    inv_cross = {
        key_ab: {f: _inv(acc_ranks_per_lang[la], feature_ranks_per_lang[lb][f])
                  for f in FEATURE_NAMES},
        key_ba: {f: _inv(acc_ranks_per_lang[lb], feature_ranks_per_lang[la][f])
                  for f in FEATURE_NAMES},
    }
    tau_feature_cross_lingual = {
        f: _tau(feature_ranks_per_lang[la][f], feature_ranks_per_lang[lb][f])
        for f in FEATURE_NAMES
    }

    crsi_cross_lingual = mean([
        mean(tau_cross[key_ab].values()),
        mean(tau_cross[key_ba].values()),
    ])
    crsi_within_la = crsi(acc_ranks_per_lang[la], feature_ranks_per_lang[la])

    return {
        "dataset": dataset,
        "lang_pair": list(lang_pair),
        "models": models,
        "accuracy_per_lang": accuracies,
        "accuracy_ranks_per_lang": acc_ranks_per_lang,
        "feature_values_per_lang": feature_values_per_lang,
        "feature_ranks_per_lang": feature_ranks_per_lang,
        "kendall_tau_within_lang_baseline": tau_within,
        "kendall_tau_cross_lingual_primary": tau_cross,
        "inversion_rate_cross_lingual_primary": inv_cross,
        "kendall_tau_feature_cross_lingual": tau_feature_cross_lingual,
        "crsi_cross_lingual_primary": crsi_cross_lingual,
        "crsi_within_lang_baseline": crsi_within_la,
        "mean_abs_divergence_per_feature": {
            f: (mean(divergences[f].values()) if divergences[f] else 0.0)
            for f in FEATURE_NAMES
        },
        "per_model_divergence": divergences,
        "matched_subset_sizes": matched_sizes,
    }


def _lang_pairs_for(dataset: str):
    """EN-ZH always; MGSM additionally EN-JA and EN-FR when feature files exist."""
    pairs = [("en", "zh")]
    if dataset == "mgsm":
        for other in ("ja", "fr"):
            if any(FEAT_DIR.glob(f"*__mgsm__{other}__cot.jsonl")):
                pairs.append(("en", other))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--exclude_models", nargs="+", default=None,
                    help="model keys to drop before analysis (e.g. Phi-4-reasoning "
                         "for the primary 4-model run; keep all for sensitivity).")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day1_rankings.json"))
    args = ap.parse_args()

    result = {}
    for d in args.datasets:
        for pair in _lang_pairs_for(d):
            key = f"{d}__{pair[0]}_{pair[1]}"
            result[key] = analyze_dataset(d, lang_pair=pair,
                                          exclude_models=args.exclude_models)
    if args.exclude_models:
        result["_excluded_models"] = list(args.exclude_models)

    # Domain contrast is EN-ZH only.
    xkey, mkey = "xcopa__en_zh", "mgsm__en_zh"
    if result.get(xkey) and result.get(mkey):
        x = result[xkey]["mean_abs_divergence_per_feature"]
        m = result[mkey]["mean_abs_divergence_per_feature"]
        result["domain_contrast"] = {
            f: {"xcopa": x.get(f, 0.0), "mgsm": m.get(f, 0.0),
                 "ratio_xcopa_over_mgsm": (x.get(f, 0.0) / m[f]) if m.get(f, 0.0) else None}
            for f in FEATURE_NAMES
        }

    # Key test 3: MGSM multilingual low-divergence pattern — compare per-feature
    # cross-lingual divergence across (en,zh), (en,ja), (en,fr).
    mgsm_keys = [k for k in result if k.startswith("mgsm__") and result[k] is not None]
    if len(mgsm_keys) > 1:
        result["mgsm_multilingual_divergence"] = {
            f: {k: result[k]["mean_abs_divergence_per_feature"].get(f, 0.0)
                for k in mgsm_keys}
            for f in FEATURE_NAMES
        }

    write_json(Path(args.out), result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
