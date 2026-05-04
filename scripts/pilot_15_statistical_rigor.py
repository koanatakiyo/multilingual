"""Statistical rigor for the 10-model main experiment (chapter 4 §4.2.2).

Three layers of statistical rigor on top of the per-feature ranking analysis:

1. Item-level bootstrap CI on inversion rate. For each (dataset, lang_pair,
   feature), resample items within each model's matched subset 1000x and
   recompute inversion rates. Report 95% CI.

2. Permutation test on language labels. Shuffle (en, zh) labels within each
   item across all models and recompute Kendall τ between accuracy and feature
   rankings. Compares observed τ to the null distribution. p < 0.05 = the
   observed cross-lingual instability is unlikely under random language
   assignment.

3. Benjamini–Hochberg FDR correction across the multiple feature × benchmark
   × language-pair tests.

Also computes the **Pearson correlation** between per-model EN-ZH accuracy
gap and per-model unmatched-step-ratio (the C3 audit-protocol headline,
"pilot 0.66"), confirmed at the 10-model scale.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, RANK_DIR  # noqa: E402
from utils.io import read_jsonl, write_json  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402
from utils.ranking import inversion_rate, kendall_tau, rank_models  # noqa: E402

FEATURE_NAMES = ["step_count", "procedural_rate", "epistemic_rate", "dependency_depth"]


def _load_per_model(dataset: str, lang_pair):
    """For each model with both langs, return:
       - full IDs (every item that has both langs in features)
       - matched IDs (subset where the model is correct in BOTH langs)
       - per-lang row lookup tables.

    Accuracy must be computed on the full set so it varies across models
    (matched-only accuracy is degenerate at 1.0 by definition); features
    must be computed on the matched subset so cross-lingual feature
    comparison is conditioned on equal accuracy.
    """
    by_model = {}
    la, lb = lang_pair
    for p in sorted(FEAT_DIR.glob(f"*__{dataset}__{la}__cot.jsonl")):
        m = p.stem.split("__")[0]
        zh_path = FEAT_DIR / f"{m}__{dataset}__{lb}__cot.jsonl"
        if not zh_path.exists():
            continue
        a = {r["id"]: r for r in read_jsonl(p)}
        b = {r["id"]: r for r in read_jsonl(zh_path)}
        common = sorted(set(a) & set(b))
        matched = sorted(accuracy_matched_ids(
            {i: bool(a[i]["correct"]) for i in common},
            {i: bool(b[i]["correct"]) for i in common},
        ))
        by_model[m] = {
            "full_ids": common,
            "matched_ids": matched,
            "matched_set": set(matched),
            la: a,
            lb: b,
        }
    return by_model


def _accuracy_on(by_model, lang, ids_per_model):
    """Mean correct over the supplied IDs per model — uses full set."""
    out = {}
    for m, data in by_model.items():
        rows = [data[lang][i] for i in ids_per_model[m] if i in data[lang]]
        out[m] = mean(r["correct"] for r in rows) if rows else 0.0
    return out


def _features_on(by_model, lang, ids_per_model, restrict_to_matched=True):
    """Mean of each feature per model. If restrict_to_matched, sampled IDs
    are filtered to the model's matched_set before averaging."""
    out = {f: {} for f in FEATURE_NAMES}
    for m, data in by_model.items():
        ids = ids_per_model[m]
        if restrict_to_matched:
            ids = [i for i in ids if i in data["matched_set"]]
        rows = [data[lang][i] for i in ids if i in data[lang]]
        for f in FEATURE_NAMES:
            vals = [r["features"][f] for r in rows]
            out[f][m] = mean(vals) if vals else 0.0
    return out


def bootstrap_inversion_ci(by_model, lang_pair, n_iter=1000, seed=0):
    """Bootstrap CI on cross-lingual inversion rate. For each iteration,
    resample item IDs per model with replacement from the model's full set;
    compute per-model accuracy on the resampled set (full) and per-model
    feature mean on the resampled-and-matched subset; rank models; compute
    inversion rate of (acc_la, feat_lb) ranks."""
    rng = random.Random(seed)
    la, lb = lang_pair
    full_ids = {m: data["full_ids"] for m, data in by_model.items() if data["full_ids"]}
    models = sorted(full_ids)
    n_models = len(models)

    feat_inv_samples = {f: [] for f in FEATURE_NAMES}
    for _ in range(n_iter):
        sampled = {m: [rng.choice(full_ids[m]) for _ in range(len(full_ids[m]))]
                   for m in models}
        acc_a = _accuracy_on(by_model, la, sampled)
        feat_b = _features_on(by_model, lb, sampled, restrict_to_matched=True)
        acc_a_ranks = rank_models(acc_a, higher_is_better=True)
        for f in FEATURE_NAMES:
            feat_b_ranks = rank_models(feat_b[f], higher_is_better=True)
            inv = inversion_rate([acc_a_ranks[m] for m in models],
                                 [feat_b_ranks[m] for m in models])
            feat_inv_samples[f].append(inv)

    out = {}
    for f, samples in feat_inv_samples.items():
        samples = sorted(samples)
        out[f] = {
            "mean": mean(samples),
            "ci_2.5": samples[int(0.025 * n_iter)],
            "ci_97.5": samples[int(0.975 * n_iter)],
            "n_iter": n_iter,
            "n_model_pairs": n_models * (n_models - 1) // 2,
        }
    return out


def permutation_test_language_labels(by_model, lang_pair, n_perm=1000, seed=1):
    """Null hypothesis: the (la, lb) language assignment per item is
    exchangeable. For each permutation, with probability 0.5 swap (la, lb)
    for each item-id across all models, recompute accuracy (on full set)
    and feature (on matched subset) per model, recompute Kendall τ between
    accuracy-la rank and feature-lb rank.

    p-value (two-sided) = fraction of permuted |τ| >= |observed τ|.
    Small p means the observed cross-lingual rank correlation is unlikely
    under random language assignment.
    """
    rng = random.Random(seed)
    la, lb = lang_pair
    full_ids = {m: data["full_ids"] for m, data in by_model.items() if data["full_ids"]}
    models = sorted(full_ids)

    # Observed τ on full set for accuracy, matched set for features.
    acc_a = _accuracy_on(by_model, la, full_ids)
    feat_b = _features_on(by_model, lb, full_ids, restrict_to_matched=True)
    acc_a_ranks = rank_models(acc_a, higher_is_better=True)
    observed_tau = {}
    for f in FEATURE_NAMES:
        feat_b_ranks = rank_models(feat_b[f], higher_is_better=True)
        observed_tau[f] = kendall_tau([acc_a_ranks[m] for m in models],
                                      [feat_b_ranks[m] for m in models])

    null_tau_samples = {f: [] for f in FEATURE_NAMES}
    all_ids = sorted({i for m in models for i in full_ids[m]})

    for _ in range(n_perm):
        flip_set = {i for i in all_ids if rng.random() < 0.5}
        # For each model: swap la/lb per item according to flip_set.
        acc_perm = {}
        feat_perm = {f: {} for f in FEATURE_NAMES}
        for m in models:
            data = by_model[m]
            ids = full_ids[m]
            # Accuracy uses full set (rows_a).
            rows_a = [data[lb][i] if i in flip_set else data[la][i] for i in ids]
            acc_perm[m] = mean(r["correct"] for r in rows_a) if rows_a else 0.0
            # Features use matched subset under the permuted labels.
            ids_m = [i for i in ids if i in data["matched_set"]]
            rows_b = [data[la][i] if i in flip_set else data[lb][i] for i in ids_m]
            for f in FEATURE_NAMES:
                vals = [r["features"][f] for r in rows_b]
                feat_perm[f][m] = mean(vals) if vals else 0.0
        acc_ranks = rank_models(acc_perm, higher_is_better=True)
        for f in FEATURE_NAMES:
            f_ranks = rank_models(feat_perm[f], higher_is_better=True)
            null_tau_samples[f].append(
                kendall_tau([acc_ranks[m] for m in models],
                            [f_ranks[m] for m in models])
            )

    out = {}
    for f in FEATURE_NAMES:
        ns = null_tau_samples[f]
        obs = observed_tau[f]
        p = sum(1 for t in ns if abs(t) >= abs(obs)) / max(n_perm, 1)
        out[f] = {
            "observed_tau": obs,
            "null_mean": mean(ns) if ns else 0.0,
            "null_p_two_sided": p,
            "n_perm": n_perm,
        }
    return out


def permutation_test_model_pairing(by_model, lang_pair, n_perm=1000, seed=1):
    """Null hypothesis: cross-lingual model identity pairing is arbitrary.

    Unlike permutation_test_language_labels(), this keeps each model's observed
    per-language item statistics intact and shuffles the mapping between the
    accuracy-language model ranking and the feature-language model ranking.
    It tests whether the observed cross-lingual rank association depends on
    pairing the same model identity across languages.
    """
    rng = random.Random(seed)
    la, lb = lang_pair
    full_ids = {m: data["full_ids"] for m, data in by_model.items() if data["full_ids"]}
    models = sorted(full_ids)

    acc_a = _accuracy_on(by_model, la, full_ids)
    feat_b = _features_on(by_model, lb, full_ids, restrict_to_matched=True)
    acc_a_ranks = rank_models(acc_a, higher_is_better=True)

    observed_tau = {}
    feature_ranks = {}
    for f in FEATURE_NAMES:
        feature_ranks[f] = rank_models(feat_b[f], higher_is_better=True)
        observed_tau[f] = kendall_tau(
            [acc_a_ranks[m] for m in models],
            [feature_ranks[f][m] for m in models],
        )

    null_tau_samples = {f: [] for f in FEATURE_NAMES}
    for _ in range(n_perm):
        permuted = models[:]
        rng.shuffle(permuted)
        for f in FEATURE_NAMES:
            null_tau_samples[f].append(
                kendall_tau(
                    [acc_a_ranks[m] for m in models],
                    [feature_ranks[f][pm] for pm in permuted],
                )
            )

    out = {}
    for f in FEATURE_NAMES:
        ns = null_tau_samples[f]
        obs = observed_tau[f]
        out[f] = {
            "observed_tau": obs,
            "null_mean": mean(ns) if ns else 0.0,
            "null_p_two_sided": sum(1 for t in ns if abs(t) >= abs(obs)) / max(n_perm, 1),
            "n_perm": n_perm,
        }
    return out


def benjamini_hochberg(p_values: dict, alpha: float = 0.05) -> dict:
    """BH FDR correction. Input: dict {test_id: p_value}. Returns
    {test_id: {p, p_adj, reject_at_alpha}}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}
    # Compute adjusted p-values
    p_adj = [None] * n
    cumulative_min = 1.0
    for rev_i in range(n - 1, -1, -1):
        rank = rev_i + 1  # 1-based rank from smallest
        raw = items[rev_i][1]
        adj = min(raw * n / rank, 1.0)
        cumulative_min = min(cumulative_min, adj)
        p_adj[rev_i] = cumulative_min
    out = {}
    for (k, raw), adj in zip(items, p_adj):
        out[k] = {"p": raw, "p_adj": adj, "reject_at_alpha": adj < alpha}
    return out


def pearson_acc_gap_vs_unmatched(dataset: str = "xcopa") -> dict:
    """C3 headline statistic: Pearson(per-model EN-ZH accuracy gap,
    per-model LaBSE unmatched-step ratio). Reads pilot_08's output."""
    p = RANK_DIR / "pilot_day3_alignment.json"
    if not p.exists():
        return {"error": f"{p} not found — run pilot_08 first"}
    d = json.load(open(p)).get(dataset)
    if not d:
        return {"error": f"no {dataset} in pilot_day3_alignment.json"}
    acc_en = d["accuracy_en_per_model"]
    acc_zh = d["accuracy_zh_per_model"]
    unmatched = d["unmatched_mean_per_model"]
    models = sorted(set(acc_en) & set(acc_zh) & set(unmatched))
    if len(models) < 3:
        return {"error": f"too few models ({len(models)})"}
    gaps = [acc_en[m] - acc_zh[m] for m in models]
    unm = [unmatched[m] for m in models]
    n = len(models)
    mx, my = sum(gaps) / n, sum(unm) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(gaps, unm))
    sx = math.sqrt(sum((x - mx) ** 2 for x in gaps))
    sy = math.sqrt(sum((y - my) ** 2 for y in unm))
    r = cov / (sx * sy) if sx and sy else 0.0
    return {
        "dataset": dataset, "n_models": n, "pearson_r": r,
        "models": models,
        "accuracy_gaps": dict(zip(models, gaps)),
        "unmatched_ratios": dict(zip(models, unm)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["xcopa", "mgsm"])
    ap.add_argument("--lang_pair", nargs=2, default=["en", "zh"])
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--permutation_mode",
                    choices=["item_label", "model_pair", "both"],
                    default="item_label",
                    help="item_label reproduces the original within-item "
                         "language-label shuffle; model_pair shuffles model "
                         "identity pairing across languages; both reports both.")
    ap.add_argument("--fdr_permutation_family",
                    choices=["item_label", "model_pair"],
                    default="item_label",
                    help="Which permutation family supplies p-values for "
                         "the top-level BH-FDR correction when --permutation_mode=both.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day3_statistical_rigor.json"))
    args = ap.parse_args()

    result = {}
    p_pool = {}  # for FDR across feature × benchmark
    for d in args.datasets:
        by_model = _load_per_model(d, tuple(args.lang_pair))
        if len(by_model) < 3:
            print(f"[skip] {d}: only {len(by_model)} models with both langs",
                  file=sys.stderr)
            continue
        boot = bootstrap_inversion_ci(by_model, tuple(args.lang_pair),
                                      n_iter=args.n_bootstrap, seed=args.seed)
        perm_item = None
        perm_model = None
        if args.permutation_mode in ("item_label", "both"):
            perm_item = permutation_test_language_labels(by_model, tuple(args.lang_pair),
                                                         n_perm=args.n_perm,
                                                         seed=args.seed + 1)
        if args.permutation_mode in ("model_pair", "both"):
            perm_model = permutation_test_model_pairing(by_model, tuple(args.lang_pair),
                                                        n_perm=args.n_perm,
                                                        seed=args.seed + 1)

        result[d] = {
            "n_models": len(by_model),
            "models": sorted(by_model),
            "bootstrap_inversion_ci": boot,
            "permutation_mode": args.permutation_mode,
        }
        if perm_item is not None:
            result[d]["permutation_language_labels"] = perm_item
        if perm_model is not None:
            result[d]["permutation_model_pairing"] = perm_model

        if args.fdr_permutation_family == "model_pair" and perm_model is not None:
            fdr_source = perm_model
        elif args.fdr_permutation_family == "item_label" and perm_item is not None:
            fdr_source = perm_item
        elif perm_model is not None:
            fdr_source = perm_model
        else:
            fdr_source = perm_item or {}

        for f, stats in fdr_source.items():
            p_pool[f"{d}::{f}"] = stats["null_p_two_sided"]

    # Benjamini-Hochberg FDR across all feature × benchmark tests
    if p_pool:
        result["bh_fdr"] = {
            "alpha": args.alpha,
            "n_tests": len(p_pool),
            "per_test": benjamini_hochberg(p_pool, alpha=args.alpha),
        }

    # C3 headline: Pearson(accuracy gap, unmatched ratio)
    result["pearson_acc_gap_vs_unmatched"] = {
        d: pearson_acc_gap_vs_unmatched(d) for d in args.datasets
    }

    write_json(Path(args.out), result)
    print(f"wrote {args.out}")
    # Compact stdout summary
    for d in args.datasets:
        if d not in result:
            continue
        print(f"\n=== {d} ===")
        print(f"  n_models: {result[d]['n_models']}")
        print(f"  bootstrap inversion 95% CI per feature:")
        for f, s in result[d]["bootstrap_inversion_ci"].items():
            print(f"    {f:<22}  mean {s['mean']:.3f}  CI [{s['ci_2.5']:.3f}, {s['ci_97.5']:.3f}]")
        if "permutation_language_labels" in result[d]:
            print(f"  permutation test (language labels shuffled):")
            for f, s in result[d]["permutation_language_labels"].items():
                sig = "***" if s["null_p_two_sided"] < 0.001 else "**" if s["null_p_two_sided"] < 0.01 else "*" if s["null_p_two_sided"] < 0.05 else " "
                print(f"    {f:<22}  obs τ {s['observed_tau']:+.3f}  null mean {s['null_mean']:+.3f}  p={s['null_p_two_sided']:.3f} {sig}")
        if "permutation_model_pairing" in result[d]:
            print(f"  permutation test (model pairing shuffled):")
            for f, s in result[d]["permutation_model_pairing"].items():
                sig = "***" if s["null_p_two_sided"] < 0.001 else "**" if s["null_p_two_sided"] < 0.01 else "*" if s["null_p_two_sided"] < 0.05 else " "
                print(f"    {f:<22}  obs τ {s['observed_tau']:+.3f}  null mean {s['null_mean']:+.3f}  p={s['null_p_two_sided']:.3f} {sig}")
    print(f"\n=== Pearson(EN-ZH accuracy gap, LaBSE unmatched ratio) ===")
    for d, r in result["pearson_acc_gap_vs_unmatched"].items():
        if "error" in r:
            print(f"  {d}: {r['error']}")
        else:
            print(f"  {d}: r = {r['pearson_r']:+.3f}  (n={r['n_models']})")


if __name__ == "__main__":
    main()
