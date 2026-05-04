"""Day 4 evening: compute hyperedge Jaccard agreement on filled annotator
sheets. Joins A and B by (task_id, lang, target_step_num), reports:

  - Overall mean Jaccard + by-lang / by-structure_type / by-model breakdowns
  - Permutation null (with stratified fallback if invalid_rate > 30%)
  - H_main test: pooled fraction of arity-≥2 hyperedges (threshold 0.25)
  - H_sub test: avg arity (EN vs ZH) and avg DAG depth (EN vs ZH)
  - step_subunit_count divergence by lang
  - Disagreement target listing (raw, for adjudication review)

Encoding-robust: tries utf-8-sig first, falls back to gbk (some Excel
locales save Chinese CSVs as gbk).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_09_manual_annotation import (  # noqa: E402
    compute_jaccard_agreement,
    jaccard_pairwise,
    parse_hyperedges_field,
    permutation_null,
)
from utils.config import REPORT_DIR  # noqa: E402
from utils.io import write_json  # noqa: E402


def _read_csv_robust(path: Path) -> tuple:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            text = raw.decode(enc)
            return text, enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"could not decode {path}")


def _parse_sheet(path: Path) -> list:
    text, enc = _read_csv_robust(path)
    buf = io.StringIO(text)
    # Skip leading metadata lines (one or more), where a metadata line is
    # any line whose first non-whitespace, non-quote character is '#' or
    # which is empty. Excel may re-save the inline `# Pre-registered ...`
    # comment row wrapped in quotes, so we strip leading quotes too.
    while True:
        pos = buf.tell()
        line = buf.readline()
        if not line:
            break
        probe = line.lstrip().lstrip('"').lstrip()
        if probe.startswith("#") or not probe.strip():
            continue
        buf.seek(pos)
        break
    rows = list(csv.DictReader(buf))
    print(f"[load] {path.name}: enc={enc}  data_rows={len(rows)}")
    return rows


def _key(r: dict) -> tuple:
    return (r["task_id"], r["lang"], int(r["target_step_num"]))


def _join(a_rows: list, b_rows: list) -> list:
    a_map = {_key(r): r for r in a_rows}
    b_map = {_key(r): r for r in b_rows}
    shared = sorted(set(a_map) & set(b_map))
    only_a = set(a_map) - set(b_map)
    only_b = set(b_map) - set(a_map)
    if only_a or only_b:
        print(f"[warn] only-A={len(only_a)}  only-B={len(only_b)}", file=sys.stderr)
    out = []
    for k in shared:
        a, b = a_map[k], b_map[k]
        out.append({
            "task_id": a["task_id"],
            "model": a["task_id"].split("::")[0],
            "structure_type": a["structure_type"],
            "lang": a["lang"],
            "target_step_num": int(a["target_step_num"]),
            "n_prior": int(a["n_prior"]),
            "label_a": parse_hyperedges_field(a.get("hyperedges", "")),
            "label_b": parse_hyperedges_field(b.get("hyperedges", "")),
            "subunit_a": int(a.get("step_subunit_count") or 1),
            "subunit_b": int(b.get("step_subunit_count") or 1),
            "comment_a": a.get("comment", ""),
            "comment_b": b.get("comment", ""),
        })
    return out


def _agreement_breakdown(joined: list, group_key) -> dict:
    by = defaultdict(list)
    for r in joined:
        by[group_key(r)].append((r["label_a"], r["label_b"]))
    out = {}
    for k, pairs in sorted(by.items()):
        la = [p[0] for p in pairs]
        lb = [p[1] for p in pairs]
        out[str(k)] = compute_jaccard_agreement(la, lb)
    return out


def _arity_fraction(joined: list, threshold: int = 2) -> dict:
    """H_main: fraction of all annotated hyperedges (across both annotators
    and all targets) with arity >= threshold. Threshold default 2."""
    total = 0
    multi = 0
    for r in joined:
        for label in (r["label_a"], r["label_b"]):
            for edge in label:
                total += 1
                if len(edge) >= threshold:
                    multi += 1
    return {
        "total_hyperedges": total,
        "n_arity_ge_2": multi,
        "fraction_arity_ge_2": (multi / total) if total else 0.0,
        "h_main_threshold": 0.25,
        "h_main_pass": (multi / total) >= 0.25 if total else False,
    }


def _avg_arity(joined: list) -> dict:
    """Avg arity per annotated hyperedge, by lang. (Empty labels excluded.)"""
    arities = defaultdict(list)
    for r in joined:
        for tag, label in (("a", r["label_a"]), ("b", r["label_b"])):
            for edge in label:
                arities[r["lang"]].append(len(edge))
    return {
        lang: {
            "n_hyperedges": len(xs),
            "avg_arity": sum(xs) / len(xs) if xs else 0.0,
        }
        for lang, xs in sorted(arities.items())
    }


def _dag_depth_per_trajectory(joined: list, annotator: str) -> dict:
    """For each (task_id, lang) trajectory, compute DAG depth using one
    annotator's hyperedges. Depth(step 1) = 0; Depth(t) = max(depth of any
    prior in any hyperedge) + 1. Targets without an annotated hyperedge fall
    back to depth(t-1) + 1 (linear assumption).
    """
    by_traj = defaultdict(list)
    for r in joined:
        by_traj[(r["task_id"], r["lang"])].append(r)

    out = {}
    label_field = f"label_{annotator}"
    for traj_key, rows in by_traj.items():
        rows = sorted(rows, key=lambda x: x["target_step_num"])
        depth = {1: 0}
        for r in rows:
            t = r["target_step_num"]
            edges = r[label_field]
            if not edges:
                depth[t] = depth.get(t - 1, t - 2) + 1
                continue
            max_p = 0
            for edge in edges:
                for p in edge:
                    max_p = max(max_p, depth.get(p, p - 1))
            depth[t] = max_p + 1
        out[traj_key] = max(depth.values()) if depth else 0
    return out


def _avg_depth_by_lang(joined: list) -> dict:
    """Average DAG depth (averaged over A and B) per language."""
    da = _dag_depth_per_trajectory(joined, "a")
    db = _dag_depth_per_trajectory(joined, "b")
    by_lang = defaultdict(list)
    for traj_key in da:
        lang = traj_key[1]
        by_lang[lang].append((da[traj_key] + db[traj_key]) / 2)
    return {
        lang: {"n_trajectories": len(xs),
               "avg_depth": sum(xs) / len(xs) if xs else 0.0}
        for lang, xs in sorted(by_lang.items())
    }


def _h_sub_test(arity: dict, depth: dict) -> dict:
    en_arity = arity.get("en", {}).get("avg_arity", 0.0)
    zh_arity = arity.get("zh", {}).get("avg_arity", 0.0)
    en_depth = depth.get("en", {}).get("avg_depth", 0.0)
    zh_depth = depth.get("zh", {}).get("avg_depth", 0.0)
    arity_diff = en_arity - zh_arity
    depth_diff = zh_depth - en_depth
    arity_supports = arity_diff >= 0.3
    depth_supports = depth_diff >= 0.3
    return {
        "en_avg_arity": en_arity, "zh_avg_arity": zh_arity,
        "arity_diff_(en-zh)": arity_diff,
        "arity_supports_h_sub": arity_supports,
        "en_avg_depth": en_depth, "zh_avg_depth": zh_depth,
        "depth_diff_(zh-en)": depth_diff,
        "depth_supports_h_sub": depth_supports,
        "h_sub_falsification_threshold": 0.3,
        "h_sub_pass": arity_supports and depth_supports,
    }


def _subunit_divergence(joined: list) -> dict:
    """% of targets flagged as multi-unit (count > 1) by either annotator,
    by lang."""
    by_lang_total = defaultdict(int)
    by_lang_flagged = defaultdict(int)
    for r in joined:
        by_lang_total[r["lang"]] += 1
        if r["subunit_a"] > 1 or r["subunit_b"] > 1:
            by_lang_flagged[r["lang"]] += 1
    return {
        lang: {
            "n_targets": by_lang_total[lang],
            "n_flagged_multiunit": by_lang_flagged[lang],
            "fraction_flagged": (
                by_lang_flagged[lang] / by_lang_total[lang]
                if by_lang_total[lang] else 0.0
            ),
        }
        for lang in sorted(by_lang_total)
    }


def _disagreement_rows(joined: list, j_threshold: float = 0.5) -> list:
    out = []
    for r in joined:
        j = jaccard_pairwise(r["label_a"], r["label_b"])
        if j < j_threshold:
            out.append({
                "task_id": r["task_id"],
                "lang": r["lang"],
                "target_step_num": r["target_step_num"],
                "structure_type": r["structure_type"],
                "label_a": [sorted(e) for e in r["label_a"]],
                "label_b": [sorted(e) for e in r["label_b"]],
                "jaccard": j,
                "comment_a": r["comment_a"],
                "comment_b": r["comment_b"],
            })
    out.sort(key=lambda x: x["jaccard"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotator_a",
        default=str(REPORT_DIR / "pilot_day4_annotation_sheet_annotator_A_annotated.csv"),
    )
    ap.add_argument(
        "--annotator_b",
        default=str(REPORT_DIR / "pilot_day4_annotation_sheet_annotator_B_annotated.csv"),
    )
    ap.add_argument("--out", default=str(REPORT_DIR / "pilot_day4_agreement.json"))
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()

    a_rows = _parse_sheet(Path(args.annotator_a))
    b_rows = _parse_sheet(Path(args.annotator_b))
    joined = _join(a_rows, b_rows)
    print(f"[join] {len(joined)} aligned target rows\n")

    overall = compute_jaccard_agreement(
        [r["label_a"] for r in joined],
        [r["label_b"] for r in joined],
    )
    by_lang = _agreement_breakdown(joined, lambda r: r["lang"])
    by_struct = _agreement_breakdown(joined, lambda r: r["structure_type"])
    by_model = _agreement_breakdown(joined, lambda r: r["model"])

    null = permutation_null(
        [r["label_a"] for r in joined],
        [r["label_b"] for r in joined],
        [r["n_prior"] for r in joined],
        n_perm=args.n_perm, seed=0,
    )

    h_main = _arity_fraction(joined)
    arity = _avg_arity(joined)
    depth = _avg_depth_by_lang(joined)
    h_sub = _h_sub_test(arity, depth)
    subunit = _subunit_divergence(joined)
    disagreements = _disagreement_rows(joined, j_threshold=0.5)

    report = {
        "n_aligned_targets": len(joined),
        "overall": overall,
        "permutation_null": null,
        "by_lang": by_lang,
        "by_structure_type": by_struct,
        "by_model": by_model,
        "h_main": h_main,
        "h_sub": h_sub,
        "avg_arity_by_lang": arity,
        "avg_depth_by_lang": depth,
        "subunit_divergence": subunit,
        "n_disagreements_below_0.5": len(disagreements),
        "disagreements": disagreements,
    }
    write_json(Path(args.out), report)

    # Compact human-readable summary
    print("=" * 60)
    print("HEADLINE")
    print("=" * 60)
    print(f"  Mean Jaccard agreement : {overall['mean_jaccard']:.3f}  "
          f"(n={overall['n_targets']})")
    print(f"  Permutation null mean  : {null['null_mean']:.3f} "
          f"± {null['null_std']:.3f}")
    print(f"  Null invalid_rate      : {null['invalid_rate']:.1%}")
    if "stratified_null_mean" in null:
        print(f"  Stratified null mean   : {null['stratified_null_mean']:.3f}  "
              "(invalid_rate exceeded 30%)")
    gate = overall["mean_jaccard"] >= 0.6
    print(f"  Gate-1 (>= 0.6)        : {'PASS' if gate else 'FAIL'}")
    print()
    print("BY LANGUAGE")
    for lang, m in by_lang.items():
        print(f"  {lang}: mean={m['mean_jaccard']:.3f}  n={m['n_targets']}  "
              f"(both_empty={m['n_both_empty']}  "
              f"both_nonempty={m['n_both_nonempty']}  "
              f"one_empty={m['n_one_empty']})")
    print()
    print("BY STRUCTURE")
    for s, m in by_struct.items():
        print(f"  {s}: mean={m['mean_jaccard']:.3f}  n={m['n_targets']}")
    print()
    print("BY MODEL")
    for m_, mm in by_model.items():
        print(f"  {m_}: mean={mm['mean_jaccard']:.3f}  n={mm['n_targets']}")
    print()
    print("H_main (fraction of arity >= 2 hyperedges)")
    print(f"  total annotated hyperedges : {h_main['total_hyperedges']}")
    print(f"  arity >= 2                 : {h_main['n_arity_ge_2']}  "
          f"({h_main['fraction_arity_ge_2']:.1%})")
    print(f"  threshold 25%, verdict     : "
          f"{'PASS' if h_main['h_main_pass'] else 'FAIL'}")
    print()
    print("H_sub (EN wider-shallower than ZH)")
    print(f"  avg arity   en={h_sub['en_avg_arity']:.2f}  "
          f"zh={h_sub['zh_avg_arity']:.2f}  "
          f"diff(en-zh)={h_sub['arity_diff_(en-zh)']:+.2f}")
    print(f"  avg depth   en={h_sub['en_avg_depth']:.2f}  "
          f"zh={h_sub['zh_avg_depth']:.2f}  "
          f"diff(zh-en)={h_sub['depth_diff_(zh-en)']:+.2f}")
    print(f"  threshold 0.3, verdict     : "
          f"{'PASS' if h_sub['h_sub_pass'] else 'FAIL'}  "
          f"(arity_supports={h_sub['arity_supports_h_sub']}  "
          f"depth_supports={h_sub['depth_supports_h_sub']})")
    print()
    print("step_subunit_count multi-unit flag rate by lang")
    for lang, s in subunit.items():
        print(f"  {lang}: {s['n_flagged_multiunit']}/{s['n_targets']} = "
              f"{s['fraction_flagged']:.1%}")
    print()
    print(f"Disagreements at Jaccard < 0.5 : {len(disagreements)} targets")
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
