"""Day 4 morning: hyperedge annotation pilot — open-set per-target design.

For each target step S in a CoT trajectory, two annotators independently
identify a set of hyperedges H(S): each hyperedge is a subset of prior items
(problem premise + earlier steps) that is jointly sufficient AND minimally
necessary to derive S. Annotators arrive at H(S) via the two-question
procedure (sufficiency + per-element necessity) documented in
pilot_day4_annotation_guide.md §2.

This procedure mirrors the planned Stage-2 LLM-judge prompt in pilot_10
(Day 5+), so human ground truth and automatic-extraction output reference
the same operational definition without re-derivation. The pre-registered
hypotheses (H_main, H_sub) and falsification thresholds also live in the
guide.

Outputs (under output/reports/):
  - pilot_day4_annotation_sheet_annotator_A.csv  (block-shuffled, seed 42)
  - pilot_day4_annotation_sheet_annotator_B.csv  (block-shuffled, seed 137)
  - pilot_day4_annotation_sheet.jsonl            (per-target machine source)
  - pilot_day4_item_contexts.md                  (premise + choices + gold +
                                                   full EN/ZH trajectories)
  - pilot_day4_annotation_meta.json              (selection summary)

Helpers exposed for downstream analysis:
  parse_hyperedges_field(s)            -> frozenset of frozensets
  jaccard_pairwise(a, b)               -> [0,1] outer-set Jaccard
  compute_jaccard_agreement(la, lb)    -> dict with mean + breakdown
  permutation_null(la, lb, n_priors)   -> dict with null_mean, null_std,
                                          invalid_rate, and (if invalid_rate
                                          > 0.30) stratified_null_mean.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, REPORT_DIR  # noqa: E402
from utils.data_loader import load_xcopa  # noqa: E402
from utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402


# Fixed (not random) so re-running regenerates identical sheets and partial
# annotation work is not invalidated by a re-emit.
SEED_A = 42
SEED_B = 137


PARALLEL_PHRASES = {
    "en": ["choice a", "choice b", "option a", "option b",
           "more likely", "more plausible", "more reasonable"],
    "zh": ["选项a", "选项b", "更可能", "更合理", "更有可能", "更直接"],
}

EPISTEMIC_PHRASES = {
    "en": ["wait", "hmm", "actually", "let me", "perhaps", "maybe",
           "not sure", "uncertain", "on second thought"],
    "zh": ["等等", "再想想", "让我", "可能", "也许", "或许",
           "不确定", "重新考虑", "重新审视"],
}

CONCLUSION_PHRASES = {
    "en": ["therefore", "thus", "hence", "as a result", "more likely",
           "more plausible", "more reasonable", "the answer is"],
    "zh": ["因此", "所以", "故", "综上", "更可能", "更合理",
           "答案是", "答案为"],
}

CHOICE_EVAL_PHRASES = {
    "en": ["choice a", "choice b", "option a", "option b"],
    "zh": ["选项a", "选项b"],
}


def _contains_any(text: str, phrases: list) -> bool:
    low = text.lower()
    return any(p in low for p in phrases)


def _classify_lang_traj(steps: list, lang: str) -> tuple:
    n = len(steps)
    text = " ".join(steps)
    has_comparison = _contains_any(text, PARALLEL_PHRASES[lang])
    has_epistemic = _contains_any(text, EPISTEMIC_PHRASES[lang])
    if n <= 4:
        return "simple", f"{lang}: n={n}<=4"
    if n >= 5 and has_comparison:
        return "parallel", f"{lang}: n={n}>=5+comparison"
    if n >= 6 or has_epistemic:
        return "mixed", f"{lang}: " + ("n>=6" if n >= 6 else "epistemic")
    return "mixed", f"{lang}: n={n} no-marker fallback"


def _classify_item(en_steps: list, zh_steps: list) -> tuple:
    """Priority parallel > mixed > simple, taking the higher tier across
    EN/ZH. Either language alone hitting `parallel` makes the item parallel.
    """
    lab_en, why_en = _classify_lang_traj(en_steps, "en")
    lab_zh, why_zh = _classify_lang_traj(zh_steps, "zh")
    priority = {"parallel": 2, "mixed": 1, "simple": 0}
    final = lab_en if priority[lab_en] >= priority[lab_zh] else lab_zh
    return final, f"[{why_en}] [{why_zh}] -> {final}"


def _build(dataset, models, n_per_bucket, seed, item_ids_override):
    en_data = {it["id"]: it for it in load_xcopa("en", split="test")}
    zh_data = {it["id"]: it for it in load_xcopa("zh", split="test")}

    feat = {}
    for m in models:
        for lang in ("en", "zh"):
            feat[(m, lang)] = {
                r["id"]: r for r in read_jsonl(
                    FEAT_DIR / f"{m}__{dataset}__{lang}__cot.jsonl"
                )
            }

    matched = None
    for m in models:
        en_corr = {i: bool(r["correct"]) for i, r in feat[(m, "en")].items()}
        zh_corr = {i: bool(r["correct"]) for i, r in feat[(m, "zh")].items()}
        m_matched = accuracy_matched_ids(en_corr, zh_corr)
        matched = m_matched if matched is None else matched & m_matched

    canonical = models[0]
    classified = []
    for iid in sorted(matched):
        en_steps = feat[(canonical, "en")][iid]["steps"]
        zh_steps = feat[(canonical, "zh")][iid]["steps"]
        if len(en_steps) < 2 or len(zh_steps) < 2:
            continue
        struct, reason = _classify_item(en_steps, zh_steps)
        classified.append({
            "iid": iid, "structure_type": struct, "reason": reason,
            "en_steps": en_steps, "zh_steps": zh_steps,
        })

    if item_ids_override:
        by_id = {c["iid"]: c for c in classified}
        picks = [by_id[i] for i in item_ids_override if i in by_id]
        missing = [i for i in item_ids_override if i not in by_id]
        if missing:
            print(f"[warn] override IDs not in matched set: {missing}",
                  file=sys.stderr)
    else:
        rng = random.Random(seed)
        by_bucket = {"simple": [], "parallel": [], "mixed": []}
        for c in classified:
            by_bucket[c["structure_type"]].append(c)
        picks = []
        for bucket in ("simple", "parallel", "mixed"):
            items = by_bucket[bucket][:]
            rng.shuffle(items)
            picks.extend(items[:n_per_bucket])
            if len(items) < n_per_bucket:
                print(f"[warn] only {len(items)} items in '{bucket}' bucket "
                      f"(requested {n_per_bucket})", file=sys.stderr)

    print(f"\n=== ITEM SELECTION REVIEW ({len(picks)} items) ===")
    for c in picks:
        en_s, zh_s = c["en_steps"], c["zh_steps"]
        print(f"\n[{c['iid']}]  EN={len(en_s)} steps  ZH={len(zh_s)} steps "
              f" -> {c['structure_type'].upper()}")
        print(f"  reason: {c['reason']}")
        print(f"  EN: {en_s[0][:90]}")
        if len(en_s) > 1:
            print(f"      {en_s[1][:90]}")
        print(f"  ZH: {zh_s[0][:90]}")
        if len(zh_s) > 1:
            print(f"      {zh_s[1][:90]}")
    print()

    targets = []
    contexts = []
    for c in picks:
        iid, struct = c["iid"], c["structure_type"]
        en_orig = en_data[iid]
        zh_orig = zh_data[iid]
        ctx = {
            "item_id": iid, "structure_type": struct,
            "classification_reason": c["reason"],
            "gold": en_orig["gold"],
            "question_type": en_orig["prompt_payload"]["question"],
            "en_premise": en_orig["prompt_payload"]["premise"],
            "en_choice_a": en_orig["prompt_payload"]["choice1"],
            "en_choice_b": en_orig["prompt_payload"]["choice2"],
            "zh_premise": zh_orig["prompt_payload"]["premise"],
            "zh_choice_a": zh_orig["prompt_payload"]["choice1"],
            "zh_choice_b": zh_orig["prompt_payload"]["choice2"],
            "trajectories": {},
        }
        for m in models:
            for lang in ("en", "zh"):
                steps = feat[(m, lang)][iid]["steps"]
                ctx["trajectories"][f"{m}__{lang}"] = steps
                for tgt in range(1, len(steps)):
                    targets.append({
                        "task_id": f"{m}::{iid}",
                        "model": m, "item_id": iid,
                        "structure_type": struct, "lang": lang,
                        "target_step_num": tgt + 1,
                        "target_step": steps[tgt],
                        "prior_steps": steps[:tgt],
                        "n_prior": tgt,
                    })
        contexts.append(ctx)

    return targets, contexts, picks


def _shuffle_blocks(records, seed):
    """Group rows by (task_id, lang). Shuffle block order with `seed`; within
    a block, keep target_step_num ascending so annotators see the chain in
    natural order.
    """
    blocks = defaultdict(list)
    for r in records:
        blocks[(r["task_id"], r["lang"])].append(r)
    for k in blocks:
        blocks[k].sort(key=lambda r: r["target_step_num"])
    keys = list(blocks.keys())
    random.Random(seed).shuffle(keys)
    out = []
    for k in keys:
        out.extend(blocks[k])
    return out


CSV_HEADER_COMMENT = (
    "# Pre-registered: H_main, H_sub -- see pilot_day4_annotation_guide.md "
    "section 1. Do NOT modify the first 7 columns. Fill ONLY hyperedges, "
    "step_subunit_count, comment."
)
CSV_COLUMNS = [
    "task_id", "structure_type", "lang", "target_step_num", "target_step",
    "prior_steps_listing", "n_prior",
    "hyperedges", "step_subunit_count", "comment",
]


def _write_csv(path: Path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(CSV_HEADER_COMMENT + "\n")
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in records:
            prior_listing = "\n".join(r["prior_steps"])
            w.writerow([
                r["task_id"], r["structure_type"], r["lang"],
                r["target_step_num"], r["target_step"],
                prior_listing, r["n_prior"],
                "", "1", "",
            ])


def _write_contexts_md(path: Path, contexts, models):
    Q = {"cause": ("What was the cause?", "原因是什么？"),
         "effect": ("What happened as a result?", "结果是什么？")}
    out = ["# Day-4 hyperedge annotation: item contexts", "",
           "Read the relevant section before annotating each item. The CSV "
           "rows give only target step + prior steps (without the original "
           "problem). For premise/choices/gold, refer here.", ""]
    for ctx in contexts:
        qtype = ctx["question_type"]
        q_en, q_zh = Q.get(qtype, (qtype, qtype))
        out += [f"## {ctx['item_id']} — `{ctx['structure_type']}`  "
                f"(gold: **{ctx['gold']}**, question: {qtype})", ""]
        out += ["### English problem", "",
                f"- Premise: {ctx['en_premise']}",
                f"- Question: {q_en}",
                f"- Choice A: {ctx['en_choice_a']}",
                f"- Choice B: {ctx['en_choice_b']}", ""]
        out += ["### 中文 problem", "",
                f"- 前提: {ctx['zh_premise']}",
                f"- 问题: {q_zh}",
                f"- 选项 A: {ctx['zh_choice_a']}",
                f"- 选项 B: {ctx['zh_choice_b']}", ""]
        for m in models:
            for lang in ("en", "zh"):
                out.append(f"### Trajectory — {m} / {lang}")
                out.append("")
                for s in ctx["trajectories"].get(f"{m}__{lang}", []):
                    out.append(f"- {s}")
                out.append("")
        out += ["---", ""]
    path.write_text("\n".join(out), encoding="utf-8")


# ---------- helpers consumed by downstream Jaccard / permutation analysis ----

def parse_hyperedges_field(s) -> frozenset:
    """'1;3,4' -> frozenset({frozenset({1}), frozenset({3, 4})}).

    Empty / whitespace / '-' / None all map to empty frozenset. Malformed
    chunks (non-integer tokens) are silently dropped — we surface this in the
    audit by counting empty cells separately downstream.
    """
    if s is None:
        return frozenset()
    s = str(s).strip()
    if not s or s == "-":
        return frozenset()
    edges = set()
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            idxs = frozenset(int(x.strip()) for x in chunk.split(",") if x.strip())
        except ValueError:
            continue
        if idxs:
            edges.add(idxs)
    return frozenset(edges)


def jaccard_pairwise(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_jaccard_agreement(labels_a, labels_b) -> dict:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must align by target")
    if not labels_a:
        return {"mean_jaccard": 1.0, "n_targets": 0,
                "n_both_empty": 0, "n_both_nonempty": 0, "n_one_empty": 0}
    pairs = [jaccard_pairwise(a, b) for a, b in zip(labels_a, labels_b)]
    return {
        "mean_jaccard": sum(pairs) / len(pairs),
        "n_targets": len(pairs),
        "n_both_empty": sum(1 for a, b in zip(labels_a, labels_b)
                            if not a and not b),
        "n_both_nonempty": sum(1 for a, b in zip(labels_a, labels_b)
                               if a and b),
        "n_one_empty": sum(1 for a, b in zip(labels_a, labels_b)
                           if bool(a) ^ bool(b)),
    }


def _label_in_range(label: frozenset, max_idx: int) -> bool:
    for edge in label:
        for idx in edge:
            if idx < 1 or idx > max_idx:
                return False
    return True


def permutation_null(labels_a, labels_b, n_priors_per_target,
                     n_perm=1000, seed=0) -> dict:
    """Method-(a) shuffle of B's labels across targets; report invalid_rate
    (post-shuffle labels referencing out-of-range indices for the new
    target). Stratified shuffle within n_prior buckets is additionally
    reported when invalid_rate > 0.30 — there post-shuffle invalidity is 0
    by construction.
    """
    n = len(labels_a)
    if len(labels_b) != n or len(n_priors_per_target) != n:
        raise ValueError("inputs must align by target")
    if n == 0 or n_perm == 0:
        return {"null_mean": 0.0, "null_std": 0.0,
                "invalid_rate": 0.0, "n_perm": n_perm}
    rng = random.Random(seed)

    null_means = []
    invalid_count = 0
    nonempty_count = 0
    for _ in range(n_perm):
        shuf = list(labels_b)
        rng.shuffle(shuf)
        agrees = []
        for i in range(n):
            b = shuf[i]
            if b:
                nonempty_count += 1
                if not _label_in_range(b, n_priors_per_target[i]):
                    invalid_count += 1
            agrees.append(jaccard_pairwise(labels_a[i], b))
        null_means.append(sum(agrees) / n)

    null_mean = sum(null_means) / n_perm
    null_std = (sum((m - null_mean) ** 2 for m in null_means) / n_perm) ** 0.5
    invalid_rate = invalid_count / max(nonempty_count, 1)

    out = {"null_mean": null_mean, "null_std": null_std,
           "invalid_rate": invalid_rate, "n_perm": n_perm}

    if invalid_rate > 0.30:
        rng2 = random.Random(seed + 1)
        buckets = defaultdict(list)
        for i, np_i in enumerate(n_priors_per_target):
            buckets[np_i].append(i)
        strat_means = []
        for _ in range(n_perm):
            shuf = list(labels_b)
            for idxs in buckets.values():
                bl = [labels_b[i] for i in idxs]
                rng2.shuffle(bl)
                for j, idx in enumerate(idxs):
                    shuf[idx] = bl[j]
            agrees = [jaccard_pairwise(labels_a[i], shuf[i]) for i in range(n)]
            strat_means.append(sum(agrees) / n)
        out["stratified_null_mean"] = sum(strat_means) / n_perm

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-8b", "llama3.1-8b"])
    ap.add_argument("--n_per_bucket", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11,
                    help="seed for within-bucket sampling (not for shuffle)")
    ap.add_argument("--item_ids", nargs="*", default=None,
                    help="manual override; bypasses auto-classification "
                         "(escape hatch only — paper procedure relies on "
                         "auto-classification for reproducibility)")
    ap.add_argument("--out_dir", default=str(REPORT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets, contexts, picks = _build(
        args.dataset, args.models, args.n_per_bucket, args.seed,
        args.item_ids,
    )

    write_jsonl(out_dir / "pilot_day4_annotation_sheet.jsonl", targets)
    a_path = out_dir / "pilot_day4_annotation_sheet_annotator_A.csv"
    b_path = out_dir / "pilot_day4_annotation_sheet_annotator_B.csv"
    _write_csv(a_path, _shuffle_blocks(targets, SEED_A))
    _write_csv(b_path, _shuffle_blocks(targets, SEED_B))
    _write_contexts_md(out_dir / "pilot_day4_item_contexts.md",
                       contexts, args.models)

    write_json(out_dir / "pilot_day4_annotation_meta.json", {
        "n_picks": len(picks),
        "models": args.models,
        "dataset": args.dataset,
        "structure_breakdown": {
            s: sum(1 for c in picks if c["structure_type"] == s)
            for s in ("simple", "parallel", "mixed")
        },
        "n_target_rows": len(targets),
        "seed_a": SEED_A, "seed_b": SEED_B,
        "sampling_seed": args.seed,
        "item_ids": [c["iid"] for c in picks],
    })

    print(f"wrote: {a_path}")
    print(f"wrote: {b_path}")
    print(f"wrote: {out_dir / 'pilot_day4_annotation_sheet.jsonl'}")
    print(f"wrote: {out_dir / 'pilot_day4_item_contexts.md'}")
    print(f"\ntotal target rows per annotator: {len(targets)}")


if __name__ == "__main__":
    main()
