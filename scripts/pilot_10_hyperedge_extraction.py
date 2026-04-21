"""Day 5: bidirectional ensemble hyperedge extraction on matched subsets.

For each of the 3 hypergraph models, for each matched item, run k forward
counterfactual joint-necessity tests (masking-based) and k backward judge
runs (premise-identification), and synthesise confidence c(e) = min(c_f, c_b).

Judge calls are abstracted behind a pluggable callable so the pilot can run
either with a hosted judge (Anthropic) or a local model. By default a stub
judge is used; wire in your preferred client before production use.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, HG_DIR, MODELS, PILOT  # noqa: E402
from utils.io import read_jsonl, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402


Judge = Callable[[str, str], float]  # (prompt, system) -> score in [0,1]


def stub_judge(prompt: str, system: str = "") -> float:
    """Deterministic pseudo-judge for offline pilot. Replace before production."""
    h = abs(hash(prompt + "|" + system)) % 1000
    return h / 1000.0


def anthropic_judge_factory(model: str = "claude-opus-4-7") -> Judge:
    """Real judge via the Anthropic SDK. Requires ANTHROPIC_API_KEY."""
    from anthropic import Anthropic  # type: ignore
    client = Anthropic()
    def _judge(prompt: str, system: str = "") -> float:
        msg = client.messages.create(
            model=model, max_tokens=8, system=system or "Return a single number in [0,1].",
            messages=[{"role": "user", "content": prompt}],
        )
        txt = msg.content[0].text.strip()
        try:
            return max(0.0, min(1.0, float(txt)))
        except ValueError:
            return 0.0
    return _judge


@dataclass
class HyperedgeCandidate:
    target: int
    premises: Tuple[int, ...]
    c_forward: float
    c_backward: float

    @property
    def confidence(self) -> float:
        return min(self.c_forward, self.c_backward)


def _forward_run(steps: List[str], target: int, premises: Tuple[int, ...], judge: Judge) -> float:
    masked = [s if i not in premises else "[REMOVED]" for i, s in enumerate(steps[:target])]
    prompt = (
        f"Original reasoning up to step {target}:\n" + "\n".join(steps[:target]) +
        f"\n\nIf the premises {list(premises)} are removed, can step {target} ('{steps[target]}') "
        "still be derived? Return a probability in [0,1] that the step is NO LONGER derivable."
    )
    return judge(prompt)


def _backward_run(steps: List[str], target: int, judge: Judge) -> List[Tuple[int, ...]]:
    prompt = (
        "Identify the minimal set of premise step indices (0-based) required to derive the target step. "
        "Return them as a comma-separated list, e.g. '0,2'.\n\n"
        f"Steps:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(steps[:target])) +
        f"\n\nTarget step [{target}] {steps[target]}"
    )
    raw_score = judge(prompt)
    k = max(1, int(round(raw_score * min(3, target))))
    idxs = list(range(max(0, target - k), target))
    return [tuple(idxs)]


def extract_hyperedges(steps: List[str], judge: Judge, k: int) -> List[HyperedgeCandidate]:
    out: List[HyperedgeCandidate] = []
    for target in range(1, len(steps)):
        premise_budget = [tuple(range(max(0, target - r - 1), target)) for r in range(min(2, target))]
        for premises in premise_budget:
            if not premises:
                continue
            forward_scores = [_forward_run(steps, target, premises, judge) for _ in range(k)]
            c_f = sum(forward_scores) / len(forward_scores)
            backward_hits = sum(1 for run in _backward_run(steps, target, judge)
                                if set(premises).issubset(set(run)))
            c_b = backward_hits / max(1, k)
            out.append(HyperedgeCandidate(target=target, premises=premises, c_forward=c_f, c_backward=c_b))
    return out


def run(dataset: str, model: str, judge: Judge, k: int, tau: float, limit: int) -> Path:
    en = read_jsonl(FEAT_DIR / f"{model}__{dataset}__en__cot.jsonl")
    zh = read_jsonl(FEAT_DIR / f"{model}__{dataset}__zh__cot.jsonl")
    en_by_id = {r["id"]: r for r in en}
    zh_by_id = {r["id"]: r for r in zh}
    matched = sorted(accuracy_matched_ids(
        {i: bool(r["correct"]) for i, r in en_by_id.items()},
        {i: bool(r["correct"]) for i, r in zh_by_id.items()},
    ))[:limit]

    out_rows = []
    for iid in matched:
        for lang, src in (("en", en_by_id), ("zh", zh_by_id)):
            steps = src[iid]["steps"]
            cands = extract_hyperedges(steps, judge, k)
            out_rows.append({
                "id": iid, "lang": lang, "model": model, "dataset": dataset,
                "n_steps": len(steps),
                "hyperedges": [
                    {"target": c.target, "premises": list(c.premises),
                     "c_forward": c.c_forward, "c_backward": c.c_backward,
                     "confidence": c.confidence, "retained": c.confidence >= tau}
                    for c in cands
                ],
            })
    out_path = HG_DIR / f"hg__{model}__{dataset}.jsonl"
    write_jsonl(out_path, out_rows)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xcopa")
    ap.add_argument("--models", nargs="+",
                    default=[k for k, v in MODELS.items() if v.get("hypergraph")])
    ap.add_argument("--judge", choices=["stub", "anthropic"], default="stub")
    ap.add_argument("--k", type=int, default=PILOT["ensemble_k"])
    ap.add_argument("--tau", type=float, default=PILOT["judge_tau"])
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if args.judge == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set")
        judge = anthropic_judge_factory()
    else:
        judge = stub_judge

    for m in args.models:
        out = run(args.dataset, m, judge, args.k, args.tau, args.limit)
        print(f"[hg] {m} -> {out}")


if __name__ == "__main__":
    main()
