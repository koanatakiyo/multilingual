"""Day 5: bidirectional ensemble hyperedge extraction on matched subsets.

For each of the 3 hypergraph models, for each matched item:

    - k forward counterfactual joint-necessity tests per candidate hyperedge
      (masking-based removal). c_f is the mean across k runs.
    - k backward judge runs per target step, each returning a SET of premise
      indices identified as necessary. For a given candidate (target, premises),
      c_b = (number of backward runs whose returned set ⊇ premises) / k.

The forward and backward judges have different return contracts:
    ForwardJudge : prompt -> float in [0, 1]  (probability of NON-derivability)
    BackwardJudge: prompt -> str              (judge-raw text, parsed here)

Keeping them separate makes the real anthropic judge trivially wireable and
makes it impossible to accidentally reuse a scalar judge for backward
extraction (the earlier bug).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import FEAT_DIR, HG_DIR, MODELS, PILOT, load_api_key  # noqa: E402
from utils.io import read_jsonl, write_jsonl  # noqa: E402
from utils.matching import accuracy_matched_ids  # noqa: E402


ForwardJudge = Callable[[str], float]
BackwardJudge = Callable[[str], str]


_STUB_WARNED = False


def _warn_stub_once() -> None:
    global _STUB_WARNED
    if not _STUB_WARNED:
        print("[WARN] stub judge(s) in use — D_HG is NOT valid. Use --judge "
              "anthropic (with ANTHROPIC_API_KEY) for real data.", file=sys.stderr)
        _STUB_WARNED = True


def stub_forward(prompt: str, run_seed: int = 0) -> float:
    """Deterministic pseudo-judge. Adds `run_seed` so pilot_14 can exercise
    real variance when --allow-stub is explicitly requested."""
    _warn_stub_once()
    h = abs(hash(f"{run_seed}|{prompt}")) % 1000
    return h / 1000.0


def stub_backward(prompt: str, run_seed: int = 0) -> str:
    """Returns a comma-separated index list based on a hash of prompt+seed."""
    _warn_stub_once()
    h = abs(hash(f"b|{run_seed}|{prompt}")) % 10_000
    # 0-2 indices; the caller clamps against target.
    n = (h // 1000) % 3
    if n == 0:
        return ""
    base = (h // 10) % 7
    return ",".join(str((base + i) % 8) for i in range(n))


def anthropic_forward_factory(model: str = "claude-opus-4-7") -> ForwardJudge:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic()
    system = "Return a single number in [0,1] and nothing else."
    def _judge(prompt: str) -> float:
        msg = client.messages.create(
            model=model, max_tokens=8, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = msg.content[0].text.strip()
        try:
            return max(0.0, min(1.0, float(txt)))
        except ValueError:
            return 0.0
    return _judge


def anthropic_backward_factory(model: str = "claude-opus-4-7") -> BackwardJudge:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic()
    system = ("Return a comma-separated list of 0-based step indices of the minimal "
              "premises required to derive the target step. Nothing else.")
    def _judge(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=32, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    return _judge


def _openai_client():
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=load_api_key("openai"))


def openai_forward_factory(model: str = "gpt-5.4-mini") -> ForwardJudge:
    client = _openai_client()
    system = "Return a single number in [0,1] and nothing else."
    def _judge(prompt: str) -> float:
        resp = client.chat.completions.create(
            model=model, max_tokens=8, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        try:
            return max(0.0, min(1.0, float(txt)))
        except ValueError:
            return 0.0
    return _judge


def openai_backward_factory(model: str = "gpt-5.4-mini") -> BackwardJudge:
    client = _openai_client()
    system = ("Return a comma-separated list of 0-based step indices of the minimal "
              "premises required to derive the target step. Nothing else.")
    def _judge(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, max_tokens=32, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    return _judge


def _parse_indices(raw: str, max_idx: int) -> Set[int]:
    """Parse judge text into a set of valid 0-based indices."""
    out: Set[int] = set()
    for tok in re.findall(r"\d+", raw or ""):
        i = int(tok)
        if 0 <= i < max_idx:
            out.add(i)
    return out


@dataclass
class HyperedgeCandidate:
    target: int
    premises: Tuple[int, ...]
    c_forward: float
    c_backward: float

    @property
    def confidence(self) -> float:
        return min(self.c_forward, self.c_backward)


def _forward_prompt(steps: List[str], target: int, premises: Tuple[int, ...]) -> str:
    return (
        f"Original reasoning up to step {target}:\n" + "\n".join(steps[:target]) +
        f"\n\nIf the premises at indices {list(premises)} are removed, can step "
        f"{target} ('{steps[target]}') still be derived? Return a probability "
        "in [0,1] that the step is NO LONGER derivable."
    )


def _backward_prompt(steps: List[str], target: int) -> str:
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(steps[:target]))
    return (
        "Identify the minimal set of premise step indices (0-based) required "
        "to derive the target step. Return them as a comma-separated list.\n\n"
        f"Steps:\n{body}\n\nTarget step [{target}] {steps[target]}"
    )


def _candidate_premise_sets(target: int) -> List[Tuple[int, ...]]:
    """Pairwise and triple-wise contiguous premise sets (arity 1 and 2)."""
    cands: List[Tuple[int, ...]] = []
    if target >= 1:
        cands.append((target - 1,))
    if target >= 2:
        cands.append((target - 2, target - 1))
    return cands


def extract_hyperedges(
    steps: List[str],
    forward_judge: ForwardJudge,
    backward_judge: BackwardJudge,
    k: int,
    run_seed: int = 0,
) -> List[HyperedgeCandidate]:
    """Run k forward + k backward ensembles per target, then score candidates.

    `run_seed` is only consulted by stub judges (to inject variance for pipeline
    tests). Real anthropic judges ignore it.
    """
    out: List[HyperedgeCandidate] = []
    for target in range(1, len(steps)):
        # k backward runs, shared across candidate premise sets for this target.
        backward_sets: List[Set[int]] = []
        bprompt = _backward_prompt(steps, target)
        for j in range(k):
            raw = _call_judge(backward_judge, bprompt, run_seed=run_seed * 100 + j)
            backward_sets.append(_parse_indices(raw, max_idx=target))

        for premises in _candidate_premise_sets(target):
            fprompt = _forward_prompt(steps, target, premises)
            forward_scores = [
                _call_judge(forward_judge, fprompt, run_seed=run_seed * 100 + j)
                for j in range(k)
            ]
            c_f = sum(forward_scores) / len(forward_scores)
            pset = set(premises)
            c_b = sum(1 for s in backward_sets if pset.issubset(s)) / k
            out.append(HyperedgeCandidate(
                target=target, premises=premises, c_forward=c_f, c_backward=c_b,
            ))
    return out


def _call_judge(judge, prompt: str, run_seed: int):
    """Dispatch to stub (with run_seed) or real judge (without)."""
    try:
        return judge(prompt, run_seed=run_seed)
    except TypeError:
        return judge(prompt)


def run(dataset: str, model: str, forward_judge: ForwardJudge,
        backward_judge: BackwardJudge, k: int, tau: float, limit: int) -> Path:
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
            cands = extract_hyperedges(steps, forward_judge, backward_judge, k)
            out_rows.append({
                "id": iid, "lang": lang, "model": model, "dataset": dataset,
                "n_steps": len(steps),
                "steps": steps,
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
    ap.add_argument("--judge", choices=["stub", "openai", "anthropic"], default="openai")
    ap.add_argument("--judge_model", default="gpt-5.4-mini",
                    help="model name passed to the chosen provider")
    ap.add_argument("--k", type=int, default=PILOT["ensemble_k"])
    ap.add_argument("--tau", type=float, default=PILOT["judge_tau"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--allow-stub", action="store_true",
                    help="permit the stub judges (pipeline tests only)")
    args = ap.parse_args()

    if args.judge == "openai":
        forward_judge = openai_forward_factory(args.judge_model)
        backward_judge = openai_backward_factory(args.judge_model)
    elif args.judge == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set")
        forward_judge = anthropic_forward_factory(args.judge_model)
        backward_judge = anthropic_backward_factory(args.judge_model)
    else:
        if not args.allow_stub:
            raise SystemExit(
                "Refusing to run hypergraph extraction with stub judges. "
                "Pass --allow-stub for pipeline tests, or --judge openai/anthropic for real data."
            )
        forward_judge = stub_forward
        backward_judge = stub_backward

    for m in args.models:
        out = run(args.dataset, m, forward_judge, backward_judge, args.k, args.tau, args.limit)
        print(f"[hg] {m} -> {out}")


if __name__ == "__main__":
    main()
