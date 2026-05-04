"""Merge per-GPU / per-model prompt-flip outputs into a single JSON.

Day 2's prompt-flip job ran on multiple GPUs in parallel, each writing to a
separate JSON (`pilot_day2_prompt_flip_{gpu2,gpu3,qwen,...}.json`). Each file
is a flat {model_key: {...}} dict. This script deduplicates and merges them
into `output/rankings/pilot_day2_prompt_flip.json`.

If the same model appears in multiple input files, the alphabetically-last
file wins. Conflicts are reported on stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import RANK_DIR  # noqa: E402
from utils.io import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="pilot_day2_prompt_flip_*.json",
                    help="glob within output/rankings/ for input shards")
    ap.add_argument("--out", default=str(RANK_DIR / "pilot_day2_prompt_flip.json"),
                    help="merged output path")
    args = ap.parse_args()

    sources = sorted(Path(RANK_DIR).glob(args.in_glob))
    # Avoid recursing into the merged output if it already exists.
    out_path = Path(args.out)
    sources = [p for p in sources if p.resolve() != out_path.resolve()]

    merged: dict = {}
    provenance: dict = {}
    for p in sources:
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {p.name}: {e}", file=sys.stderr)
            continue
        for model, payload in d.items():
            if model in merged:
                if merged[model] != payload:
                    print(f"[conflict] {model}: {provenance[model]} vs "
                          f"{p.name} differ; keeping {p.name}", file=sys.stderr)
            merged[model] = payload
            provenance[model] = p.name

    write_json(out_path, merged)
    print(f"wrote {out_path}  ({len(merged)} models from {len(sources)} files)")
    for m in sorted(merged):
        print(f"  {m:<32} <- {provenance[m]}")


if __name__ == "__main__":
    main()
