"""Average@K for the AnyText OCR metrics (anytext-senacc, anytext-ned).

Standalone, read-only companion to aggregate-bestofn.py. Where the Best-of-N
aggregator takes the *max* over seeds (capability ceiling), this reports the
AnyText paper's *average* protocol: per prompt, mean over the first K seeds
(default 4 = eval_dgocr.py's num_samples), then mean over prompts.

Macro (per-prompt equal weight), matching DPG-Bench's Average DPG-Score
(aggregate-bestofn.py:_aggregate_dpg / AVG_DPG_NUM_SEEDS). This differs from the
official eval_dgocr.py micro-average (which pools every text line of all 4
samples flat, so line-rich prompts weigh more); choose this script's macro form
to compare prompts on equal footing.

NOTE: even at average@4 these numbers do NOT equal the official AnyText
benchmark, because scoring here is position-free (full-image DuGuang detection)
rather than GT-polygon crops -- a deliberate methodology difference for plain
T2I models (see evaluation/benchmarks/AnyText/anytext_scorer.py). This script
only reproduces the *averaging* protocol, not the official cropping.

Usage:
  python evaluation/metrics/average-anytext-seeds.py \
      --output_dir /.../bestofn-eval/sd-3.5-m/base-sd3/anytext-en
  # optionally: --num_seeds 4 --metrics anytext-senacc anytext-ned
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_path):
    if not os.path.exists(results_path):
        raise SystemExit(f"No results file at {results_path}; run scoring first.")
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_matrix(rows, metric):
    """rows -> (num_prompts, n_max) ndarray of `metric` scores, NaN for holes.

    Indexed by (sorted sample_id, seed_index), mirroring
    aggregate-bestofn.py:build_score_matrix so the seed columns line up.
    Returns (mat, sample_ids) or (None, None) if no row carries the metric.
    """
    grouped = defaultdict(dict)
    for r in rows:
        scores = r.get("scores") or {}
        if metric not in scores:
            continue
        grouped[r["sample_id"]][r.get("seed_index", 0)] = scores[metric]
    if not grouped:
        return None, None
    sample_ids = sorted(grouped.keys())
    n_max = max(max(v.keys()) for v in grouped.values()) + 1
    mat = np.full((len(sample_ids), n_max), np.nan, dtype=float)
    for i, sid in enumerate(sample_ids):
        for seed_idx, val in grouped[sid].items():
            mat[i, seed_idx] = val
    return mat, sample_ids


def average_at_k(mat, k):
    """Macro average@K: per-prompt mean over the first k seeds, then mean over prompts.

    Returns (score, per_prompt_avg). Raises if fewer than k seeds exist or any
    of the first k columns has a hole (so the average is well-defined).
    """
    n_max = mat.shape[1]
    if n_max < k:
        raise SystemExit(
            f"average@{k} needs >= {k} seeds per prompt; matrix has n_max={n_max}. "
            f"Generate more seeds or lower --num_seeds."
        )
    window = mat[:, :k]
    n_missing = int(np.isnan(window).sum())
    if n_missing:
        raise SystemExit(
            f"{n_missing} of the first {k} (sample_id, seed_index) scores are missing; "
            f"cannot average@{k}. Re-score so every prompt has seeds 0..{k - 1}."
        )
    per_prompt_avg = window.mean(axis=1)            # (n_prompts,)
    return float(per_prompt_avg.mean()), per_prompt_avg


def main(args):
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    rows = load_results(results_path)

    k = args.num_seeds
    summary = {"num_seeds": k, "metrics": {}}
    per_prompt_out = defaultdict(dict)   # sample_id -> {metric: avg}
    prompts = {}

    print(f"average@{k} (macro, per-prompt equal weight) — {args.output_dir}")
    for metric in args.metrics:
        mat, sample_ids = build_matrix(rows, metric)
        if mat is None:
            print(f"  {metric:<16}: (no scores found, skipped)")
            continue
        score, per_prompt_avg = average_at_k(mat, k)
        summary["metrics"][metric] = {
            "average_at_k": score,
            "num_prompts": len(sample_ids),
            "n_max": mat.shape[1],
        }
        print(f"  {metric:<16}: {score:.4f}   (num_prompts={len(sample_ids)}, n_max={mat.shape[1]})")
        for sid, avg in zip(sample_ids, per_prompt_avg):
            per_prompt_out[sid][metric] = float(avg)

    if not summary["metrics"]:
        raise SystemExit("None of the requested metrics were found in the results.")

    # prompt text for the per-prompt table
    for r in rows:
        prompts.setdefault(r["sample_id"], r.get("prompt", ""))

    out_dir = os.path.join(args.output_dir, "bestofn")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"anytext_average_at_{k}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    pp_path = os.path.join(out_dir, f"per_prompt_anytext_average_at_{k}.jsonl")
    with open(pp_path, "w") as f:
        for sid in sorted(per_prompt_out.keys()):
            row = {"sample_id": sid, "prompt": prompts.get(sid, "")}
            row.update(per_prompt_out[sid])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {pp_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Average@K for AnyText OCR metrics (macro, per-prompt equal weight)."
    )
    ap.add_argument("--output_dir", required=True,
                    help="Directory containing evaluation_results.jsonl.")
    ap.add_argument("--num_seeds", type=int, default=4,
                    help="K: average over the first K seeds (default 4, = eval_dgocr num_samples).")
    ap.add_argument("--metrics", nargs="+",
                    default=["anytext-senacc", "anytext-ned"],
                    help="Per-image score keys to average (default: anytext-senacc anytext-ned).")
    main(ap.parse_args())
