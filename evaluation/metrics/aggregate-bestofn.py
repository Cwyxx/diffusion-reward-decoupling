"""Compute Best-of-N curves for one (method, dataset) directory.

Inputs:
  ${output_dir}/evaluation_results.jsonl   (rows keyed by (sample_id, seed_index))

Outputs:
  ${output_dir}/bestofn/curves.json
  ${output_dir}/bestofn/plots/<metric>_curve_log.png
  ${output_dir}/bestofn/csv/<metric>_curve.csv

Aggregation per metric:
  - HP-style continuous (pickscore, hpsv3, deqa, aesthetic, ...): mean over
    prompts of max over the first n samples.
  - OCR (continuous, thresholded for BoN): pass_at_n with default threshold
    1.0 (exact match). Re-running with a different threshold needs only
    re-aggregation, not re-scoring.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Metrics whose continuous scores get thresholded into pass/fail for BoN.
# Everything else uses mean-of-max over the continuous values.
BINARY_METRICS = {"ocr"}


# ---------- BoN primitives (pure math) ----------

def bon_continuous(scores: np.ndarray, n: int) -> float:
    """HP-style: mean over prompts of max over first n samples."""
    if not 1 <= n <= scores.shape[1]:
        raise ValueError(f"n={n} out of range [1, {scores.shape[1]}]")
    return float(np.mean(np.max(scores[:, :n], axis=1)))


def pass_at_n(scores: np.ndarray, n: int, threshold: float = 1.0) -> float:
    """Binary: mean over prompts of (any of first n samples >= threshold)."""
    if not 1 <= n <= scores.shape[1]:
        raise ValueError(f"n={n} out of range [1, {scores.shape[1]}]")
    return float(np.mean(np.any(scores[:, :n] >= threshold, axis=1)))


def aggregate_curve(
    scores: np.ndarray,
    kind: Literal["continuous", "binary"],
    threshold: float = 1.0,
) -> Dict[int, float]:
    """Compute the BoN value for every n in [1, n_max]."""
    n_max = scores.shape[1]
    if kind == "continuous":
        return {n: bon_continuous(scores, n) for n in range(1, n_max + 1)}
    elif kind == "binary":
        return {n: pass_at_n(scores, n, threshold) for n in range(1, n_max + 1)}
    else:
        raise ValueError(f"unknown kind={kind!r}")


# ---------- IO + matrix building ----------

def load_results(results_path):
    rows = []
    with open(results_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def build_score_matrix(rows, metric):
    """rows -> (num_prompts, n_max) ndarray of `metric` scores.

    Returns None if no row has this metric. Raises ValueError if some rows
    have it but the matrix is incomplete (any (sid, seed) is unscored).
    """
    grouped = defaultdict(dict)
    for r in rows:
        if metric not in r["scores"]:
            continue
        # BoN rows have seed_index; legacy rows fall back to 0.
        seed_idx = r.get("seed_index", 0)
        grouped[r["sample_id"]][seed_idx] = r["scores"][metric]

    if not grouped:
        return None

    sample_ids = sorted(grouped.keys())
    n_max = max(max(v.keys()) for v in grouped.values()) + 1
    mat = np.full((len(sample_ids), n_max), np.nan, dtype=float)
    for i, sid in enumerate(sample_ids):
        for seed_idx, val in grouped[sid].items():
            mat[i, seed_idx] = val

    if np.isnan(mat).any():
        n_missing = int(np.isnan(mat).sum())
        raise ValueError(
            f"Score matrix for metric={metric!r} has {n_missing} NaN entries; "
            f"some (sample_id, seed_index) pairs are unscored. "
            f"Re-run scoring (with --force if needed) before aggregating."
        )
    return mat


def write_curve_csv(curve, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "value"])
        for n in sorted(curve.keys()):
            writer.writerow([n, f"{curve[n]:.6f}"])


def plot_curve(curve, metric, kind, threshold, out_path):
    ns = sorted(curve.keys())
    ys = [curve[n] for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ns, ys, marker="o", markersize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (samples per prompt)")
    if kind == "binary":
        ax.set_ylabel(f"pass@N (threshold={threshold})")
        ax.set_title(f"BoN curve: {metric} (binary)")
    else:
        ax.set_ylabel(f"BoN({metric})")
        ax.set_title(f"BoN curve: {metric}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- main ----------

def main(args):
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    rows = load_results(results_path)

    metrics = sorted({m for r in rows for m in r["scores"].keys()})
    if not metrics:
        sys.exit(f"No scores found in {results_path}; run scoring first.")

    bestofn_dir = os.path.join(args.output_dir, "bestofn")
    plots_dir = os.path.join(bestofn_dir, "plots")
    csv_dir = os.path.join(bestofn_dir, "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    out = {}
    for metric in metrics:
        mat = build_score_matrix(rows, metric)
        if mat is None:
            continue
        kind = "binary" if metric in BINARY_METRICS else "continuous"
        threshold = args.ocr_threshold if metric == "ocr" else 1.0
        curve = aggregate_curve(mat, kind=kind, threshold=threshold)

        info = {
            "kind": kind,
            "n_max": mat.shape[1],
            "num_prompts": mat.shape[0],
            "curve": curve,
            "ceiling_lift": curve[mat.shape[1]] - curve[1],
        }
        if kind == "binary":
            info["threshold"] = threshold
        out[metric] = info

        plot_curve(curve, metric, kind, threshold,
                   os.path.join(plots_dir, f"{metric}_curve_log.png"))
        write_curve_csv(curve, os.path.join(csv_dir, f"{metric}_curve.csv"))

    curves_path = os.path.join(bestofn_dir, "curves.json")
    with open(curves_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {curves_path}")
    for m, info in out.items():
        thresh_note = f", threshold={info['threshold']}" if "threshold" in info else ""
        print(f"  {m:<20} kind={info['kind']:<10} N={info['n_max']:>3}{thresh_note}  "
              f"ceiling_lift={info['ceiling_lift']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compute Best-of-N curves for one (method, dataset) directory."
    )
    ap.add_argument("--output_dir", required=True,
                    help="Directory containing evaluation_results.jsonl.")
    ap.add_argument("--ocr_threshold", type=float, default=1.0,
                    help="Threshold for OCR pass@N (default 1.0 = exact match).")
    main(ap.parse_args())
