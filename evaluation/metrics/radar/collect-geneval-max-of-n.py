# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect per-GenEval-subskill Best-of-N (pass@N) curves per method.

A focused sibling of ``collect-max-of-n-metrics.py``: instead of the single
GenEval *Overall* capability axis, this one breaks GenEval down into its 7
radar axes (the 6 official tags + the macro-averaged Overall):

  single_object / two_object / counting / colors / position / color_attr /
  overall

For every method it reads the per-(method, geneval) ``evaluation_results.jsonl``
produced by ``score-images.py`` under
``<base_root>/<bestofn_subdir>/<family>/<method>/geneval/`` and, for every N in
``--n_list``, computes:

  * each tag    : pass@N = per prompt (metadata.tag == tag), did *any* of the
                  first N samples (seed_index 0..N-1) score geneval == 1, then
                  mean over that tag's prompts. Higher = better.
  * overall     : macro-average of the 6 tag pass@N values (the official
                  GenEval Overall), matching geneval_overall_at_n in
                  collect-max-of-n-metrics.py and aggregate-bestofn.py.

pass@N convention mirrors aggregate-bestofn.py and collect-max-of-n-metrics.py,
restricted to the first N samples. Prompts missing a seed < N simply contribute
"did any available sample pass", so a partial run still produces a usable number.
A tag with no rows yields a null for that tag and a null Overall (warned).

This is intentionally aggregation-only: it assumes the geneval images were
already generated and scored. It never touches a GPU.

Output: ``<output_dir>/geneval-max-of-n.json`` with the same schema as
``collect-max-of-n-metrics.py``'s max-of-n.json (``metrics`` / ``n_list`` /
``data`` / ``vectors`` / ``config``), so ``plot-radar-geneval.py`` can read it
exactly the way ``plot-radar-overall.py`` reads max-of-n.json.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

# Radar axes (= json metric keys), in the order they appear in each method's
# `vectors` entry. The 6 official GenEval tags + macro-averaged Overall.
GENEVAL_TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
METRIC_ORDER = GENEVAL_TAGS + ["overall"]

GENEVAL_DATASET = "geneval"
GENEVAL_KEY = "geneval"

DEFAULT_METHODS = [
    "base-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusionnft-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
]

DEFAULT_BASE_ROOT = "/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling"


def _results_path(base_root, bestofn_subdir, family, method, dataset):
    return os.path.join(
        base_root, bestofn_subdir, family, method, dataset, "evaluation_results.jsonl"
    )


def load_rows(path):
    """Return list of row dicts, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _group_by_prompt(rows, score_key, n):
    """sample_id -> list of scores[score_key] with seed_index < n (skips holes)."""
    by_prompt = defaultdict(list)
    for r in rows:
        if r.get("seed_index", 0) >= n:
            continue
        v = (r.get("scores") or {}).get(score_key)
        if v is None:
            continue
        by_prompt[r["sample_id"]].append(float(v))
    return by_prompt


def pass_at_n(rows, score_key, n, threshold=1.0):
    """pass@N: per prompt (any of first N samples >= threshold), mean over prompts.

    Returns (value, n_prompts). None if no prompt has a usable score.
    """
    by_prompt = _group_by_prompt(rows, score_key, n)
    if not by_prompt:
        return None, 0
    passed = [1.0 if any(v >= threshold for v in vals) else 0.0 for vals in by_prompt.values()]
    return float(np.mean(passed)), len(passed)


def geneval_subskills_at_n(rows, n):
    """Return ({tag: pass@N | None, "overall": macro-avg | None}, warnings_detail).

    Each tag's prompts (metadata.tag) contribute their own pass@N; Overall is the
    equal-weight mean of the 6 tag rates (the official GenEval Overall). Overall
    is None unless all 6 tags have a usable rate.
    """
    by_tag = defaultdict(list)
    for r in rows:
        tag = (r.get("metadata") or {}).get("tag")
        if tag is not None:
            by_tag[tag].append(r)

    out = {}
    detail = {}
    tag_rates = []
    for tag in GENEVAL_TAGS:
        sub = by_tag.get(tag)
        if not sub:
            out[tag] = None
            detail[tag] = "missing_tag"
            continue
        rate, _ = pass_at_n(sub, GENEVAL_KEY, n, threshold=1.0)
        out[tag] = rate
        if rate is None:
            detail[tag] = "no_scores"
        else:
            tag_rates.append(rate)

    out["overall"] = float(np.mean(tag_rates)) if len(tag_rates) == len(GENEVAL_TAGS) else None
    return out, detail


def compute_method(method, base_root, bestofn_subdir, family, n_list):
    """Return (data_for_method, warnings) where data_for_method is
    {metric_name: {n: value_or_None}}."""
    warnings = []
    path = _results_path(base_root, bestofn_subdir, family, method, GENEVAL_DATASET)
    rows = load_rows(path)
    data = {m: {} for m in METRIC_ORDER}

    if rows is None:
        warnings.append(f"[{method}] missing: {path}")
        for m in METRIC_ORDER:
            for n in n_list:
                data[m][n] = None
        return data, warnings

    for n in n_list:
        vals, detail = geneval_subskills_at_n(rows, n)
        for m in METRIC_ORDER:
            data[m][n] = vals[m]
            if vals[m] is None:
                why = detail.get(m, "macro-avg unavailable (a tag is missing)")
                warnings.append(f"[{method}] geneval {m} unavailable at n={n}: {why}")
    return data, warnings


def main(args):
    n_list = sorted(set(args.n_list))
    all_data = {}
    all_warnings = []

    for method in args.methods:
        data, warnings = compute_method(
            method, args.base_root, args.bestofn_subdir, args.family, n_list
        )
        all_data[method] = data
        all_warnings.extend(warnings)

    # Build per-method radar vectors at each N (axis order = METRIC_ORDER).
    vectors = {}
    for method, data in all_data.items():
        vectors[method] = {
            str(n): [data[m][n] for m in METRIC_ORDER] for n in n_list
        }

    out = {
        "metrics": METRIC_ORDER,
        "n_list": n_list,
        # JSON object keys are strings; stringify n for portability.
        "data": {
            method: {m: {str(n): data[m][n] for n in n_list} for m in METRIC_ORDER}
            for method, data in all_data.items()
        },
        "vectors": vectors,
        "config": {
            "base_root": args.base_root,
            "bestofn_subdir": args.bestofn_subdir,
            "family": args.family,
            "methods": args.methods,
            "dataset": GENEVAL_DATASET,
            "score_key": GENEVAL_KEY,
            "aggregation": "pass@N (Best-of-N): per-prompt any-of-first-N geneval==1, then mean over prompts; "
                           "overall = macro-avg over the 6 tags",
            "tags": GENEVAL_TAGS,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "geneval-max-of-n.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary: one table per N showing the 7 axes for each method.
    for n in n_list:
        print(f"\n=== geneval pass@N={n} ===")
        print(f"{'method':<26} " + "".join(f"{m:>14}" for m in METRIC_ORDER))
        for method in args.methods:
            cells = []
            for m in METRIC_ORDER:
                v = all_data[method][m][n]
                cells.append("          n/a" if v is None else f"{v:>14.4f}")
            print(f"{method:<26} " + "".join(cells))

    if all_warnings:
        print(f"\n--- {len(all_warnings)} warning(s) ---")
        for w in all_warnings:
            print(f"  {w}")

    print(f"\nSaved geneval max-of-n data to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect per-GenEval-subskill pass@N (Best-of-N) curves "
        "(single_object / two_object / counting / colors / position / "
        "color_attr / overall) for a set of methods, into a single "
        "geneval-max-of-n.json for downstream radar plotting."
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help="Root holding <bestofn_subdir>/<family>/<method>/geneval/ outputs.",
    )
    ap.add_argument(
        "--bestofn_subdir", default="bestofn-eval",
        help="Subdir under base_root that holds the Best-of-N eval outputs.",
    )
    ap.add_argument(
        "--family", default="sd-3.5-m",
        help="Model family dir (run-bestofn.sh maps *-sd3 -> sd-3.5-m).",
    )
    ap.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help="Method dirs to collect (default: the 6 SD-3.5-M methods).",
    )
    ap.add_argument(
        "--n_list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
        help="N values (samples per prompt) at which to compute pass@N.",
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Folder to write geneval-max-of-n.json into.",
    )
    main(ap.parse_args())
