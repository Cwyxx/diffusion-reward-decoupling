# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect four capability radar-chart metrics (max@N / Best-of-N) per method.

Sibling of ``collect-mean-of-n-metrics.py``. Where that script averages over the
first N samples (per-image rates / bias / safety), this one takes the *max* over
the first N samples per prompt (the Best-of-N capability ceiling), then averages
over prompts. It reads the per-(method, dataset) ``evaluation_results.jsonl``
produced by ``score-images.py`` under
``<base_root>/<bestofn_subdir>/<family>/<method>/<dataset>/`` and computes, for
every N in ``--n_list``, the four metrics:

  Object Alignment : GenEval Overall = macro-avg over the 6 GenEval tags of
                     pass@N (any of the first N samples scores geneval==1).
                     dataset=geneval, score_key=geneval. Higher = better.
  Dense Prompt     : DPG-Bench Best-of-N = mean over prompts of max over the
                     first N samples of dpg-score-mplug. dataset=dpg_bench,
                     score_key=dpg-score-mplug. Higher = better.
  World Knowledge  : WISE Overall = weighted sum over the 6 WISE categories
                     (CULTURE 0.40, others 0.12) of pass@N (any of first N
                     samples scores wise==1). dataset=wise, score_key=wise.
                     Higher = better.
  Visual Text      : AnyText (English) Best-of-N = mean over prompts of max over
                     the first N samples of anytext-senacc (Sen.ACC, the AnyText
                     headline). dataset=anytext-en, score_key=anytext-senacc.
                     Higher = better.

max@N convention: per prompt take the first N samples (seed_index 0..N-1), take
the max of the per-image metric (or "did any pass" for the binary GenEval/WISE
metrics), then average over prompts. The aggregation mirrors aggregate-bestofn.py
(bon_continuous / pass_at_n + GenEval macro-avg / WISE weighted-sum) restricted
to the first N samples. Prompts missing a seed < N simply contribute the max over
whatever samples they do have (saturating at the available n_max), so a partial
run still produces a usable number.

This is intentionally aggregation-only: it assumes the images were already
generated and scored by run-bestofn(-batch).sh. It never touches a GPU.

Output: ``<output_dir>/max-of-n.json`` with::

  {
    "metrics": ["Object Alignment", "Dense Prompt", "World Knowledge", "Visual Text"],
    "n_list": [1, 2, 4, 8, 16, 32],
    "data":    { method: { metric: { "1": value, "2": value, ... } } },
    "vectors": { method: { "1": [Obj, Dense, World, Text], ... } },
    "config":  { ... provenance ... }
  }

``data`` is convenient for per-metric reads; ``vectors`` gives each method's
4-axis radar vector at every N. A missing value (dataset not scored yet) is
``null``; the script warns and keeps going so a partial run is still useful.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

# Radar axes, in the order they appear in each method's `vectors` entry.
METRIC_ORDER = ["Object Alignment", "Dense Prompt", "World Knowledge", "Visual Text"]

DEFAULT_METHODS = [
    "base-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusionnft-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
]

DEFAULT_BASE_ROOT = "/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling"

# Dataset dir -> score key, per metric.
GENEVAL_DATASET = "geneval"
GENEVAL_KEY = "geneval"
DPG_DATASET = "dpg_bench"
DPG_KEY = "dpg-score-mplug"
WISE_DATASET = "wise"
WISE_KEY = "wise"
ANYTEXT_DATASET = "anytext-en"
ANYTEXT_KEY = "anytext-senacc"

# The 6 GenEval dimensions, matching aggregate-bestofn.py:GENEVAL_TAGS and the
# official summary_scores.py grouping. GenEval Overall = macro-avg over these.
GENEVAL_TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]

# WISE_Verified categories + weights, matching aggregate-bestofn.py:WISE_CATEGORY_SPEC
# and evaluation/benchmarks/WISE/calculate_verified.py:246. Weights sum to 1.0.
WISE_CATEGORY_SPEC = [
    # (name, prompt_id range [closed-open], weight)
    ("CULTURE",   (1, 401),    0.40),
    ("TIME",      (401, 521),  0.12),
    ("SPACE",     (521, 641),  0.12),
    ("BIOLOGY",   (641, 761),  0.12),
    ("PHYSICS",   (761, 881),  0.12),
    ("CHEMISTRY", (881, 1001), 0.12),
]


def _wise_category_for(prompt_id):
    for name, (lo, hi), _w in WISE_CATEGORY_SPEC:
        if lo <= prompt_id < hi:
            return name
    return None


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


def max_at_n(rows, score_key, n):
    """Best-of-N: per prompt max over the first N samples, then mean over prompts.

    Returns (value, n_prompts). None if no prompt has a usable score.
    """
    by_prompt = _group_by_prompt(rows, score_key, n)
    if not by_prompt:
        return None, 0
    maxes = [max(vals) for vals in by_prompt.values()]
    return float(np.mean(maxes)), len(maxes)


def pass_at_n(rows, score_key, n, threshold=1.0):
    """pass@N: per prompt (any of first N samples >= threshold), mean over prompts.

    Returns (value, n_prompts). None if no prompt has a usable score.
    """
    by_prompt = _group_by_prompt(rows, score_key, n)
    if not by_prompt:
        return None, 0
    passed = [1.0 if any(v >= threshold for v in vals) else 0.0 for vals in by_prompt.values()]
    return float(np.mean(passed)), len(passed)


def geneval_overall_at_n(rows, n):
    """GenEval Overall = macro-avg over the 6 tags of pass@N. Returns (val, detail).

    Each tag's prompts (metadata.tag) contribute their own pass@N; the 6 tag
    rates are averaged with equal weight (the official "Overall"). Requires all
    6 tags to be present; otherwise returns (None, ...).
    """
    by_tag = defaultdict(list)
    for r in rows:
        tag = (r.get("metadata") or {}).get("tag")
        if tag is not None:
            by_tag[tag].append(r)

    tag_rates = {}
    for tag in GENEVAL_TAGS:
        sub = by_tag.get(tag)
        if not sub:
            return None, {"missing_tag": tag}
        rate, _ = pass_at_n(sub, GENEVAL_KEY, n, threshold=1.0)
        if rate is None:
            return None, {"no_scores_tag": tag}
        tag_rates[tag] = rate
    return float(np.mean([tag_rates[t] for t in GENEVAL_TAGS])), {"tag_rates": tag_rates}


def wise_overall_at_n(rows, n):
    """WISE Overall = weighted sum over the 6 categories of pass@N. Returns (val, detail).

    Categories are read from metadata.prompt_id; weights match WISE_CATEGORY_SPEC
    (CULTURE 0.40, others 0.12; sum 1.0). Requires all 6 categories present;
    otherwise returns (None, ...).
    """
    by_cat = defaultdict(list)
    for r in rows:
        pid = (r.get("metadata") or {}).get("prompt_id")
        if pid is None:
            continue
        cat = _wise_category_for(pid)
        if cat is not None:
            by_cat[cat].append(r)

    overall = 0.0
    cat_rates = {}
    for name, _rng, w in WISE_CATEGORY_SPEC:
        sub = by_cat.get(name)
        if not sub:
            return None, {"missing_category": name}
        rate, _ = pass_at_n(sub, WISE_KEY, n, threshold=1.0)
        if rate is None:
            return None, {"no_scores_category": name}
        cat_rates[name] = rate
        overall += w * rate
    return float(overall), {"category_rates": cat_rates}


def compute_method(method, base_root, bestofn_subdir, family, n_list):
    """Return (data_for_method, warnings) where data_for_method is
    {metric_name: {n: value_or_None}}."""
    warnings = []

    def load(dataset):
        path = _results_path(base_root, bestofn_subdir, family, method, dataset)
        rows = load_rows(path)
        if rows is None:
            warnings.append(f"[{method}] missing: {path}")
        return rows

    geneval_rows = load(GENEVAL_DATASET)
    dpg_rows = load(DPG_DATASET)
    wise_rows = load(WISE_DATASET)
    anytext_rows = load(ANYTEXT_DATASET)

    data = {m: {} for m in METRIC_ORDER}
    for n in n_list:
        # Object Alignment (GenEval Overall, macro pass@N)
        if geneval_rows is None:
            data["Object Alignment"][n] = None
        else:
            val, detail = geneval_overall_at_n(geneval_rows, n)
            data["Object Alignment"][n] = val
            if val is None:
                warnings.append(f"[{method}] geneval Overall unavailable at n={n}: {detail}")

        # Dense Prompt (DPG-Bench Best-of-N, continuous max@N)
        if dpg_rows is None:
            data["Dense Prompt"][n] = None
        else:
            val, _ = max_at_n(dpg_rows, DPG_KEY, n)
            data["Dense Prompt"][n] = val
            if val is None:
                warnings.append(f"[{method}] dpg_bench: no '{DPG_KEY}' scores at n={n}")

        # World Knowledge (WISE Overall, weighted pass@N)
        if wise_rows is None:
            data["World Knowledge"][n] = None
        else:
            val, detail = wise_overall_at_n(wise_rows, n)
            data["World Knowledge"][n] = val
            if val is None:
                warnings.append(f"[{method}] wise Overall unavailable at n={n}: {detail}")

        # Visual Text (AnyText-EN Best-of-N, continuous max@N on Sen.ACC)
        if anytext_rows is None:
            data["Visual Text"][n] = None
        else:
            val, _ = max_at_n(anytext_rows, ANYTEXT_KEY, n)
            data["Visual Text"][n] = val
            if val is None:
                warnings.append(f"[{method}] anytext-en: no '{ANYTEXT_KEY}' scores at n={n}")

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
            "aggregation": "max@N (Best-of-N): per-prompt max/any over first N samples, then mean over prompts",
            "metric_sources": {
                "Object Alignment": {
                    "dataset": GENEVAL_DATASET, "score_key": GENEVAL_KEY,
                    "definition": "GenEval Overall = macro-avg over 6 tags of pass@N (threshold=1.0)",
                    "full_name": "Object focused text-to-image alignment",
                },
                "Dense Prompt": {
                    "dataset": DPG_DATASET, "score_key": DPG_KEY,
                    "definition": "DPG-Bench Best-of-N = mean over prompts of max over first N samples",
                    "full_name": "Dense prompt following",
                },
                "World Knowledge": {
                    "dataset": WISE_DATASET, "score_key": WISE_KEY,
                    "definition": "WISE Overall = weighted sum over 6 categories of pass@N (CULTURE 0.40, others 0.12)",
                    "full_name": "World Knowledge",
                },
                "Visual Text": {
                    "dataset": ANYTEXT_DATASET, "score_key": ANYTEXT_KEY,
                    "definition": "AnyText-EN Best-of-N = mean over prompts of max over first N samples of Sen.ACC",
                    "full_name": "Visual text generation",
                },
            },
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "max-of-n.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary: one table per N showing the 4 axes for each method.
    for n in n_list:
        print(f"\n=== max@N={n} ===")
        print(f"{'method':<26} " + "".join(f"{m:>18}" for m in METRIC_ORDER))
        for method in args.methods:
            cells = []
            for m in METRIC_ORDER:
                v = all_data[method][m][n]
                cells.append("              n/a" if v is None else f"{v:>18.4f}")
            print(f"{method:<26} " + "".join(cells))

    if all_warnings:
        print(f"\n--- {len(all_warnings)} warning(s) ---")
        for w in all_warnings:
            print(f"  {w}")

    print(f"\nSaved max-of-n data to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect the four capability radar-chart metrics (Object "
        "Alignment / Dense Prompt / World Knowledge / Visual Text) as max@N "
        "(Best-of-N) curves for a set of methods, into a single max-of-n.json "
        "for downstream radar plotting."
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help="Root holding <bestofn_subdir>/<family>/<method>/<dataset>/ outputs.",
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
        help="N values (samples per prompt) at which to compute max@N.",
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Folder to write max-of-n.json into.",
    )
    main(ap.parse_args())
