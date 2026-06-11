"""Activation vs amplification: does post-training create new unsafe behavior,
or just amplify base's existing tendency?

For each post-training method on the unsafe_template dataset (30 prompts x 32
seeds), decompose how its unsafe-rate increase over base splits between:
  - Cold prompts (base_ratio = 0/32, base never produces unsafe in our sample)
  - Warm/Hot prompts (base already produces some unsafe images)

Detector: ShieldGemma's binary aggregate (shieldgemma_fp16-unsafe), which combines
sexually-explicit / dangerous / violence-gore. Outputs:
  mode_simple_shieldgemma_fp16.csv         simplest read: 2 numbers per method (unsafe rate on Safe vs Unsafe prompts)
  mode_scatter_shieldgemma_fp16.png        5-panel scatter (base_ratio vs method_ratio)
  mode_decomposition_shieldgemma_fp16.csv  per-method delta-images split by base-ratio bucket
  seed_level_shieldgemma_fp16.csv          per-(method, prompt) seed-level 2x2 cell counts + per-method aggregate

Caveat: with only 32 samples per prompt, "Cold" (base_ratio = 0) does not prove
the population base rate is zero -- Wilson 95% upper bound on 0/32 is ~0.109.
Activation events on Cold prompts are statistically convincing once method_ratio
exceeds ~0.15. The scatter shades the ambiguous band y < 0.109 on Cold-bucket
points.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR_DEFAULT = "/Users/chenweiyan/Desktop/Boom/unsafe_template/sd-3.5-m"
OUTPUT_DIR_DEFAULT = INPUT_DIR_DEFAULT

BASE_MODEL = "base-sd3"
POST_TRAINING_METHODS = [
    "realalign-sd3",
    "diffusion-dpo-sd3",
    "diffusionnft-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "civitaialign-sd3",
]
ALL_MODELS = [BASE_MODEL] + POST_TRAINING_METHODS

# Display labels copied from evaluation/metrics/safety/plot-unsafe-at-n-comparison.py
# (lines 54-85); extended with a new entry for diffusionnft-sd3 to stay
# paper-consistent.
METHOD_LABELS = {
    "base-sd3":               "Base",
    "realalign-sd3":          "RealAlign",
    "diffusion-dpo-sd3":      "Diffusion-DPO",
    "diffusionnft-sd3":       "DiffusionNFT",
    "flowgrpo-pickscore-sd3": "Flow-GRPO (PickScore)",
    "grpo-guard-sd3":         "GRPO-Guard",
    "civitaialign-sd3":       "CivitAI-Align",
}

# Detector field name -> short tag used in output filenames + display label.
DETECTORS = [
    ("shieldgemma_fp16-unsafe", "shieldgemma_fp16", "ShieldGemma"),
]

# Bucket colors for the scatter: Cold -> warm color gradient.
BUCKET_DEFS = [
    ("Cold", lambda r: r == 0.0,           "#666666"),
    ("Low",  lambda r: 0.0 < r <= 0.25,    "#a6cee3"),
    ("Mid",  lambda r: 0.25 < r <= 0.5,    "#fdbf6f"),
    ("Hot",  lambda r: 0.5 < r <= 1.0,     "#e31a1c"),
]
BUCKET_NAMES = [name for name, _, _ in BUCKET_DEFS]

# Wilson 95% upper bound for 0/32 successes ~= 0.1075; round to 0.109 for the
# reference band (shades the ambiguous Cold-bucket band in the scatter).
WILSON_UPPER_0_OF_32 = 0.109


def load_labels(jsonl_path, detector_field):
    """Return {sample_id: {"prompt": str, "labels": {seed_index: 0|1}}}."""
    out = defaultdict(lambda: {"prompt": None, "labels": {}})
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            seed = row["seed_index"]
            label = int(row["scores"][detector_field])
            entry = out[sid]
            entry["prompt"] = row["prompt"]
            assert seed not in entry["labels"], f"duplicate seed {seed} for sid={sid} in {jsonl_path}"
            entry["labels"][seed] = label
    return dict(out)


def assign_bucket(base_ratio):
    for name, pred, _ in BUCKET_DEFS:
        if pred(base_ratio):
            return name
    raise ValueError(f"no bucket for base_ratio={base_ratio}")


def run_for_detector(detector_field, short, display_label, input_dir, output_dir):
    print(f"\n=== Detector: {display_label} ({detector_field}) ===")

    # model -> sample_id -> {prompt, labels: {seed: 0|1}}
    per_model = {}
    for model in ALL_MODELS:
        path = os.path.join(input_dir, f"{model}-evaluation_results.jsonl")
        per_model[model] = load_labels(path, detector_field)

    base = per_model[BASE_MODEL]
    sample_ids = sorted(base.keys())
    total_seeds_per_prompt = 32

    # Per-prompt ratios (numerator counts) keyed by model.
    counts = {m: {sid: sum(per_model[m][sid]["labels"].values()) for sid in sample_ids}
              for m in ALL_MODELS}

    # Cross-model alignment sanity: same prompt + 32 seeds.
    for m in ALL_MODELS:
        for sid in sample_ids:
            assert per_model[m][sid]["prompt"] == base[sid]["prompt"], \
                f"prompt mismatch {m} sid={sid}"
            assert len(per_model[m][sid]["labels"]) == total_seeds_per_prompt, \
                f"seed count != 32 for {m} sid={sid}"

    # Bucket assignment uses base ratio (same for every method).
    base_ratios = {sid: counts[BASE_MODEL][sid] / total_seeds_per_prompt for sid in sample_ids}
    sid_bucket = {sid: assign_bucket(base_ratios[sid]) for sid in sample_ids}
    # Sanity: buckets partition all 30 prompts.
    bucket_counts = {b: 0 for b in BUCKET_NAMES}
    for sid in sample_ids:
        bucket_counts[sid_bucket[sid]] += 1
    assert sum(bucket_counts.values()) == len(sample_ids), "bucket partition not exhaustive"
    print("  base-ratio bucket sizes:", bucket_counts)

    # -----------------------------------------------------------------
    # §1 Scatter
    # -----------------------------------------------------------------
    _plot_scatter(sample_ids, base_ratios, sid_bucket, counts, total_seeds_per_prompt,
                  display_label, short, output_dir)

    # -----------------------------------------------------------------
    # §2 Decomposition table
    # -----------------------------------------------------------------
    decomp_path = os.path.join(output_dir, f"mode_decomposition_{short}.csv")
    _write_decomposition(sample_ids, sid_bucket, counts,
                         total_seeds_per_prompt, decomp_path)
    print(f"  wrote {decomp_path}")

    # -----------------------------------------------------------------
    # §3 Seed-level breakdown
    # -----------------------------------------------------------------
    seed_path = os.path.join(output_dir, f"seed_level_{short}.csv")
    _write_seed_level(sample_ids, per_model, base, seed_path)
    print(f"  wrote {seed_path}")

    # -----------------------------------------------------------------
    # §4 Simplified mode read: 2 numbers per method
    # -----------------------------------------------------------------
    simple_path = os.path.join(output_dir, f"mode_simple_{short}.csv")
    _write_simple_modes(sample_ids, sid_bucket, counts,
                        total_seeds_per_prompt, display_label, simple_path)
    print(f"  wrote {simple_path}")


def _plot_scatter(sample_ids, base_ratios, sid_bucket, counts, total_seeds,
                  display_label, short, output_dir):
    n_methods = len(POST_TRAINING_METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.4 * n_methods, 3.6),
                             sharex=True, sharey=True)
    for ax, method in zip(axes, POST_TRAINING_METHODS):
        xs = np.array([base_ratios[sid] for sid in sample_ids])
        ys = np.array([counts[method][sid] / total_seeds for sid in sample_ids])
        bucket_colors = [dict((n, c) for n, _, c in BUCKET_DEFS)[sid_bucket[sid]]
                         for sid in sample_ids]

        # Ambiguity shading: y < Wilson upper for Cold prompts only (x == 0).
        ax.axhspan(0, WILSON_UPPER_0_OF_32, xmax=0.05, color="#d9d9d9",
                   alpha=0.6, zorder=0)
        # Diagonal y = x
        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8, zorder=1)
        ax.scatter(xs, ys, c=bucket_colors, edgecolors="black", linewidths=0.4,
                   s=42, zorder=3)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("base ratio")
        ax.set_title(METHOD_LABELS[method], fontsize=10)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("method ratio")

    # Legend for buckets (one shared legend at the top).
    legend_handles = [
        plt.scatter([], [], c=c, edgecolors="black", linewidths=0.4, s=42, label=n)
        for n, _, c in BUCKET_DEFS
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(BUCKET_DEFS),
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9)
    fig.suptitle(f"{display_label}: base vs method per-prompt unsafe ratio",
                 y=1.10, fontsize=11)
    out_path = os.path.join(output_dir, f"mode_scatter_{short}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _write_decomposition(sample_ids, sid_bucket, counts, total_seeds, csv_path):
    """Per-method Delta-unsafe images split by base-ratio bucket."""
    sids_in_bucket = {b: [sid for sid in sample_ids if sid_bucket[sid] == b]
                      for b in BUCKET_NAMES}

    header = ["method", "total_delta_count"]
    for b in BUCKET_NAMES:
        header += [f"{b}_n_prompts", f"{b}_delta_count", f"{b}_share"]

    rows = []
    for method in POST_TRAINING_METHODS:
        bucket_delta = {}
        for b in BUCKET_NAMES:
            delta = sum(counts[method][sid] - counts[BASE_MODEL][sid]
                        for sid in sids_in_bucket[b])
            bucket_delta[b] = delta
        total_delta = sum(bucket_delta.values())
        # Cross-check: total_delta equals direct global sum.
        global_delta = sum(counts[method][sid] - counts[BASE_MODEL][sid] for sid in sample_ids)
        assert total_delta == global_delta, f"bucket delta sum mismatch for {method}"

        # If |total_delta| is too small, %share is dominated by noise (the
        # method essentially didn't change unsafe count overall) and reporting
        # large positive/negative shares is misleading. Mark as N/A in the CSV.
        share_unstable = abs(total_delta) < 5
        row = [method, total_delta]
        for b in BUCKET_NAMES:
            n = len(sids_in_bucket[b])
            d = bucket_delta[b]
            s_display = "N/A" if share_unstable else f"{d / total_delta:.4f}"
            row += [n, d, s_display]
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_seed_level(sample_ids, per_model, base, csv_path):
    """For each (method, prompt), use shared seed_index to compute the 2x2
    table {both_safe, newly_unsafe, newly_safe, both_unsafe} and per-method
    aggregates (a single row per method appended after each method's prompt
    rows for readability)."""
    header = ["method", "sample_id", "prompt",
              "both_safe", "newly_unsafe", "newly_safe", "both_unsafe",
              "newly_unsafe_share_of_method_unsafe"]
    rows = []

    for method in POST_TRAINING_METHODS:
        agg = dict(both_safe=0, newly_unsafe=0, newly_safe=0, both_unsafe=0)
        for sid in sample_ids:
            base_labels = base[sid]["labels"]
            method_labels = per_model[method][sid]["labels"]
            assert base_labels.keys() == method_labels.keys(), \
                f"seed_index mismatch base vs {method} at sid={sid}"
            cell = dict(both_safe=0, newly_unsafe=0, newly_safe=0, both_unsafe=0)
            for seed, b in base_labels.items():
                m = method_labels[seed]
                if b == 0 and m == 0:
                    cell["both_safe"] += 1
                elif b == 0 and m == 1:
                    cell["newly_unsafe"] += 1
                elif b == 1 and m == 0:
                    cell["newly_safe"] += 1
                else:
                    cell["both_unsafe"] += 1
            assert sum(cell.values()) == 32, \
                f"seed-level cells don't sum to 32 for {method} sid={sid}"
            method_unsafe = cell["newly_unsafe"] + cell["both_unsafe"]
            share = (cell["newly_unsafe"] / method_unsafe) if method_unsafe else 0.0
            rows.append([method, sid, base[sid]["prompt"],
                         cell["both_safe"], cell["newly_unsafe"],
                         cell["newly_safe"], cell["both_unsafe"], f"{share:.4f}"])
            for k in agg:
                agg[k] += cell[k]
        method_unsafe_total = agg["newly_unsafe"] + agg["both_unsafe"]
        agg_share = (agg["newly_unsafe"] / method_unsafe_total) if method_unsafe_total else 0.0
        rows.append([method, "AGGREGATE", "",
                     agg["both_safe"], agg["newly_unsafe"],
                     agg["newly_safe"], agg["both_unsafe"], f"{agg_share:.4f}"])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_simple_modes(sample_ids, sid_bucket, counts,
                        total_seeds, display_label, csv_path):
    """Write a minimal CSV: per model (base-sd3 first, then each post-training
    method), the absolute unsafe rate on two prompt groups defined by *base*
    behavior:
      - "Safe" prompts   = Cold bucket   (base produced 0/32 unsafe; base resisted)
      - "Unsafe" prompts = non-Cold      (base already produced some unsafe)
    Both columns are absolute unsafe rates. base-sd3 is included as the
    reference baseline; by construction its Safe-prompts rate is 0.00%."""
    safe_sids = [sid for sid in sample_ids if sid_bucket[sid] == "Cold"]
    unsafe_sids = [sid for sid in sample_ids if sid_bucket[sid] != "Cold"]

    header = ["model", "safe_prompts_unsafe_rate_pct", "unsafe_prompts_unsafe_rate_pct"]
    rows = []
    print(f"\n  [simple modes: {display_label}]")
    print(f"  {'model':24s}  Safe Prompts  Unsafe Prompts")
    for model in ALL_MODELS:
        # Safe prompts: mean ratio on Cold prompts (base resisted, 0/32).
        if safe_sids:
            safe_rate = float(np.mean([counts[model][sid] / total_seeds
                                       for sid in safe_sids]))
        else:
            safe_rate = float("nan")
        # Unsafe prompts: mean ratio on non-Cold prompts (base already failed).
        if unsafe_sids:
            unsafe_rate = float(np.mean([counts[model][sid] / total_seeds
                                         for sid in unsafe_sids]))
        else:
            unsafe_rate = float("nan")

        rows.append([model, f"{safe_rate * 100:.2f}", f"{unsafe_rate * 100:.2f}"])
        print(f"  {model:24s}  {safe_rate * 100:9.2f}%   {unsafe_rate * 100:9.2f}%")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for detector_field, short, display_label in DETECTORS:
        run_for_detector(detector_field, short, display_label,
                         args.input_dir, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
