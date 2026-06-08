"""Plot unsafe-rate-vs-N curves for Responsible-AI safety eval.

Writes a separate figure per (classifier, dataset) pair, overlaying all
methods so you can see whether a given training method changes the safety
behaviour on a given adversarial prompt source.

Two modes (--mode, default average), both produced by
evaluation/metrics/core/aggregate-bestofn.py from a 0/1 unsafe flag:

  average (default): unsafe@average-N -- mean over prompts of the fraction
    of the first N seeds flagged unsafe. The expected probability that any
    single generated image is unsafe. Budget-independent (flat in N in
    expectation), so it compares per-image unsafe rates across methods
    fairly. Reads <metric>_average_curve.csv. The reportable headline is the
    value at the largest N (e.g. N=32).

  union: unsafe@N -- pass@N with threshold=1.0, the fraction of prompts for
    which AT LEAST ONE of the first N seeds is unsafe. Monotonically rising;
    a worst-case / attack-success view. Reads <metric>_curve.csv.

Reads ${base_root}/<method>/<dataset>/bestofn/csv/<stem>_curve.csv.

Output files (in --out_dir), with an "_average" suffix in average mode:
  sd-safety-flag_template[_average].{png,pdf}        (+ _4chan, _lexica)
  shieldgemma_fp32-unsafe_template[_average].{png,pdf}    (+ _4chan, _lexica)

Usage:
  python evaluation/metrics/safety/plot-unsafe-at-n-comparison.py --out_dir ./unsafe_plots
  python evaluation/metrics/safety/plot-unsafe-at-n-comparison.py --mode union --out_dir ./unsafe_plots
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_ROOT = (
    "/data_center/data2/dataset/chenwy/21164-data/"
    "diffusion-reward-decoupling/bestofn-eval/sd-3.5-m"
)

METHODS = [
    "base-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
]
METHOD_LABELS = {
    "base-sd3":               "Base",
    "flowgrpo-pickscore-sd3": "Flow-GRPO (PickScore)",
    "grpo-guard-sd3":         "GRPO-Guard",
    "diffusion-dpo-sd3":      "Diffusion-DPO",
    "realalign-sd3":          "RealAlign",
}
# Mirror plot-bestofn-wise-comparison.py palette so figures in the paper
# share the same method colors across benchmarks.
METHOD_COLORS = {
    "base-sd3":               "#a8a8a8",
    "flowgrpo-pickscore-sd3": "#8da0cb",
    "grpo-guard-sd3":         "#66c2a5",
    "diffusion-dpo-sd3":      "#fc8d62",
    "realalign-sd3":          "#e78ac3",
}
METHOD_LINESTYLES = {
    "base-sd3":               "--",
    "flowgrpo-pickscore-sd3": "-",
    "grpo-guard-sd3":         "-",
    "diffusion-dpo-sd3":      "-",
    "realalign-sd3":          "-",
}
# Distinct marker per method so the overlaid curves stay distinguishable
# without relying on color alone. Matches plot-bestofn-dpg-comparison.py.
METHOD_MARKERS = {
    "base-sd3":               "o",
    "flowgrpo-pickscore-sd3": "s",
    "grpo-guard-sd3":         "^",
    "diffusion-dpo-sd3":      "D",
    "realalign-sd3":          "v",
}

# (dataset_dir matching evaluation/benchmarks/ResponsibleAI, short filename stem)
DATASETS = [
    ("unsafe_template", "template"),
    ("unsafe_4chan",    "4chan"),
    ("unsafe_lexica",   "lexica"),
]

# (metric key written by aggregate-bestofn.py, display label for axis/title)
CLASSIFIERS = [
    ("sd-safety-flag",     "SD safety-checker"),
    ("shieldgemma_fp32-unsafe", "ShieldGemma"),
]


def load_curve(base_root, method, dataset, metric, mode):
    # union  -> unsafe@N           (pass@N):    {metric}_curve.csv
    # average-> unsafe@average-N (mean rate): {metric}_average_curve.csv
    stem = f"{metric}_average" if mode == "average" else metric
    path = os.path.join(
        base_root, method, dataset, "bestofn", "csv", f"{stem}_curve.csv"
    )
    ns, ys = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ns.append(int(row[0]))
            ys.append(float(row[1]))
    return np.array(ns), np.array(ys)


def plot_one(metric, metric_label, dataset, dataset_stem, base_root, out_dir, mode):
    loaded = []
    for method in METHODS:
        try:
            ns, ys = load_curve(base_root, method, dataset, metric, mode)
        except FileNotFoundError as e:
            print(f"[warn] missing {e.filename}", file=sys.stderr)
            continue
        loaded.append((method, ns, ys))

    if not loaded:
        print(f"[skip] no data for {metric} on {dataset}", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    n_max = max(int(ns.max()) for _, ns, _ in loaded)

    for method, ns, ys in loaded:
        ax.plot(
            ns, ys,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

    if n_max >= 16:
        step = 4
    elif n_max >= 8:
        step = 2
    else:
        step = 1
    ax.set_xticks(np.arange(step, n_max + 1, step))
    ax.set_xlim(0, n_max + 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("N (samples per prompt)")
    if mode == "average":
        ax.set_ylabel(f"unsafe@average-N ({metric_label})")
    else:
        ax.set_ylabel(f"unsafe@N ({metric_label})")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='#dddddd',
        fontsize=10,
    )

    suffix = "_average" if mode == "average" else ""
    stem = f"{metric}_{dataset_stem}{suffix}"
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {png_path} and {pdf_path}")


def main(args):
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 4.5,
        "font.family": "DejaVu Sans",
    })

    os.makedirs(args.out_dir, exist_ok=True)
    for metric, metric_label in CLASSIFIERS:
        for dataset, dataset_stem in DATASETS:
            plot_one(metric, metric_label, dataset, dataset_stem,
                     args.base_root, args.out_dir, args.mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot unsafe@N curves, one figure per (classifier, dataset).",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_unsafe_plots",
        help="Directory to write per-(classifier,dataset) PNG + PDF files into.",
    )
    ap.add_argument(
        "--mode", choices=["average", "union"], default="average",
        help="average: unsafe@average-N (mean per-image rate, budget-independent, "
             "reads {metric}_average_curve.csv). union: unsafe@N pass@N (reads "
             "{metric}_curve.csv). default: average.",
    )
    main(ap.parse_args())
