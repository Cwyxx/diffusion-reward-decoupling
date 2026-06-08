"""Plot AnyText OCR Best-of-N curves across SD-3.5-M methods, one figure per metric.

Reads ${base_root}/<method>/<dataset>/bestofn/csv/<metric>_curve.csv (produced by
evaluation/metrics/core/aggregate-bestofn.py) and saves a separate PNG (+ PDF) per
metric into --out_dir, with the methods overlaid on each plot.

For the AnyText datasets (dataset=anytext-en / anytext-zh) the aggregator writes
three curves (see aggregate-bestofn.py main loop + BINARY_METRICS / DUAL_METRICS):
  anytext-senacc_curve.csv             pass@N, threshold 1.0
                                       ("at least one of N seeds got EVERY text
                                       line exactly right")
  anytext-senacc_continuous_curve.csv  mean over prompts of max over N seeds
                                       ("fraction of text lines the best seed got
                                       right" -- captures sub-threshold gains)
  anytext-ned_curve.csv                mean over prompts of max over N seeds of NED
                                       (soft edit-distance recall, 1 - lev/maxlen)

All three are higher-is-better in [0, 1]. Scoring is position-free (full-image
DuGuang OCR, not GT-polygon crops), so these are NOT the official AnyText
benchmark numbers -- they rank methods on legibility for plain T2I models. See
evaluation/benchmarks/AnyText/anytext_scorer.py.

Output files (in --out_dir):
  senacc.png / .pdf
  senacc_continuous.png / .pdf
  ned.png / .pdf

Usage:
  python evaluation/metrics/capability/plot-bestofn-anytext-comparison.py --out_dir ./anytext_plots
  python evaluation/metrics/capability/plot-bestofn-anytext-comparison.py --dataset anytext-zh --out_dir ./anytext_zh_plots
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

# All SD-3.5-M methods run through run-bestofn.sh. Methods without anytext
# results are skipped with a warning (load_curve catches FileNotFoundError).
METHODS = [
    "base-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
    "diffusionnft-sd3",
    "civitaialign-sd3",
]
METHOD_LABELS = {
    "base-sd3":               "Base",
    "flowgrpo-pickscore-sd3": "Flow-GRPO (PickScore)",
    "grpo-guard-sd3":         "GRPO-Guard",
    "diffusion-dpo-sd3":      "Diffusion-DPO",
    "realalign-sd3":          "RealAlign",
    "diffusionnft-sd3":       "DiffusionNFT",
    "civitaialign-sd3":       "CivitaiAlign",
}
# Muted neutral gray for the baseline + ColorBrewer Set2 hues for the
# post-trained methods. The first five match plot-bestofn-geneval-comparison.py
# / plot-bestofn-wise-comparison.py so methods keep a consistent color across
# benchmarks; the last two extend the same Set2 palette.
METHOD_COLORS = {
    "base-sd3":               "#a8a8a8",
    "flowgrpo-pickscore-sd3": "#8da0cb",
    "grpo-guard-sd3":         "#66c2a5",
    "diffusion-dpo-sd3":      "#fc8d62",
    "realalign-sd3":          "#e78ac3",
    "diffusionnft-sd3":       "#a6d854",
    "civitaialign-sd3":       "#ffd92f",
}
METHOD_LINESTYLES = {
    "base-sd3":               "--",
    "flowgrpo-pickscore-sd3": "-",
    "grpo-guard-sd3":         "-",
    "diffusion-dpo-sd3":      "-",
    "realalign-sd3":          "-",
    "diffusionnft-sd3":       "-",
    "civitaialign-sd3":       "-",
}

# (display_label, csv_filename, output_stem, ylabel). csv_filename matches the
# names written by aggregate-bestofn.py for the anytext-ocr metric.
METRICS = [
    ("Sen.ACC (pass@N)",      "anytext-senacc_curve.csv",            "senacc",
     "Sen.ACC pass@N (all lines exact)"),
    ("Sen.ACC (continuous)",  "anytext-senacc_continuous_curve.csv", "senacc_continuous",
     "Sen.ACC mean-of-max (line fraction)"),
    ("NED",                   "anytext-ned_curve.csv",               "ned",
     "NED mean-of-max (1 - lev/maxlen)"),
]


def load_curve(base_root, method, dataset, csv_name):
    path = os.path.join(base_root, method, dataset, "bestofn", "csv", csv_name)
    ns, ys = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ns.append(int(row[0]))
            ys.append(float(row[1]))
    return np.array(ns), np.array(ys)


def plot_one(label, csv_name, stem, ylabel, dataset, base_root, out_dir):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)

    plotted = 0
    for method in METHODS:
        try:
            ns, ys = load_curve(base_root, method, dataset, csv_name)
        except FileNotFoundError as e:
            print(f"[warn] missing {e.filename}", file=sys.stderr)
            continue
        ax.plot(
            ns, ys,
            marker='o',
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
        plotted += 1

    if plotted == 0:
        print(f"[warn] no curves found for {csv_name}; skipping {stem}", file=sys.stderr)
        plt.close(fig)
        return

    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"AnyText ({dataset}) Best-of-N: {label}")
    ax.set_xticks(np.arange(4, 33, 4))
    ax.set_xlim(0, 33)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='#dddddd',
        fontsize=10,
    )

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
    for label, csv_name, stem, ylabel in METRICS:
        plot_one(label, csv_name, stem, ylabel, args.dataset, args.base_root, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot AnyText OCR BoN comparison across SD-3.5-M methods, "
                    "one figure per metric (Sen.ACC pass@N, Sen.ACC continuous, NED).",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--dataset", default="anytext-en", choices=["anytext-en", "anytext-zh"],
        help="Which AnyText dataset subdir to read (default: anytext-en).",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_anytext_plots",
        help="Directory to write per-metric PNG + PDF files into.",
    )
    main(ap.parse_args())
