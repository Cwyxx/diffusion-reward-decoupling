"""Plot GenEval Best-of-N curves comparing 5 SD-3.5-M methods, one figure per tag.

Reads ${base_root}/<method>/geneval/bestofn/csv/geneval_<TAG>_curve.csv
(and geneval_curve.csv for the macro-averaged Overall) and saves a separate
PNG (+ PDF) per GenEval dimension into --out_dir, with the five methods
overlaid on each plot.

The per-tag and overall CSVs are produced by
evaluation/metrics/aggregate-bestofn.py:_aggregate_geneval. Tags follow the
official GenEval grouping (GENEVAL_TAGS at aggregate-bestofn.py:66); Overall
is the macro-average over the 6 tags at each N (aggregate-bestofn.py:298-301).

Output files (in --out_dir):
  single_object.png / .pdf
  two_object.png / .pdf
  counting.png / .pdf
  colors.png / .pdf
  position.png / .pdf
  color_attr.png / .pdf
  overall.png / .pdf

Usage:
  python evaluation/metrics/plot-bestofn-geneval-comparison.py --out_dir ./geneval_plots
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
# Muted neutral gray for the baseline + ColorBrewer Set2 hues for the four
# post-trained methods. Same palette as plot-bestofn-wise-comparison.py so
# methods keep a consistent color across benchmarks.
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

# (display_label, csv_filename, output_stem). csv_filename matches the names
# written by aggregate-bestofn.py:_aggregate_geneval.
CATEGORIES = [
    ("Single object",    "geneval_single_object_curve.csv", "single_object"),
    ("Two object",       "geneval_two_object_curve.csv",    "two_object"),
    ("Counting",         "geneval_counting_curve.csv",      "counting"),
    ("Colors",           "geneval_colors_curve.csv",        "colors"),
    ("Position",         "geneval_position_curve.csv",      "position"),
    ("Color attribution", "geneval_color_attr_curve.csv",   "color_attr"),
    ("Overall (macro-avg)", "geneval_curve.csv",            "overall"),
]


def load_curve(base_root, method, csv_name):
    path = os.path.join(base_root, method, "geneval", "bestofn", "csv", csv_name)
    ns, ys = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ns.append(int(row[0]))
            ys.append(float(row[1]))
    return np.array(ns), np.array(ys)


def plot_one(label, csv_name, stem, base_root, out_dir):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)

    for method in METHODS:
        try:
            ns, ys = load_curve(base_root, method, csv_name)
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

    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel(f"GenEval pass@N — {label}")
    ax.set_title(f"GenEval Best-of-N: {label}")
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
    for label, csv_name, stem in CATEGORIES:
        plot_one(label, csv_name, stem, args.base_root, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot GenEval BoN comparison across 5 SD-3.5-M methods, "
                    "one figure per tag (+ Overall macro-avg).",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_geneval_plots",
        help="Directory to write per-tag PNG + PDF files into.",
    )
    main(ap.parse_args())
