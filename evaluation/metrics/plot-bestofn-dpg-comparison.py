"""Plot the DPG-Bench Best-of-N curve comparing five SD-3.5-M methods.

Style is kept identical to plot-bestofn-wise-comparison.py (same rcParams,
linear x-axis with ticks 4..32, grid, legend, figure size, save args) so
the DPG and WISE BoN figures look like a matched set. The ONE intentional
difference: each method gets a distinct marker (not all circles) so the
five overlaid curves stay tellable apart in print/grayscale.

Reads, for each method:
  ${base_root}/<method>/dpg_bench/bestofn/csv/dpg-score_curve.csv
(written by aggregate-bestofn.py's _aggregate_dpg). Curve values are
0..1; DPG-Bench is conventionally reported on 0..100, so the y-axis is
scaled by 100 to match aggregate-bestofn.py's printed numbers.

Usage:
  python evaluation/metrics/plot-bestofn-dpg-comparison.py --out_dir ./plots
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
# post-trained methods. Same palette family as plot-bestofn-wise-comparison.py.
METHOD_COLORS = {
    "base-sd3":               "#a8a8a8",
    "flowgrpo-pickscore-sd3": "#8da0cb",
    "grpo-guard-sd3":         "#66c2a5",
    "diffusion-dpo-sd3":      "#fc8d62",
    "realalign-sd3":          "#e78ac3",
}
# The single intentional departure from the WISE plot's style: a distinct
# marker per method so the five overlaid curves stay distinguishable
# without relying on color alone.
METHOD_MARKERS = {
    "base-sd3":               "o",
    "flowgrpo-pickscore-sd3": "s",
    "grpo-guard-sd3":         "^",
    "diffusion-dpo-sd3":      "D",
    "realalign-sd3":          "v",
}
METHOD_LINESTYLES = {
    "base-sd3":               "--",
    "flowgrpo-pickscore-sd3": "-",
    "grpo-guard-sd3":         "-",
    "diffusion-dpo-sd3":      "-",
    "realalign-sd3":          "-",
}

DATASET = "dpg_bench"
CSV_NAME = "dpg-score_curve.csv"
OUTPUT_STEM = "dpg-score"
Y_SCALE = 100.0  # csv stores 0..1; DPG-Bench is reported on 0..100.


def load_curve(base_root, method, csv_name):
    path = os.path.join(base_root, method, DATASET, "bestofn", "csv", csv_name)
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
            ns, ys * Y_SCALE,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel(f"DPG-Score (0–100) — {label}")
    ax.set_title(f"DPG-Bench Best-of-N: {label}")
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
    plot_one("Overall", CSV_NAME, OUTPUT_STEM, args.base_root, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot the DPG-Bench Best-of-N curve, 5 SD-3.5-M methods overlaid.",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_plots",
        help="Directory to write dpg-score.png + .pdf into.",
    )
    main(ap.parse_args())
