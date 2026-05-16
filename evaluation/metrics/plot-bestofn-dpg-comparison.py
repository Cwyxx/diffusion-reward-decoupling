"""Plot the DPG-Bench Best-of-N curve comparing five SD-3.5-M methods.

Follows plot-bestofn-comparison.py, but: (a) targets the SD-3.5-M family
and its 5 methods, (b) plots only the dpg_bench dpg-score curve, and
(c) uses a distinct marker per method (not all circles) so overlaid
curves stay tellable apart in print/grayscale.

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
# Muted neutral gray for the baseline + ColorBrewer Set2 hues for the
# four post-training variants.
METHOD_COLORS = {
    "base-sd3":               "#a8a8a8",
    "flowgrpo-pickscore-sd3": "#8da0cb",
    "grpo-guard-sd3":         "#66c2a5",
    "diffusion-dpo-sd3":      "#fc8d62",
    "realalign-sd3":          "#e78ac3",
}
# Distinct marker per method so the five overlaid curves stay
# distinguishable without relying on color alone.
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


def load_curve(base_root, method):
    path = os.path.join(base_root, method, DATASET, "bestofn", "csv", CSV_NAME)
    ns, ys = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ns.append(int(row[0]))
            ys.append(float(row[1]))
    return np.array(ns), np.array(ys)


def main(args):
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
        "font.family": "DejaVu Sans",
    })
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    max_n = 0
    for method in METHODS:
        try:
            ns, ys = load_curve(args.base_root, method)
        except FileNotFoundError as e:
            print(f"[warn] missing {e.filename}", file=sys.stderr)
            continue
        max_n = max(max_n, int(ns.max()))
        ax.plot(
            ns, ys * Y_SCALE,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
            markevery=max(1, len(ns) // 12),
        )

    if max_n == 0:
        sys.exit(f"No dpg-score curves found under {args.base_root}/<method>/"
                 f"{DATASET}/bestofn/csv/{CSV_NAME}")

    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel("DPG-Score (0–100)")
    ax.set_title("Best-of-N: DPG-Bench")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, max_n)
    ax.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#dddddd",
        fontsize=10,
    )

    png_path = os.path.join(args.out_dir, f"{OUTPUT_STEM}.png")
    pdf_path = os.path.join(args.out_dir, f"{OUTPUT_STEM}.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path} and {pdf_path}")


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
