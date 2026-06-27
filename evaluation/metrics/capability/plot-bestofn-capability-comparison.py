"""Plot a 1x4 capability overview of Best-of-N curves for the paper.

One subplot per capability axis, styled identically to
plot-bestofn-wise-comparison.py's plot_panels (portrait panels, categorical
x-axis with EQUAL marker spacing, per-method color/marker, legend inside the
GenEval subplot, title fontsize 15, y formatted to 2 decimals):

  1. Human Preference  -> drawbench-unique / pickscore_curve.csv       (PickScore)
  2. GenEval           -> geneval          / geneval_curve.csv        (macro-avg)
  3. DPG-Bench         -> dpg_bench        / dpg-score-mplug_curve.csv (mPLUG, x100)
  4. WISE              -> wise             / wise_curve.csv            (weighted)

Each curve is read from
  ${base_root}/<method>/<dataset>/bestofn/csv/<csv_name>
(produced by evaluation/metrics/core/aggregate-bestofn.py). Methods without a
given curve are skipped with a warning, so methods missing from a benchmark
(e.g. DPG has no DiffusionNFT/CivitaiAlign) just drop out of that panel.

Output files (in --out_dir):
  capability_panels.png / .pdf

Usage:
  python evaluation/metrics/capability/plot-bestofn-capability-comparison.py --out_dir ./capability_plots
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DEFAULT_BASE_ROOT = (
    "/data_center/data2/dataset/chenwy/21164-data/"
    "diffusion-reward-decoupling/bestofn-eval/sd-3.5-m"
)

METHODS = [
    "base-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusionnft-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
    "civitaialign-sd3",
    "gardo-pickscore-sd3",
    "flow-opd-sd3",
]
METHOD_LABELS = {
    "base-sd3":               "Base",
    "flowgrpo-pickscore-sd3": "Flow-GRPO",
    "grpo-guard-sd3":         "GRPO-Guard",
    "diffusionnft-sd3":       "DiffusionNFT",
    "diffusion-dpo-sd3":      "Diffusion-DPO",
    "realalign-sd3":          "RealAlign",
    "civitaialign-sd3":       "CivitaiAlign",
    "gardo-pickscore-sd3":    "GARDO-PickScore",
    "flow-opd-sd3":           "Flow-OPD",
}
METHOD_COLORS = {
    "base-sd3":               "#f57c6e",
    "flowgrpo-pickscore-sd3": "#f2b56f",
    "grpo-guard-sd3":         "#fae693",
    "diffusionnft-sd3":       "#84c3b7",
    "diffusion-dpo-sd3":      "#88d8db",
    "realalign-sd3":          "#71b7ed",
    "civitaialign-sd3":       "#b8aeeb",
    "gardo-pickscore-sd3":    "#eaa7cd",
    "flow-opd-sd3":           "#c8b18c",
}
METHOD_LINESTYLES = {
    "base-sd3":               "--",
    "flowgrpo-pickscore-sd3": "-",
    "grpo-guard-sd3":         "-",
    "diffusionnft-sd3":       "-",
    "diffusion-dpo-sd3":      "-",
    "realalign-sd3":          "-",
    "civitaialign-sd3":       "-",
    "gardo-pickscore-sd3":    "-",
    "flow-opd-sd3":           "-",
}
METHOD_MARKERS = {
    "base-sd3":               "o",
    "flowgrpo-pickscore-sd3": "s",
    "grpo-guard-sd3":         "^",
    "diffusionnft-sd3":       "D",
    "diffusion-dpo-sd3":      "v",
    "realalign-sd3":          "P",
    "civitaialign-sd3":       "X",
    "gardo-pickscore-sd3":    "*",
    "flow-opd-sd3":           "h",
}

# N values that get a marker on the curve, laid out at EQUAL spacing along the
# x-axis (categorical positions), regardless of their numeric value.
PLOT_NS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
# Subset of PLOT_NS that get a tick label.
X_TICKS = [1, 8, 16, 24, 32]
# Map each N to its equally-spaced index position.
N_TO_POS = {n: i for i, n in enumerate(PLOT_NS)}

# (title, dataset_subdir, csv_filename, y_scale, y_fmt). One per panel, left to
# right. y_scale rescales the stored 0..1 value (DPG-Bench is reported 0..100).
PANELS = [
    ("Human Preference", "drawbench-unique", "pickscore_curve.csv",       1.0,   "%.2f"),
    ("GenEval",          "geneval",          "geneval_curve.csv",        1.0,   "%.2f"),
    ("DPG-Bench",        "dpg_bench",         "dpg-score-mplug_curve.csv", 100.0, "%.1f"),
    ("WISE",             "wise",             "wise_curve.csv",           1.0,   "%.2f"),
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


def plot_panels(panels, base_root, out_dir, stem="capability_panels"):
    """Plot the given panels in a 1xN row with the legend inside the third
    subplot, for inclusion in the paper."""
    nrow, ncol = 1, len(panels)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(3.5 * ncol, 4.0 * nrow),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, (label, dataset, csv_name, y_scale, y_fmt)) in enumerate(zip(axes, panels)):
        for method in METHODS:
            try:
                ns, ys = load_curve(base_root, method, dataset, csv_name)
            except FileNotFoundError as e:
                print(f"[warn] missing {e.filename}", file=sys.stderr)
                continue
            keep = np.isin(ns, PLOT_NS)
            ns, ys = ns[keep], ys[keep]
            xs = [N_TO_POS[n] for n in ns]
            ax.plot(
                xs, ys * y_scale,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                label=METHOD_LABELS[method],
                zorder=5 if method == "base-sd3" else 3,
            )

        ax.set_title(label, fontsize=15)
        ax.set_box_aspect(1)            # square subplot
        ax.set_xticks([N_TO_POS[t] for t in X_TICKS])
        ax.set_xticklabels([str(t) for t in X_TICKS])
        ax.set_xlim(-0.5, len(PLOT_NS) - 0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.set_xlabel("Number of Samples N")  # 1 row -> every panel is bottom row
        if i == 0:                            # left column
            ax.set_ylabel("Best@N")

    # Legend on the GenEval panel (index 1): a panel with the full method set,
    # since DPG-Bench may be missing some methods and would give a partial legend.
    axes[1].legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#dddddd",
        fontsize=11,
    )

    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path} and {pdf_path}")


def main(args):
    plt.rcParams.update({
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 15,
        "axes.titleweight": "normal",
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
    })

    os.makedirs(args.out_dir, exist_ok=True)
    plot_panels(PANELS, args.base_root, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot a 1x4 capability overview (Human Preference, GenEval, "
                    "DPG-Bench, WISE) of BoN curves across "
                    "SD-3.5-M methods, styled like the WISE/GenEval panels.",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_capability_plots",
        help="Directory to write capability_panels.png + .pdf into.",
    )
    main(ap.parse_args())
