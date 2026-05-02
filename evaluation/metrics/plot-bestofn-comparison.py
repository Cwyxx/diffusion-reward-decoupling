"""Compare BoN curves across methods, one panel per metric.

Reads ${base_root}/<method>/<dataset>/bestofn/csv/<metric>_curve.csv
and plots a 2x3 panel comparing base vs post-training methods on
PickScore, HPSv3, DeQA, Aesthetic, and OCR (continuous mean-of-max).
The 6th cell holds the shared legend.

Usage:
  python evaluation/metrics/plot-bestofn-comparison.py --out comparison.png
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
    "diffusion-reward-decoupling/bestofn-eval/sd-v1-5"
)

METHODS = ["base", "dpo", "inpo", "spo"]
METHOD_LABELS = {"base": "Base", "dpo": "DPO", "inpo": "InPO", "spo": "SPO"}
# Neutral gray for base, distinct hues for the three HPA variants.
METHOD_COLORS = {
    "base": "#888888",
    "dpo":  "#1f77b4",
    "inpo": "#2ca02c",
    "spo":  "#d62728",
}
METHOD_LINESTYLES = {"base": "--", "dpo": "-", "inpo": "-", "spo": "-"}

# (display_label, dataset_subdir, csv_filename)
METRICS = [
    ("PickScore",        "drawbench-unique", "pickscore_curve.csv"),
    ("HPSv3",            "drawbench-unique", "hpsv3_curve.csv"),
    ("DeQA",             "drawbench-unique", "deqa_curve.csv"),
    ("Aesthetic",        "drawbench-unique", "aesthetic_curve.csv"),
    ("OCR (continuous)", "ocr",              "ocr_continuous_curve.csv"),
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


def main(args):
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.9,
        "lines.markersize": 4,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    for i, (label, dataset, csv_name) in enumerate(METRICS):
        ax = axes_flat[i]
        for method in METHODS:
            try:
                ns, ys = load_curve(args.base_root, method, dataset, csv_name)
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
        ax.set_title(label)
        ax.set_xticks(np.arange(4, 33, 4))
        ax.set_xlim(0, 33)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

    # Sixth cell: shared legend.
    legend_ax = axes_flat[-1]
    legend_ax.axis('off')
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels,
        loc='center',
        frameon=False,
        fontsize=14,
        title="Method",
        title_fontsize=14,
        handlelength=2.2,
        labelspacing=1.0,
    )

    fig.suptitle(
        "Best-of-N curves: Base vs. post-training methods (SD-v1.5)",
        fontsize=14, fontweight='bold', y=1.02,
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"saved {out_path} and {pdf_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare BoN curves across methods, one panel per metric.",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out", default="bestofn_comparison.png",
        help="Output PNG path (a .pdf is also written next to it).",
    )
    main(ap.parse_args())
