"""Plot BoN curves comparing base vs post-training methods, one figure per metric.

Reads ${base_root}/<method>/<dataset>/bestofn/csv/<metric>_curve.csv
and saves a separate PNG (+ PDF) per metric into --out_dir, with the
four methods (base, dpo, inpo, spo) overlaid on each plot.

Output files (in --out_dir):
  pickscore.png / .pdf
  hpsv3.png / .pdf
  deqa.png / .pdf
  aesthetic.png / .pdf
  ocr_continuous.png / .pdf

Usage:
  python evaluation/metrics/plot-bestofn-comparison.py --out_dir ./plots
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
# Pastel palette: muted neutral gray for the baseline + ColorBrewer Set2 hues
# for the three HPA variants. Light enough to feel soft, distinct enough to
# tell apart even when overlaid.
METHOD_COLORS = {
    "base": "#a8a8a8",
    "dpo":  "#8da0cb",
    "inpo": "#66c2a5",
    "spo":  "#fc8d62",
}
METHOD_LINESTYLES = {"base": "--", "dpo": "-", "inpo": "-", "spo": "-"}

# (display_label, dataset_subdir, csv_filename, output_stem)
METRICS = [
    ("PickScore",        "drawbench-unique", "pickscore_curve.csv",        "pickscore"),
    ("HPSv3",            "drawbench-unique", "hpsv3_curve.csv",            "hpsv3"),
    ("DeQA",             "drawbench-unique", "deqa_curve.csv",             "deqa"),
    ("Aesthetic",        "drawbench-unique", "aesthetic_curve.csv",        "aesthetic"),
    ("OCR (continuous)", "ocr",              "ocr_continuous_curve.csv",   "ocr_continuous"),
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


def plot_one(label, dataset, csv_name, stem, base_root, out_dir):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)

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

    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel(label)
    ax.set_title(f"Best-of-N: {label}")
    ax.set_xticks(np.arange(4, 33, 4))
    ax.set_xlim(0, 33)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.9,
        edgecolor='#dddddd',
        fontsize=11,
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
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 4.5,
        "font.family": "DejaVu Sans",
    })

    os.makedirs(args.out_dir, exist_ok=True)
    for label, dataset, csv_name, stem in METRICS:
        plot_one(label, dataset, csv_name, stem, args.base_root, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot BoN comparison, one figure per metric.",
    )
    ap.add_argument(
        "--base_root", default=DEFAULT_BASE_ROOT,
        help=f"default: {DEFAULT_BASE_ROOT}",
    )
    ap.add_argument(
        "--out_dir", default="bestofn_plots",
        help="Directory to write per-metric PNG + PDF files into.",
    )
    main(ap.parse_args())
