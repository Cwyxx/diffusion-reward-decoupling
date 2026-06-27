"""Plot GenEval Best-of-N curves comparing SD-3.5-M methods, one figure per tag.

Reads ${base_root}/<method>/geneval/bestofn/csv/geneval_<TAG>_curve.csv
(and geneval_curve.csv for the macro-averaged Overall) and saves a separate
PNG (+ PDF) per GenEval dimension into --out_dir, with the methods overlaid
on each plot.

The per-tag and overall CSVs are produced by
evaluation/metrics/core/aggregate-bestofn.py:_aggregate_geneval. Tags follow the
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
  geneval_panels.png / .pdf   (the 6 per-tag curves in a 2x3 grid, for the paper)

Usage:
  python evaluation/metrics/capability/plot-bestofn-geneval-comparison.py --out_dir ./geneval_plots
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

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
# Same palette as plot-bestofn-anytext-comparison.py /
# plot-bestofn-wise-comparison.py so methods keep a consistent color across
# the BoN figures.
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

# The 6 per-tag panels laid out in a 2x3 grid for the paper.
PANEL_CATEGORIES = [
    ("Single object",     "geneval_single_object_curve.csv", "single_object"),
    ("Two object",        "geneval_two_object_curve.csv",    "two_object"),
    ("Counting",          "geneval_counting_curve.csv",      "counting"),
    ("Colors",            "geneval_colors_curve.csv",        "colors"),
    ("Position",          "geneval_position_curve.csv",      "position"),
    ("Color attribution", "geneval_color_attr_curve.csv",    "color_attr"),
]

# Tag order + display labels for the GenEval-official N=4 summary table.
# Matches GENEVAL_TAGS in aggregate-bestofn.py:66 and the official
# summary_scores.py per-task breakdown.
GENEVAL_TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
TAG_LABELS = {
    "single_object": "Single object",
    "two_object":    "Two object",
    "counting":      "Counting",
    "colors":        "Colors",
    "position":      "Position",
    "color_attr":    "Color attr.",
}


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


def compute_official_n4(base_root, method, n_seeds=4):
    """Replicate geneval-official summary_scores.py at N=n_seeds (default 4).

    Loads ${base_root}/<method>/geneval/evaluation_results.jsonl, restricts
    each prompt to seed_index 0..n_seeds-1, then computes per-tag mean over
    all (prompt × seed) images and Overall as the macro-avg of the 6 tags.

    Returns {"single_object": float, ..., "color_attr": float, "overall": float}.
    Raises FileNotFoundError if the results file is missing, ValueError if
    any (prompt, seed_index) in [0, n_seeds) is unscored for a tag.
    """
    results_path = os.path.join(base_root, method, "geneval", "evaluation_results.jsonl")
    grouped = defaultdict(lambda: defaultdict(dict))  # tag -> sid -> {seed_idx: score}
    with open(results_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if "geneval" not in r["scores"]:
                continue
            tag = (r.get("metadata") or {}).get("tag")
            if tag not in GENEVAL_TAGS:
                continue
            seed_idx = r.get("seed_index", 0)
            if seed_idx >= n_seeds:
                continue
            grouped[tag][r["sample_id"]][seed_idx] = r["scores"]["geneval"]

    out = {}
    per_tag_scores = []
    for tag in GENEVAL_TAGS:
        sid_map = grouped.get(tag, {})
        if not sid_map:
            raise ValueError(f"{method}: no rows with metadata.tag={tag!r}")
        sample_ids = sorted(sid_map.keys())
        mat = np.full((len(sample_ids), n_seeds), np.nan, dtype=float)
        for i, sid in enumerate(sample_ids):
            for s, v in sid_map[sid].items():
                mat[i, s] = v
        if np.isnan(mat).any():
            n_missing = int(np.isnan(mat).sum())
            raise ValueError(
                f"{method}/{tag}: {n_missing} (sample_id, seed_index) entries "
                f"in seeds 0..{n_seeds-1} are unscored."
            )
        score = float(mat.mean())
        out[tag] = score
        per_tag_scores.append(score)
    out["overall"] = float(np.mean(per_tag_scores))
    return out


def print_official_n4_markdown(base_root, n_seeds=4):
    """Print a markdown method × {6 tags + Overall} table to stdout."""
    rows = {}
    for method in METHODS:
        try:
            rows[method] = compute_official_n4(base_root, method, n_seeds=n_seeds)
        except FileNotFoundError as e:
            print(f"[warn] missing {e.filename}", file=sys.stderr)
            rows[method] = None

    headers = ["Method"] + [TAG_LABELS[t] for t in GENEVAL_TAGS] + ["Overall"]
    print()
    print(f"GenEval-official score @ N={n_seeds} (seed_index 0..{n_seeds-1}, "
          f"per-image mean per tag, macro-avg over 6 tags)")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for method in METHODS:
        scores = rows[method]
        label = METHOD_LABELS[method]
        if scores is None:
            cells = ["—"] * (len(GENEVAL_TAGS) + 1)
        else:
            cells = [f"{scores[t]:.2%}" for t in GENEVAL_TAGS] + [f"{scores['overall']:.2%}"]
        print("| " + " | ".join([label] + cells) + " |")
    print()


def plot_one(label, csv_name, stem, base_root, out_dir):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)

    for method in METHODS:
        try:
            ns, ys = load_curve(base_root, method, csv_name)
        except FileNotFoundError as e:
            print(f"[warn] missing {e.filename}", file=sys.stderr)
            continue
        keep = np.isin(ns, PLOT_NS)
        ns, ys = ns[keep], ys[keep]
        xs = [N_TO_POS[n] for n in ns]
        ax.plot(
            xs, ys,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
            zorder=5 if method == "base-sd3" else 3,
        )

    ax.set_xlabel("Number of Samples N")
    ax.set_ylabel("GenEval Best@N")
    ax.set_title(label)
    ax.set_xticks([N_TO_POS[t] for t in X_TICKS])
    ax.set_xticklabels([str(t) for t in X_TICKS])
    ax.set_xlim(-0.5, len(PLOT_NS) - 0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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


def plot_panels(categories, base_root, out_dir, stem="geneval_panels"):
    """Plot the given categories in a 2x3 grid with the legend inside the
    first subplot, for inclusion in the paper."""
    nrow, ncol = 1, 6
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(3.5 * ncol, 4.0 * nrow),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for i, (ax, (label, csv_name, _stem)) in enumerate(zip(axes, categories)):
        for method in METHODS:
            try:
                ns, ys = load_curve(base_root, method, csv_name)
            except FileNotFoundError as e:
                print(f"[warn] missing {e.filename}", file=sys.stderr)
                continue
            keep = np.isin(ns, PLOT_NS)
            ns, ys = ns[keep], ys[keep]
            xs = [N_TO_POS[n] for n in ns]
            ax.plot(
                xs, ys,
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
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        if i // ncol == nrow - 1:           # bottom row
            ax.set_xlabel("Number of Samples N")
        if i % ncol == 0:                   # left column
            ax.set_ylabel("Best@N")

    axes[0].legend(
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
    for label, csv_name, stem in CATEGORIES:
        plot_one(label, csv_name, stem, args.base_root, args.out_dir)
    # Combined 2x3 panel of the 6 per-tag curves for the paper.
    plot_panels(PANEL_CATEGORIES, args.base_root, args.out_dir)

    print_official_n4_markdown(args.base_root, n_seeds=args.official_n_seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot GenEval BoN comparison across SD-3.5-M methods, "
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
    ap.add_argument(
        "--official_n_seeds", type=int, default=4,
        help="N for the geneval-official summary table printed to stdout "
             "after plotting (default: 4, matching the official protocol).",
    )
    main(ap.parse_args())
