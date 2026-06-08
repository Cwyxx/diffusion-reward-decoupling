"""Plot trade-off scatter: aggregated human-preference vs GenEval / WISE BoN@32.

X axis: aggregated human-preference score, defined as the equal-weight mean
of three per-method scores (PickScore, ImageReward, HPSv3) after each is
min-max normalized to [0, 1] across the 5 methods. The base SD-3.5-M model
is therefore at x=0 by construction (it is the lowest on every HPS column),
and GRPO-Guard is at x close to 1.

Y axis: GenEval / WISE Best-of-N Overall at N=32 — the per-prompt pass@32
ceiling. This captures the model's *generation capability ceiling* rather
than the mean. RL post-trained methods that collapse modes typically have a
LOWER BoN@32 ceiling than the base model even when their mean rises, which
is exactly the trade-off these scatters are meant to expose.

Each method is one scatter point, colored to match the existing
plot-bestofn-{geneval,wise}-comparison.py palette, with the method name
annotated next to the marker. No reference / diagonal lines are drawn.

Outputs (under --out_dir):
  tradeoff_hps_vs_geneval.png / .pdf
  tradeoff_hps_vs_wise.png    / .pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
METHOD_COLORS = {
    "base-sd3":               "#a8a8a8",
    "flowgrpo-pickscore-sd3": "#8da0cb",
    "grpo-guard-sd3":         "#66c2a5",
    "diffusion-dpo-sd3":      "#fc8d62",
    "realalign-sd3":          "#e78ac3",
}

# Raw per-method numbers supplied by the user.
HPS_RAW = {
    "base-sd3":               {"pickscore": 0.862132, "imagereward": 0.792718, "hpsv3": 10.025507},
    "flowgrpo-pickscore-sd3": {"pickscore": 0.906308, "imagereward": 1.263731, "hpsv3": 12.625672},
    "grpo-guard-sd3":         {"pickscore": 0.923846, "imagereward": 1.370297, "hpsv3": 12.514752},
    "diffusion-dpo-sd3":      {"pickscore": 0.872932, "imagereward": 0.959584, "hpsv3": 10.859254},
    "realalign-sd3":          {"pickscore": 0.877041, "imagereward": 1.082241, "hpsv3": 12.768349},
}
# GenEval / WISE Best-of-N @ N=32 (generation ceiling, not mean score).
GENEVAL_BON32 = {
    "base-sd3":               0.94,
    "flowgrpo-pickscore-sd3": 0.92,
    "grpo-guard-sd3":         0.91,
    "diffusion-dpo-sd3":      0.94,
    "realalign-sd3":          0.90,
}
WISE_BON32 = {
    "base-sd3":               0.74,
    "flowgrpo-pickscore-sd3": 0.67,
    "grpo-guard-sd3":         0.62,
    "diffusion-dpo-sd3":      0.71,
    "realalign-sd3":          0.66,
}

HPS_KEYS = ["pickscore", "imagereward", "hpsv3"]


def aggregated_hps():
    """Min-max normalize each HPS column across methods, then equal-weight mean.

    Returns {method: float in [0,1]}. With 5 methods, the per-column min lands
    at 0 and the per-column max at 1; the aggregated score is the mean of the
    three normalized values.
    """
    cols = {k: np.array([HPS_RAW[m][k] for m in METHODS], dtype=float) for k in HPS_KEYS}
    normed = {}
    for k, v in cols.items():
        lo, hi = v.min(), v.max()
        normed[k] = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    stacked = np.stack([normed[k] for k in HPS_KEYS], axis=0)  # (3, n_methods)
    agg = stacked.mean(axis=0)
    return {m: float(agg[i]) for i, m in enumerate(METHODS)}


def plot_one(x_by_method, y_by_method, x_label, y_label, title, out_path_stem):
    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)

    xs = np.array([x_by_method[m] for m in METHODS])
    ys = np.array([y_by_method[m] for m in METHODS])

    for m, x, y in zip(METHODS, xs, ys):
        ax.scatter(
            [x], [y],
            s=140,
            color=METHOD_COLORS[m],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
            label=METHOD_LABELS[m],
        )
        ax.annotate(
            METHOD_LABELS[m],
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9.5,
            color="#333333",
        )

    # Pad the axes so labels don't run off the canvas.
    x_pad = 0.05 * (xs.max() - xs.min() + 1e-9)
    y_pad = 0.10 * (ys.max() - ys.min() + 1e-9)
    ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad + 0.18)
    ax.set_ylim(ys.min() - y_pad, ys.max() + y_pad)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)

    png_path = f"{out_path_stem}.png"
    pdf_path = f"{out_path_stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path} and {pdf_path}")


def print_hps_breakdown(agg_hps):
    print()
    print("Aggregated human-preference score (min-max normalized per metric, then mean):")
    print("| Method | PickScore | ImageReward | HPSv3 | Aggregated |")
    print("|---|---|---|---|---|")
    for m in METHODS:
        raw = HPS_RAW[m]
        print(
            f"| {METHOD_LABELS[m]} | {raw['pickscore']:.4f} | "
            f"{raw['imagereward']:.4f} | {raw['hpsv3']:.4f} | {agg_hps[m]:.4f} |"
        )
    print()


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
        "font.family": "DejaVu Sans",
    })

    os.makedirs(args.out_dir, exist_ok=True)
    agg_hps = aggregated_hps()
    print_hps_breakdown(agg_hps)

    x_label = "Human preference (norm. mean of PickScore, ImageReward, HPSv3)"

    plot_one(
        x_by_method=agg_hps,
        y_by_method=GENEVAL_BON32,
        x_label=x_label,
        y_label="GenEval Best-of-N (N=32) Overall",
        title="Trade-off: Human preference vs GenEval BoN@32",
        out_path_stem=os.path.join(args.out_dir, "tradeoff_hps_vs_geneval"),
    )
    plot_one(
        x_by_method=agg_hps,
        y_by_method=WISE_BON32,
        x_label=x_label,
        y_label="WISE Best-of-N (N=32) Overall",
        title="Trade-off: Human preference vs WISE BoN@32",
        out_path_stem=os.path.join(args.out_dir, "tradeoff_hps_vs_wise"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Scatter the 5 SD-3.5-M methods on (aggregated human-preference, "
                    "GenEval Overall) and (aggregated human-preference, WISE Overall).",
    )
    ap.add_argument(
        "--out_dir", default="tradeoff_plots",
        help="Directory to write the two PNG + PDF files into.",
    )
    main(ap.parse_args())
