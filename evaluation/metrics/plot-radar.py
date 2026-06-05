# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-style radar chart for the 8 SD-3.5-M evaluation axes.

Plotter: it loads the two collector JSONs (every axis is already on a
"higher = better" scale — the Social-Bias axis is flipped to a bias-control
score inside collect-mean-of-n-metrics.py), then **min-max normalizes each axis
across the 6 methods** so every spoke spans 0..1 and the per-method spread on an
axis is fully visible regardless of that metric's native range. The radial
values are therefore relative (0 = worst method on that axis, 1 = best), not raw
scores — do not read absolute numbers off the rings, and do not compare two
charts axis-for-axis. It draws a circular radar with a decorative two-tier outer
ring:

  * an inner ring of light per-metric colour sectors (one wedge per axis), and
  * an outer ring split into two big background regions that group the 8 axes
    into the two metric families, each labelled along its arc:
        - Capability Upper Bound (Best-of-n) : GenEval / DPG-Bench / WISE / AnyText
        - Generation Properties (Mean-of-n)  : Safety / Social-Bias /
                                               No Artifact Rate / Real Score

Each of the 6 methods is one closed polyline with dot markers and a faint fill.
Radial grid is light grey dashed with ticks at 0.25/0.50/0.75/1.00, range 0..1.
The legend sits at the bottom, horizontal.

USAGE
-----
  python evaluation/metrics/plot-radar.py \
      --max_json  plot-radar/max-of-n.json \
      --mean_json plot-radar/mean-of-n/mean-of-n.json \
      --n 16 \
      --out plot-radar/radar.png

``max-of-n.json`` (collect-max-of-n-metrics.py) supplies the four Best-of-N
capability axes (axis order "Object Alignment / Dense Prompt / World Knowledge /
Visual Text"); ``mean-of-n.json`` (collect-mean-of-n-metrics.py) supplies the
four mean@N property axes ("Safe / Social Bias / Clean-rate / Real Score").

With no --max_json/--mean_json the script renders built-in demo values so you can
verify the styling offline. Missing/null entries become NaN (the polygon simply
skips that vertex) and a warning is printed.

FONT
----
Per request the script first looks for an "Intern" font; if absent it falls back
through Inter / Times New Roman / DejaVu Serif / serif and prints which it used.
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---- the 8 radar axes, in clockwise-from-top order -------------------------
METRICS = [
    "GenEval", "DPG-Bench", "WISE", "AnyText",          # group 1 (Best-of-n)
    "Safety", "Social-Bias", "No Artifact\nRate", "Real Score",  # group 2 (Mean-of-n)
]
GROUPS = [
    ("Capability Upper Bound (Best-of-n)", (0, 4)),  # axis indices [start, end)
    ("Generation Properties (Mean-of-n)", (4, 8)),
]

# ---- the 6 methods (display order = legend order) --------------------------
METHODS = ["Base", "Flow-GRPO", "GRPO-Guard", "DiffusionNFT", "Diffusion-DPO", "RealAlign"]

# NPG-style qualitative palette (distinct, print-friendly).
METHOD_COLORS = {
    "Base":          "#7F7F7F",
    "Flow-GRPO":     "#E64B35",
    "GRPO-Guard":    "#4DBBD5",
    "DiffusionNFT":  "#00A087",
    "Diffusion-DPO": "#3C5488",
    "RealAlign":     "#F39B7F",
}

# Light per-sector backgrounds (8), and the two group-region colours.
SECTOR_COLORS = [
    "#A8D0E6", "#A8D8C4", "#BFE3B0", "#E6D6A8",   # group 1 — cool/green
    "#F4C7A1", "#F2B5B5", "#E4C2DE", "#C9C2E4",   # group 2 — warm/violet
]
GROUP_COLORS = ["#CFE3F2", "#F6DEC9"]  # group-region ring fills

# ---- collector axis names -> radar axis indices ----------------------------
MAX_AXES = ["Object Alignment", "Dense Prompt", "World Knowledge", "Visual Text"]
MEAN_AXES = ["Safe", "Social Bias", "Clean-rate", "Real Score"]
METHOD_DIR_TO_NAME = {
    "base-sd3": "Base",
    "flowgrpo-pickscore-sd3": "Flow-GRPO",
    "grpo-guard-sd3": "GRPO-Guard",
    "diffusionnft-sd3": "DiffusionNFT",
    "diffusion-dpo-sd3": "Diffusion-DPO",
    "realalign-sd3": "RealAlign",
}


# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
def setup_font(preferred="Intern"):
    """Set rcParams font to `preferred` if installed, else a serif fallback."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chain = [preferred, "Inter", "Times New Roman", "DejaVu Serif", "serif"]
    for name in chain:
        if name == "serif" or name in available:
            plt.rcParams["font.family"] = name
            if name != preferred:
                print(f"[font] '{preferred}' not found; using '{name}'.")
            else:
                print(f"[font] using '{name}'.")
            return name


# ---------------------------------------------------------------------------
# Data (read-only: values are used exactly as stored in the JSONs)
# ---------------------------------------------------------------------------
def _demo_data():
    """Plausible 0..1 values (8 axes x 6 methods) for offline style checks."""
    rng = np.random.default_rng(7)
    base = {
        "Base":          [0.55, 0.78, 0.40, 0.45, 0.62, 0.50, 0.70, 0.58],
        "Flow-GRPO":     [0.72, 0.85, 0.55, 0.60, 0.75, 0.60, 0.80, 0.74],
        "GRPO-Guard":    [0.70, 0.83, 0.58, 0.57, 0.88, 0.78, 0.82, 0.80],
        "DiffusionNFT":  [0.68, 0.86, 0.52, 0.63, 0.70, 0.65, 0.85, 0.72],
        "Diffusion-DPO": [0.64, 0.80, 0.50, 0.55, 0.80, 0.72, 0.78, 0.77],
        "RealAlign":     [0.66, 0.82, 0.54, 0.58, 0.78, 0.70, 0.88, 0.90],
    }
    return {m: list(np.clip(np.array(v) + rng.normal(0, 0.01, 8), 0, 1)) for m, v in base.items()}


def _at_n(metric_block, n):
    """Pull value at N from a collector ``data[method][axis]`` mapping."""
    if metric_block is None:
        return np.nan
    v = metric_block.get(str(n), metric_block.get(n))
    return np.nan if v is None else float(v)


def normalize_minmax(data):
    """Min-max scale each of the 8 axes across the methods, in place-safe copy.

    For every axis, the 6 method values are linearly mapped so the smallest -> 0
    and the largest -> 1; NaNs are ignored. If every method ties on an axis (zero
    range) the axis is set to 0.5 (no information to spread). Returns a new dict.
    """
    methods = list(data.keys())
    mat = np.array([data[m] for m in methods], dtype=float)  # (n_methods, 8)
    out = mat.copy()
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.all(np.isnan(col)):
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi - lo < 1e-12:
            out[:, j] = np.where(np.isnan(col), np.nan, 0.5)
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return {m: out[i].tolist() for i, m in enumerate(methods)}


def load_from_json(max_json, mean_json, n, warnings):
    """Assemble {method: [8 values]} from the two collector JSONs at N=n.

    Raw values are read here; per-axis min-max normalization is applied later in
    main() so it covers both the JSON path and the demo path uniformly.
    """
    data = {m: [np.nan] * 8 for m in METHODS}

    def ingest(path, axes, offset):
        if not path:
            return
        if not os.path.exists(path):
            warnings.append(f"missing JSON: {path}")
            return
        with open(path) as f:
            blob = json.load(f)
        for dirname, block in blob.get("data", {}).items():
            name = METHOD_DIR_TO_NAME.get(dirname, dirname)
            if name not in data:
                continue
            for j, axis in enumerate(axes):
                data[name][offset + j] = _at_n(block.get(axis), n)

    ingest(max_json, MAX_AXES, 0)
    ingest(mean_json, MEAN_AXES, 4)

    for m in METHODS:
        if all(np.isnan(x) for x in data[m]):
            warnings.append(f"no data for method '{m}' at N={n}")
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _screen_deg(theta_rad):
    """Display angle (deg, math convention) for a polar theta under our offset."""
    # theta_offset = pi/2, theta_direction = -1  ->  screen = 90 - theta_deg
    return 90.0 - np.degrees(theta_rad)


def _tangential_rotation(theta_rad):
    """Text rotation (deg) tangent to the ring, kept upright."""
    rot = _screen_deg(theta_rad) - 90.0
    if rot < -90:
        rot += 180
    elif rot > 90:
        rot -= 180
    return rot


def plot_radar(data, out_path, size=9.0, dpi=300, title=None):
    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # radial layout (data lives in 0..1; decoration sits beyond)
    R_DATA = 1.0
    SECTOR_B, SECTOR_H = 1.05, 0.13          # per-metric colour ring
    GROUP_B, GROUP_H = 1.20, 0.14            # group-region ring
    FRAME_R1, FRAME_R2 = 1.345, 1.365        # double outer frame
    R_OUTER = 1.40

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, R_OUTER)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    grid_kw = dict(color="#BFBFBF", lw=0.8, ls=(0, (4, 4)), zorder=1)

    # --- manual radial circles (dashed) at the four ticks ---
    tick_r = [0.25, 0.50, 0.75, 1.00]
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    for r in tick_r:
        ax.plot(circle_theta, [r] * len(circle_theta), **grid_kw)
    # --- manual spokes from centre to r=1 ---
    for a in angles:
        ax.plot([a, a], [0, R_DATA], **grid_kw)

    # --- radial tick labels along the top spoke ---
    for r in tick_r:
        ax.text(angles[0], r, f"{r:.2f}", ha="center", va="center",
                fontsize=8.5, color="#4D4D4D", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    sector_w = 2 * np.pi / n
    gap = np.deg2rad(1.6)

    # --- per-metric colour ring ---
    ax.bar(angles, height=SECTOR_H, width=sector_w - gap, bottom=SECTOR_B,
           color=SECTOR_COLORS, edgecolor="white", linewidth=1.0,
           alpha=0.85, zorder=2, align="center")

    # --- two group-region arcs ---
    for gi, (label, (s, e)) in enumerate(GROUPS):
        center = (angles[s] + angles[e - 1]) / 2.0
        width = (e - s) * sector_w - gap
        ax.bar(center, height=GROUP_H, width=width, bottom=GROUP_B,
               color=GROUP_COLORS[gi], edgecolor="white", linewidth=1.2,
               alpha=0.95, zorder=2, align="center")
        rot = _tangential_rotation(center)
        ax.text(center, GROUP_B + GROUP_H / 2.0, label, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", fontsize=10.5,
                fontweight="bold", color="#33363B", zorder=6)

    # --- double outer frame ---
    for r in (FRAME_R1, FRAME_R2):
        ax.plot(circle_theta, [r] * len(circle_theta), color="#2B2B2B", lw=1.3, zorder=3)

    # --- metric labels inside their sectors ---
    for a, name in zip(angles, METRICS):
        rot = _tangential_rotation(a)
        ax.text(a, SECTOR_B + SECTOR_H / 2.0, name, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", fontsize=10.5,
                fontweight="bold", color="#222222", zorder=6, linespacing=0.9)

    # --- method polylines ---
    closed_angles = np.concatenate([angles, angles[:1]])
    handles, labels = [], []
    for m in METHODS:
        vals = np.array(data[m], dtype=float)
        closed = np.concatenate([vals, vals[:1]])
        color = METHOD_COLORS[m]
        line, = ax.plot(closed_angles, closed, color=color, lw=2.0,
                        marker="o", markersize=5, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax.fill(closed_angles, closed, color=color, alpha=0.10, zorder=4)
        handles.append(line)
        labels.append(m)

    # --- legend at the bottom, horizontal ---
    ncol = 6 if len(METHODS) <= 6 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, fontsize=11, handlelength=1.8,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.005))

    if title:
        fig.suptitle(title, y=0.97, fontsize=14, fontweight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.10)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved radar chart to {out_path}  ({dpi} dpi)")


# ---------------------------------------------------------------------------
def main(args):
    setup_font(args.font)

    warnings = []
    if args.max_json or args.mean_json:
        data = load_from_json(args.max_json, args.mean_json, args.n, warnings)
    else:
        print("[data] no --max_json/--mean_json given; using built-in demo data.")
        data = _demo_data()

    # Per-axis min-max normalization across methods (relative ranking view).
    data = normalize_minmax(data)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.title:
        title = args.title
    elif args.max_json or args.mean_json:
        title = f"SD-3.5-M  (N = {args.n}, per-axis min-max normalized)"
    else:
        title = None
    plot_radar(data, args.out, size=args.size, dpi=args.dpi, title=title)

    if warnings:
        print(f"\n--- {len(warnings)} warning(s) ---")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Paper-style 8-axis radar chart for the 6 SD-3.5-M methods."
    )
    ap.add_argument("--max_json", default=None,
                    help="max-of-n.json from collect-max-of-n-metrics.py (capability axes).")
    ap.add_argument("--mean_json", default=None,
                    help="mean-of-n.json from collect-mean-of-n-metrics.py (property axes).")
    ap.add_argument("--n", type=int, default=16,
                    help="Which N to plot from the collector JSONs (default 16).")
    ap.add_argument("--out", default="radar.png", help="Output PNG path.")
    ap.add_argument("--size", type=float, default=9.0, help="Figure size in inches (square).")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (>=300).")
    ap.add_argument("--title", default=None, help="Optional figure title.")
    ap.add_argument("--font", default="Intern", help="Preferred font family (default Intern).")
    main(ap.parse_args())
