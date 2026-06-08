# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-style Best-of-N capability radar (max-of-n.json).

Renders a 2x3 grid of radars (one per N in 1/2/4/8/16/32) over 6 axes. The top
half holds Qwen-Image-Bench / WISE / DPG-Bench, the bottom half GenEval /
Human Preference / AnyText. Qwen-Image-Bench has no experiment results yet, so
it is drawn as an **empty axis** (the axis label shows, but no method has a
vertex there) — fill it in later by wiring a data source in ``AXIS_SOURCES``.

Each panel uses **rank layering**: on every axis the 6 methods are snapped to 6
fixed concentric rings by their rank (rank 1 = best = outer ring, rank 6 = worst
= inner ring), so the relative ordering is immediately legible and no two
methods overlap on an axis. The absolute "growth with N" is intentionally
dropped (every panel fills the same rings — only the ordering shifts across
panels). Decoration: a thin per-metric colour ring (one wedge per axis) plus a
double outer frame; the family name sits in the figure suptitle.

USAGE
-----
  python evaluation/metrics/radar/plot-radar-max-of-n.py \
      --max_json plot-radar/max-of-n.json \
      --out plot-radar/max-of-n/radar.png

``max-of-n.json`` (collect-max-of-n-metrics.py) supplies the capability axes
(collector axis order "Object Alignment / Dense Prompt / World Knowledge /
Visual Text"). With no --max_json the script renders built-in demo values.

FONT
----
Looks for an "Intern" font first; if absent it falls back through
Inter / Times New Roman / DejaVu Serif / serif and prints which it used.
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath


FAMILY = "Capability Upper Bound (Best-of-n)"

# Radial floor / innermost (worst) rank ring.
R_MIN = 0.1

# ---- axes (clockwise from top); first TOP_COUNT are centred at the top -----
AXES = ["Qwen-Image-Bench", "WISE", "DPG-Bench",     # top half
        "GenEval", "Human Preference", "AnyText"]    # bottom half
TOP_COUNT = 3
# Per-axis data source: (collector_axis) in max-of-n.json, or None for a
# placeholder (empty) axis whose label still shows but carries no data.
AXIS_SOURCES = [
    None,                 # Qwen-Image-Bench  (TBD)
    "World Knowledge",    # WISE
    "Dense Prompt",       # DPG-Bench
    "Object Alignment",   # GenEval
    "Human Preference",   # Human Preference
    "Visual Text",        # AnyText
]

# ---- the 6 methods (display order = legend order) --------------------------
METHODS = ["Base", "Flow-GRPO", "GRPO-Guard", "DiffusionNFT", "Diffusion-DPO", "RealAlign"]
FOCUS_METHOD = "Base"  # drawn on the topmost layer (z-order only)

METHOD_COLORS = {
    "Base":          "#00A087",
    "Flow-GRPO":     "#E64B35",
    "GRPO-Guard":    "#4DBBD5",
    "DiffusionNFT":  "#7F7F7F",
    "Diffusion-DPO": "#3C5488",
    "RealAlign":     "#F39B7F",
}
METHOD_STYLES = {
    "Base":          ("-",                  "o"),
    "Flow-GRPO":     ((0, (5, 2)),          "s"),
    "GRPO-Guard":    ((0, (1, 1.4)),        "^"),
    "DiffusionNFT":  ((0, (6, 2, 1, 2)),    "D"),
    "Diffusion-DPO": ((0, (3, 1.6)),        "v"),
    "RealAlign":     ("-.",                 "P"),
}
SECTOR_PALETTE = [
    "#A8D0E6", "#A8D8C4", "#BFE3B0", "#E6D6A8",
    "#F4C7A1", "#F2B5B5", "#E4C2DE", "#C9C2E4",
]
METHOD_DIR_TO_NAME = {
    "base-sd3": "Base",
    "flowgrpo-pickscore-sd3": "Flow-GRPO",
    "grpo-guard-sd3": "GRPO-Guard",
    "diffusionnft-sd3": "DiffusionNFT",
    "diffusion-dpo-sd3": "Diffusion-DPO",
    "realalign-sd3": "RealAlign",
}


# ---------------------------------------------------------------------------
def setup_font(preferred="Intern"):
    """Set rcParams font to `preferred` if installed, else a serif fallback."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chain = [preferred, "Inter", "Times New Roman", "DejaVu Serif", "serif"]
    for name in chain:
        if name == "serif" or name in available:
            plt.rcParams["font.family"] = name
            print(f"[font] using '{name}'." if name == preferred
                  else f"[font] '{preferred}' not found; using '{name}'.")
            return name


def _demo_data(n_axes):
    """Plausible 0..1 values (n_axes x 6 methods) for offline style checks."""
    rng = np.random.default_rng(7)
    return {m: list(np.clip(rng.uniform(0.35, 0.95, n_axes), 0, 1)) for m in METHODS}


def _at_n(metric_block, n):
    """Pull value at N from a collector ``data[method][axis]`` mapping."""
    if metric_block is None:
        return np.nan
    v = metric_block.get(str(n), metric_block.get(n))
    return np.nan if v is None else float(v)


def load_data(max_json, n, warnings):
    """Assemble {method: [len(AXES) values]} from max-of-n.json at N=n."""
    data = {m: [np.nan] * len(AXES) for m in METHODS}
    if not max_json:
        return data
    if not os.path.exists(max_json):
        warnings.append(f"missing JSON: {max_json}")
        return data
    with open(max_json) as f:
        blob = json.load(f)
    for dirname, block in blob.get("data", {}).items():
        name = METHOD_DIR_TO_NAME.get(dirname, dirname)
        if name not in data:
            continue
        for j, axis in enumerate(AXIS_SOURCES):
            if axis is None:
                continue  # placeholder axis: stays NaN for everyone
            data[name][j] = _at_n(block.get(axis), n)
    return data


def rank_positions(data_by_n):
    """Place methods on fixed concentric rings by their rank, per panel & axis.

    rank 1 (best) -> outer ring (1.0), rank M (worst) -> inner ring (R_MIN).
    Axes where every method is NaN (placeholder axes) stay NaN for all methods.
    """
    out = {}
    for nv, block in data_by_n.items():
        methods = list(block.keys())
        mat = np.array([block[m] for m in methods], dtype=float)  # (M, n_axes)
        rings = np.linspace(1.0, R_MIN, len(methods))  # ring[0]=best(outer)
        res = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            idx = np.where(~np.isnan(col))[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(-col[idx], kind="stable")]  # best -> worst
            for rank_pos, mi in enumerate(order):
                res[mi, j] = rings[rank_pos]
        out[nv] = {m: res[i].tolist() for i, m in enumerate(methods)}
    return out


# ---------------------------------------------------------------------------
def _screen_deg(theta_rad):
    """Display angle (deg, math convention) for a polar theta under our offset."""
    return 90.0 - np.degrees(theta_rad)  # theta_offset=pi/2, theta_direction=-1


def _tangential_rotation(theta_rad):
    """Text rotation (deg) tangent to the ring, kept upright."""
    rot = _screen_deg(theta_rad) - 90.0
    if rot < -90:
        rot += 180
    elif rot > 90:
        rot -= 180
    return rot


def _arc_text(ax, fig, center_theta, radius, text, fontsize=13,
              fontweight="bold", color="#33363B", zorder=6, tracking=1.12):
    """Lay ``text`` out character-by-character along the arc at ``radius``."""
    fig.canvas.draw()  # realise transforms before measuring
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((0.0, 1.0))
    px_per_unit = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    pts_per_unit = px_per_unit * 72.0 / fig.dpi

    fp = FontProperties(family=plt.rcParams.get("font.family"),
                        size=fontsize, weight=fontweight)
    ang_w = []
    for ch in text:
        probe = "n" if ch == " " else ch
        w_pts = TextPath((0, 0), probe, size=fontsize, prop=fp).get_extents().width
        if ch == " ":
            w_pts *= 0.55
        ang_w.append((w_pts * tracking) / pts_per_unit / radius)

    chars = list(text)
    if np.sin(np.radians(_screen_deg(center_theta))) < 0:  # lower half: reverse
        chars = chars[::-1]
        ang_w = ang_w[::-1]

    total = sum(ang_w)
    cursor = center_theta - total / 2.0
    for ch, w in zip(chars, ang_w):
        a = cursor + w / 2.0
        ax.text(a, radius, ch, ha="center", va="center",
                rotation=_tangential_rotation(a), rotation_mode="anchor",
                fontsize=fontsize, fontweight=fontweight, color=color, zorder=zorder)
        cursor += w


def _draw_radar(ax, fig, data, title=None,
                fs_metric=8.0, fs_tick=6.5, fs_title=13.0, lw=1.6, ms=4.8):
    """Draw one rank-layered radar onto ``ax``; returns (handles, labels)."""
    n = len(AXES)
    sector_w = 2 * np.pi / n
    center_idx = (TOP_COUNT - 1) / 2.0          # centre the top group at theta=0
    start_offset = -center_idx * sector_w
    angles = np.arange(n) * sector_w + start_offset
    sector_colors = SECTOR_PALETTE[:n]

    R_DATA = 1.0
    SECTOR_B, SECTOR_H = 1.05, 0.13
    FRAME_R1, FRAME_R2 = 1.245, 1.265
    R_OUTER = 1.30

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, R_OUTER)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    grid_kw = dict(color="#BFBFBF", lw=0.8, ls=(0, (4, 4)), zorder=1)
    circle_theta = np.linspace(0, 2 * np.pi, 200)

    # one ring per rank: rank 1 (best) on the outer ring, rank 6 (worst) inner
    ring_r = list(np.linspace(R_DATA, R_MIN, len(METHODS)))
    ring_lab = [str(i + 1) for i in range(len(METHODS))]

    for r in ring_r:
        ax.plot(circle_theta, [r] * len(circle_theta), **grid_kw)
    for a in angles:
        ax.plot([a, a], [R_MIN, R_DATA], **grid_kw)

    for r, lab in zip(ring_r, ring_lab):
        ax.text(0.0, r, lab, ha="center", va="center",
                fontsize=fs_tick, color="#4D4D4D", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    gap = np.deg2rad(1.6)
    ax.bar(angles, height=SECTOR_H, width=sector_w - gap, bottom=SECTOR_B,
           color=sector_colors, edgecolor="white", linewidth=1.0,
           alpha=0.85, zorder=2, align="center")

    for r in (FRAME_R1, FRAME_R2):
        ax.plot(circle_theta, [r] * len(circle_theta), color="#2B2B2B", lw=1.3, zorder=3)

    for a, name in zip(angles, AXES):
        _arc_text(ax, fig, a, SECTOR_B + SECTOR_H / 2.0, name,
                  fontsize=fs_metric, fontweight="bold", color="#222222", zorder=6)

    closed_angles = np.concatenate([angles, angles[:1]])
    handles, labels = [], []
    for m in METHODS:
        vals = np.array(data[m], dtype=float)
        closed = np.concatenate([vals, vals[:1]])
        color = METHOD_COLORS[m]
        ls, mk = METHOD_STYLES[m]
        line, = ax.plot(closed_angles, closed, color=color, lw=lw, linestyle=ls,
                        marker=mk, markersize=ms, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=6 if m == FOCUS_METHOD else 5)
        ax.fill(closed_angles, closed, color=color, alpha=0.06, zorder=4)
        handles.append(line)
        labels.append(m)

    if title:
        ax.set_title(title, fontsize=fs_title, fontweight="bold", pad=14)
    return handles, labels


def plot_grid(data_by_n, n_list, out_path, panel_size=5.8, dpi=300):
    """A 2x3 grid of rank-layered radars, one per N, with a shared legend."""
    ncols = 3
    nrows = int(np.ceil(len(n_list) / ncols))
    fig = plt.figure(figsize=(panel_size * ncols, panel_size * nrows + 0.9))

    handles = labels = None
    for i, nv in enumerate(n_list):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        h, l = _draw_radar(ax, fig, data_by_n[nv], title=f"N={nv}")
        if handles is None:
            handles, labels = h, l

    ncol = 6 if len(METHODS) <= 6 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, fontsize=11, handlelength=1.8,
               columnspacing=1.8, bbox_to_anchor=(0.5, 0.012))

    fig.suptitle(FAMILY, y=0.995, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.09,
                        wspace=0.12, hspace=0.22)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(n_list)}-panel radar grid to {out_path}  ({dpi} dpi)")


def main(args):
    setup_font(args.font)
    warnings = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    n_list = [int(x) for x in str(args.n_list).split(",") if x.strip()]
    if args.max_json:
        data_by_n = {nv: load_data(args.max_json, nv, warnings) for nv in n_list}
    else:
        print("[data] no --max_json given; using built-in demo data.")
        data_by_n = {nv: _demo_data(len(AXES)) for nv in n_list}
    data_by_n = rank_positions(data_by_n)
    plot_grid(data_by_n, n_list, args.out, panel_size=args.panel_size, dpi=args.dpi)

    if warnings:
        print(f"\n--- {len(warnings)} warning(s) ---")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Best-of-N capability radar grid (max-of-n.json), 6 methods."
    )
    ap.add_argument("--max_json", default=None,
                    help="max-of-n.json from collect-max-of-n-metrics.py.")
    ap.add_argument("--out", default="plot-radar/max-of-n/radar.png",
                    help="Output PNG path (default plot-radar/max-of-n/radar.png).")
    ap.add_argument("--n_list", default="1,2,4,8,16,32",
                    help="Comma-separated N values for the grid (default 1,2,4,8,16,32).")
    ap.add_argument("--panel_size", type=float, default=5.8,
                    help="Per-panel size in inches (default 5.8).")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (>=300).")
    ap.add_argument("--font", default="Intern", help="Preferred font family (default Intern).")
    main(ap.parse_args())
