# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-style combined overview radar (capability + properties on one chart).

A single radar over 9 axes drawn from both collectors, at one N (``--n``,
default 16). The axes are arranged in two families separated by a two-tier outer
ring (a thin per-metric colour wedge ring, plus a group-region band with a
curved family label):

  * Generation Capability (Best-of-n) — top 5 axes, from max-of-n.json:
        Object Alignment / Dense Prompt / World Knowledge / Visual Text /
        Human Preference
  * Generation Property (Mean-of-n)   — bottom 4 axes, from mean-of-n.json:
        Safe / Gender Balance / Clean Rate / Realism

Each axis is min-max normalized across the 6 methods, then lifted onto the
[R_MIN, 1] radial floor (0 = worst method on that axis, 1 = best); ring ticks
0.25/0.50/0.75/1.00 are relative, not raw scores. The radial values are
therefore relative (0 = worst method on that axis, 1 = best), not raw scores —
do not read absolute numbers off the rings, and do not compare two charts
axis-for-axis.

USAGE
-----
  python evaluation/metrics/plot-radar.py \
      --max_json  plot-radar/max-of-n.json \
      --mean_json plot-radar/mean-of-n.json \
      --out plot-radar/radar.png --n 16

``--single`` renders one full-size chart at ``--n``; the default renders a 2x3
grid over ``--n_list`` (each panel min-max'd within its own N). With no
--max_json/--mean_json the script renders built-in demo values. Missing/null
entries become NaN (the polygon skips that vertex) and a warning is printed.

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


# Radial floor: remap [0,1] -> [R_MIN, 1] so the worst method sits on the inner
# ring instead of the dead centre.
R_MIN = 0.2

# ---- the 9 axes (clockwise from top) ---------------------------------------
# Group 1 (top, 5 axes) = capability; group 2 (bottom, 4 axes) = properties.
AXES = [
    "World Knowledge", "Dense Prompt", "Object Alignment", "Visual Text",
    "Human Preference",                                  # group 1 (Best-of-n)
    "Clean Rate", "Realism", "Safe", "Gender Balance",   # group 2 (Mean-of-n)
]
GROUPS = [
    ("Generation Capability (Best-of-n)", (0, 5)),  # axis indices [start, end)
    ("Generation Property (Mean-of-n)", (5, 9)),
]
# Per-axis source: ("max"|"mean", collector_axis). The display name in AXES may
# differ from the collector axis key (e.g. "Gender Balance" <- "Social Bias").
AXIS_SOURCES = [
    ("max", "World Knowledge"),    # World Knowledge
    ("max", "Dense Prompt"),       # Dense Prompt
    ("max", "Object Alignment"),   # Object Alignment
    ("max", "Visual Text"),        # Visual Text
    ("max", "Human Preference"),   # Human Preference
    ("mean", "Clean-rate"),        # Clean Rate
    ("mean", "Realism"),           # Realism
    ("mean", "Safe"),              # Safe
    ("mean", "Social Bias"),       # Gender Balance
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
# Light per-sector backgrounds (9): cool/green for capability, warm/violet for
# properties. The two group-region band colours follow.
SECTOR_COLORS = [
    "#A8D0E6", "#A8D8C4", "#BFE3B0", "#CDE3A6", "#A8CBE6",  # capability (5)
    "#F4C7A1", "#F2B5B5", "#E4C2DE", "#C9C2E4",             # properties (4)
]
GROUP_COLORS = ["#CFE3F2", "#F6DEC9"]
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


def _load_blob(path, warnings):
    """Read a collector JSON, or return None (with a warning) if absent."""
    if not path:
        return None
    if not os.path.exists(path):
        warnings.append(f"missing JSON: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_data(blobs, n, warnings):
    """Assemble {method: [len(AXES) values]} from both collectors at N=n."""
    data = {m: [np.nan] * len(AXES) for m in METHODS}
    for j, src in enumerate(AXIS_SOURCES):
        if src is None:
            continue  # placeholder axis: stays NaN for everyone
        jkey, axis = src
        blob = blobs.get(jkey)
        if blob is None:
            continue
        for dirname, block in blob.get("data", {}).items():
            name = METHOD_DIR_TO_NAME.get(dirname, dirname)
            if name not in data:
                continue
            data[name][j] = _at_n(block.get(axis), n)
    return data


def normalize_minmax(data):
    """Min-max scale each axis across the methods (NaNs ignored; ties -> 0.5)."""
    methods = list(data.keys())
    mat = np.array([data[m] for m in methods], dtype=float)  # (n_methods, n_axes)
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


def _floor(vals):
    """Lift normalized 0..1 values onto the [R_MIN, 1] radial floor."""
    return [R_MIN + (1.0 - R_MIN) * x for x in vals]


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
                fs_metric=10.0, fs_group=12.5, fs_tick=8.5, fs_title=14.0,
                lw=2.0, ms=7.0):
    """Draw one value-based radar (rings, spokes, two-tier outer ring, method
    polylines) onto ``ax``; returns (handles, labels)."""
    n = len(AXES)
    sector_w = 2 * np.pi / n
    # Centre group 1 (capability) at the top; group 2 (properties) falls to bottom.
    (s0, e0) = GROUPS[0][1]
    start_offset = -((s0 + (e0 - 1)) / 2.0) * sector_w
    angles = np.arange(n) * sector_w + start_offset

    R_DATA = 1.0
    SECTOR_B, SECTOR_H = 1.05, 0.13          # per-metric colour ring
    GROUP_B, GROUP_H = 1.20, 0.16            # group-region band
    FRAME_R1, FRAME_R2 = 1.365, 1.385        # double outer frame
    R_OUTER = 1.42

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

    ring_r = [0.25, 0.50, 0.75, 1.00]
    ring_lab = [f"{r:.2f}" for r in ring_r]

    for r in ring_r:
        ax.plot(circle_theta, [r] * len(circle_theta), **grid_kw)
    for a in angles:
        ax.plot([a, a], [R_MIN, R_DATA], **grid_kw)

    tick_theta = np.deg2rad(15)  # offset the tick labels off the top spoke
    for r, lab in zip(ring_r, ring_lab):
        ax.text(tick_theta, r, lab, ha="center", va="center",
                fontsize=fs_tick, color="#4D4D4D", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    gap = np.deg2rad(1.6)

    # --- per-metric colour ring ---
    ax.bar(angles, height=SECTOR_H, width=sector_w - gap, bottom=SECTOR_B,
           color=SECTOR_COLORS, edgecolor="white", linewidth=1.0,
           alpha=0.85, zorder=2, align="center")

    # --- two group-region bands with curved family labels ---
    for gi, (label, (s, e)) in enumerate(GROUPS):
        center = (angles[s] + angles[e - 1]) / 2.0
        width = (e - s) * sector_w - gap
        ax.bar(center, height=GROUP_H, width=width, bottom=GROUP_B,
               color=GROUP_COLORS[gi], edgecolor="white", linewidth=1.2,
               alpha=0.95, zorder=2, align="center")
        _arc_text(ax, fig, center, GROUP_B + GROUP_H / 2.0, label,
                  fontsize=fs_group, fontweight="bold", color="#33363B", zorder=6)

    # --- double outer frame ---
    for r in (FRAME_R1, FRAME_R2):
        ax.plot(circle_theta, [r] * len(circle_theta), color="#2B2B2B", lw=1.3, zorder=3)

    # --- metric labels curved inside their sectors ---
    for a, name in zip(angles, AXES):
        _arc_text(ax, fig, a, SECTOR_B + SECTOR_H / 2.0, name,
                  fontsize=fs_metric, fontweight="bold", color="#222222", zorder=6)

    # --- method polylines ---
    closed_angles = np.concatenate([angles, angles[:1]])
    handles, labels = [], []
    for m in METHODS:
        vals = np.array(data[m], dtype=float)
        closed = np.concatenate([vals, vals[:1]])
        color = METHOD_COLORS[m]
        ls, mk = METHOD_STYLES[m]
        line, = ax.plot(closed_angles, closed, color=color, lw=lw, linestyle=ls,
                        marker=mk, markersize=ms, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=6 if m == FOCUS_METHOD else 5)
        ax.fill(closed_angles, closed, color=color, alpha=0.06, zorder=4)
        handles.append(line)
        labels.append(m)

    if title:
        ax.set_title(title, fontsize=fs_title, fontweight="bold", pad=16)
    return handles, labels


def plot_single(data, out_path, n, size=10.0, dpi=300):
    """Single full-size combined radar (one N) with a bottom legend."""
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="polar")
    handles, labels = _draw_radar(ax, fig, data, title=f"N={n}")

    ncol = 6 if len(METHODS) <= 6 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, fontsize=11, handlelength=1.8,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved combined radar chart to {out_path}  ({dpi} dpi)")


def plot_radar_grid(data_by_n, n_list, out_path, panel_size=5.8, dpi=300):
    """A 2x3 (rows x cols) grid of combined radars, one per N in ``n_list`` (each
    panel titled ``N=x``), sharing one horizontal legend at the bottom. Each
    panel is min-max normalized within its own N (across the 6 methods), so the
    panels read independently — they are not on a common cross-N scale."""
    ncols = 3
    nrows = int(np.ceil(len(n_list) / ncols))
    fig = plt.figure(figsize=(panel_size * ncols, panel_size * nrows + 0.9))

    handles = labels = None
    for i, nv in enumerate(n_list):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        h, l = _draw_radar(ax, fig, data_by_n[nv], title=f"N={nv}",
                           fs_metric=7.5, fs_group=9.5, fs_tick=6.5, fs_title=13.0,
                           lw=1.6, ms=7.5)
        if handles is None:
            handles, labels = h, l

    ncol = 6 if len(METHODS) <= 6 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               frameon=False, fontsize=11, handlelength=1.8,
               columnspacing=1.8, bbox_to_anchor=(0.5, 0.012))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08,
                        wspace=0.16, hspace=0.20)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {len(n_list)}-panel combined radar grid to {out_path}  ({dpi} dpi)")


def main(args):
    setup_font(args.font)
    warnings = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    have_json = bool(args.max_json or args.mean_json)
    blobs = {"max": _load_blob(args.max_json, warnings),
             "mean": _load_blob(args.mean_json, warnings)} if have_json else None
    if not have_json:
        print("[data] no --max_json/--mean_json given; using built-in demo data.")

    if args.single:
        # --- one full-size chart at --n ---
        data = load_data(blobs, args.n, warnings) if have_json else _demo_data(len(AXES))
        data = normalize_minmax(data)
        data = {m: _floor(v) for m, v in data.items()}
        plot_single(data, args.out, n=args.n, size=args.size, dpi=args.dpi)
    else:
        # --- 2x3 grid over n_list, each panel min-max'd within its own N ---
        n_list = [int(x) for x in str(args.n_list).split(",") if x.strip()]
        if have_json:
            data_by_n = {nv: load_data(blobs, nv, warnings) for nv in n_list}
        else:
            data_by_n = {nv: _demo_data(len(AXES)) for nv in n_list}
        data_by_n = {nv: {m: _floor(v) for m, v in normalize_minmax(block).items()}
                     for nv, block in data_by_n.items()}
        plot_radar_grid(data_by_n, n_list, args.out,
                        panel_size=args.panel_size, dpi=args.dpi)

    if warnings:
        print(f"\n--- {len(warnings)} warning(s) ---")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Combined capability+properties overview radar (9 axes), 6 methods."
    )
    ap.add_argument("--max_json", default=None,
                    help="max-of-n.json from collect-max-of-n-metrics.py (capability axes).")
    ap.add_argument("--mean_json", default=None,
                    help="mean-of-n.json from collect-mean-of-n-metrics.py (property axes).")
    ap.add_argument("--out", default="plot-radar/radar.png",
                    help="Output PNG path (default plot-radar/radar.png).")
    ap.add_argument("--single", action="store_true",
                    help="Render one full-size chart at --n instead of the 2x3 grid.")
    ap.add_argument("--n", type=int, default=16,
                    help="Which N to plot in --single mode (default 16).")
    ap.add_argument("--n_list", default="1,2,4,8,16,32",
                    help="Comma-separated N values for the grid (default 1,2,4,8,16,32).")
    ap.add_argument("--panel_size", type=float, default=5.8,
                    help="Per-panel size in inches for the grid (default 5.8).")
    ap.add_argument("--size", type=float, default=10.0,
                    help="Figure size in inches (square) for --single mode (default 10.0).")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (>=300).")
    ap.add_argument("--font", default="Intern", help="Preferred font family (default Intern).")
    main(ap.parse_args())
