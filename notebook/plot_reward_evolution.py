import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = {
    "human_preference": ["pickscore", "imagereward", "hpsv3"],
    "image_quality": ["deqa", "visualquality_r1", "aesthetic"],
    "aigi_detector": ["omniaid_remote"],
}

CATEGORY_TITLES = {
    "human_preference": "Human Preference",
    "image_quality": "Image Quality",
    "aigi_detector": "AIGI Detector",
}

CATEGORY_COLORS = {
    "human_preference": ["#1f77b4", "#4a90d9", "#7eb6e6"],  # blues
    "image_quality":    ["#2ca02c", "#5bbf5b", "#8fd48f"],  # greens
    "aigi_detector":    ["#ff7f0e", "#ffa14a", "#ffc485"],  # oranges
}

ALL_REWARDS = [r for rs in CATEGORIES.values() for r in rs]


def parse_categories(s):
    s = s.strip().lower()
    if s in ("", "all"):
        return list(CATEGORIES.keys())
    out = []
    for part in s.split(","):
        p = part.strip().replace("-", "_")
        if p not in CATEGORIES:
            raise ValueError(
                f"Unknown category: {part}. Choose from: {list(CATEGORIES.keys())} or 'all'."
            )
        out.append(p)
    return out


def minmax_normalize(values):
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    return norm, lo, hi


def load_prompt_json(input_dir, prompt_idx):
    json_path = os.path.join(input_dir, f"prompt_{prompt_idx}_rewards.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Reward JSON not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def plot_categories(scores, prompt_text, prompt_idx, categories, out_path):
    ncols = len(categories)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False
    )
    axes = axes[0]

    for ax, cat in zip(axes, categories):
        colors = CATEGORY_COLORS[cat]
        plotted_any = False
        for i, reward in enumerate(CATEGORIES[cat]):
            raw = scores.get(reward)
            if raw is None:
                continue
            norm, lo, hi = minmax_normalize(raw)
            steps = np.arange(len(norm))
            label = f"{reward}  [{lo:.3g}, {hi:.3g}]"
            ax.plot(
                steps, norm, marker="o", linewidth=2,
                color=colors[i % len(colors)], label=label,
            )
            plotted_any = True

        ax.set_title(CATEGORY_TITLES[cat], fontsize=13, fontweight="bold")
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Min-max normalized reward")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        if plotted_any:
            ax.legend(loc="best", fontsize=8, framealpha=0.85)
        else:
            ax.text(
                0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", color="gray",
            )

    _finalize(fig, prompt_text, prompt_idx, out_path)


def plot_compare_pair(scores, prompt_text, prompt_idx, pair, out_path):
    """Two rewards on one plot with dual y-axes (original values)."""
    r1, r2 = pair
    for r in pair:
        if r not in ALL_REWARDS:
            raise ValueError(
                f"Unknown reward: {r}. Supported: {ALL_REWARDS}"
            )
        if scores.get(r) is None:
            raise ValueError(f"Reward '{r}' has no scores in this prompt's JSON.")

    raw1 = np.asarray(scores[r1], dtype=np.float64)
    raw2 = np.asarray(scores[r2], dtype=np.float64)
    steps1 = np.arange(len(raw1))
    steps2 = np.arange(len(raw2))

    color1, color2 = "#1f77b4", "#d62728"  # blue, red
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax2 = ax1.twinx()

    l1, = ax1.plot(
        steps1, raw1, marker="o", linewidth=2.2, color=color1, label=r1,
    )
    l2, = ax2.plot(
        steps2, raw2, marker="s", linewidth=2.2, color=color2, linestyle="--", label=r2,
    )

    ax1.set_xlabel("Denoising step")
    ax1.set_ylabel(f"{r1} (original)", color=color1)
    ax2.set_ylabel(f"{r2} (original)", color=color2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Integer x-ticks for step indices.
    n_steps = max(len(raw1), len(raw2))
    ax1.set_xticks(np.arange(n_steps))

    ax1.legend(
        handles=[l1, l2], labels=[r1, r2],
        loc="best", framealpha=0.9,
    )

    _finalize(fig, prompt_text, prompt_idx, out_path,
              title_suffix=f" — {r1} vs {r2}")


def _finalize(fig, prompt_text, prompt_idx, out_path, title_suffix=""):
    prompt_preview = prompt_text if len(prompt_text) <= 140 else prompt_text[:137] + "..."
    fig.suptitle(
        f"Reward evolution — prompt_{prompt_idx}{title_suffix}\n{prompt_preview}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


def main(args):
    data = load_prompt_json(args.input_dir, args.prompt_idx)
    scores = data.get("scores", {})
    prompt_text = data.get("prompt", "")

    if args.compare:
        pair = [p.strip() for p in args.compare.split(",") if p.strip()]
        if len(pair) != 2:
            raise ValueError("--compare requires exactly 2 reward names, e.g. 'pickscore,aesthetic'")
        out_path = args.output or os.path.join(
            args.input_dir,
            f"reward_compare_{pair[0]}_vs_{pair[1]}_prompt{args.prompt_idx}.png",
        )
        plot_compare_pair(scores, prompt_text, args.prompt_idx, pair, out_path)
    else:
        categories = parse_categories(args.categories)
        out_path = args.output or os.path.join(
            args.input_dir, f"reward_evolution_prompt{args.prompt_idx}.png"
        )
        plot_categories(scores, prompt_text, args.prompt_idx, categories, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-step reward evolution for a single prompt. "
                    "Two modes: (1) category mode — one subplot per category, "
                    "rewards min-max normalized; (2) compare mode (--compare a,b) "
                    "— two rewards on a single plot with dual y-axes (original values)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/chenweiyan/Downloads/reward-evolution-v2",
        help="Directory containing prompt_{idx}_rewards.json files.",
    )
    parser.add_argument("--prompt_idx", type=int, default=1)
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="[Category mode] Comma-separated category names, or 'all'. "
             "Choices: human_preference, image_quality, aigi_detector. "
             "Ignored when --compare is set.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="[Compare mode] Two reward names separated by comma, e.g. "
             "'pickscore,aesthetic'. Plots them on one figure with dual "
             "y-axes using original (non-normalized) values.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. Defaults: category mode -> "
             "<input_dir>/reward_evolution_prompt{idx}.png; compare mode -> "
             "<input_dir>/reward_compare_{A}_vs_{B}_prompt{idx}.png.",
    )
    args = parser.parse_args()
    main(args)

# Examples:
#   # Category mode (original):
#   python plot_reward_evolution.py --prompt_idx 1
#   python plot_reward_evolution.py --prompt_idx 1 --categories human_preference,image_quality
#
#   # Compare mode (two rewards, dual y-axes, original values):
#   python plot_reward_evolution.py --prompt_idx 1 --compare pickscore,aesthetic
#   python plot_reward_evolution.py --prompt_idx 1 --compare hpsv3,omniaid_remote
