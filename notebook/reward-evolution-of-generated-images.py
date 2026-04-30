import argparse
import gc
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flow_grpo.rewards import multi_score


PROMPT_DIR_RE = re.compile(r"^prompt_(\d+)$")

# Rewards analyzed independently. Each is computed with weight 1.0, so the
# "avg" field produced by multi_score equals the raw score.
DEFAULT_REWARDS = [
    "pickscore",
    "imagereward",
    "hpsv3",
    "deqa",
    "visualquality_r1",
    "omniaid_remote",
    "aesthetic",
]

# How each reward wants its images delivered.
#   "tensor" -> NCHW float tensor in [0, 1]
#   "paths"  -> list of file paths
REWARD_FORMATS = {
    "pickscore": "tensor",
    "imagereward": "tensor",
    "hpsv3": "paths",
    "deqa": "paths",    # local deqa accepts tensor or paths; use paths per user preference
    "visualquality_r1": "paths",
    "omniaid_remote": "tensor",
    "aesthetic": "tensor",
}


def load_prompts(dataset, split="test", limit=10):
    """Mirror TextPromptDataset in the-evolution-of-generated-images.py."""
    path = os.path.join(
        Path(__file__).resolve().parent.parent, "dataset", dataset, f"{split}.txt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    with open(path, "r") as f:
        prompts = [line.strip() for line in f.readlines()][0:limit]
    return prompts


def find_prompt_dirs(input_dir):
    items = []
    for name in os.listdir(input_dir):
        m = PROMPT_DIR_RE.match(name)
        if m:
            items.append((int(m.group(1)), os.path.join(input_dir, name)))
    items.sort(key=lambda x: x[0])
    return items


def load_step_images(prompt_dir):
    paths = sorted(glob.glob(os.path.join(prompt_dir, "*step_*.png")))
    if not paths:
        return [], None
    images = np.stack(
        [np.array(Image.open(p).convert("RGB")) for p in paths], axis=0
    )  # (N, H, W, C) uint8
    return paths, images


def normalize_reward_name(name):
    # Accept hyphens or underscores, e.g. 'visualquality-r1'
    return name.strip().replace("-", "_")


def parse_rewards(s):
    names = []
    for part in s.split(","):
        part = normalize_reward_name(part)
        if part:
            names.append(part)
    return names


def _to_jsonable(scores):
    if hasattr(scores, "tolist"):
        return scores.tolist()
    if isinstance(scores, dict):
        return {k: _to_jsonable(v) for k, v in scores.items()}
    return list(scores)


def _merge_per_prompt_json(path, new_res):
    """Merge new_res into existing file at `path` (if any), unioning `scores`.

    Enables running the script multiple times (once per reward, in different
    conda envs) without losing previously computed scores.
    """
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        merged_scores = {**existing.get("scores", {}), **new_res.get("scores", {})}
        merged = {**existing, **new_res, "scores": merged_scores}
    else:
        merged = new_res
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)


def _merge_summary_json(path, new_summary):
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing_rewards = existing.get("rewards", [])
        merged_rewards = list(dict.fromkeys(existing_rewards + new_summary.get("rewards", [])))
        merged_results = dict(existing.get("results", {}))
        for k, v in new_summary.get("results", {}).items():
            if k in merged_results:
                merged_scores = {
                    **merged_results[k].get("scores", {}),
                    **v.get("scores", {}),
                }
                merged_results[k] = {**merged_results[k], **v, "scores": merged_scores}
            else:
                merged_results[k] = v
        merged = {**existing, **new_summary, "rewards": merged_rewards, "results": merged_results}
    else:
        merged = new_summary
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rewards = parse_rewards(args.rewards) if args.rewards else DEFAULT_REWARDS
    unknown = [r for r in rewards if r not in REWARD_FORMATS]
    if unknown:
        raise ValueError(
            f"Unknown reward(s): {unknown}. Supported: {list(REWARD_FORMATS.keys())}"
        )
    print(f"Analyzing rewards: {rewards}")

    prompts_all = load_prompts(args.dataset, split=args.split, limit=args.limit)
    prompt_dirs = find_prompt_dirs(args.input_dir)
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt_* subdirectories under {args.input_dir}")
    print(f"Found {len(prompt_dirs)} prompt directories; dataset has {len(prompts_all)} prompts")

    # Load step images once per prompt and reuse across rewards.
    prompt_data = {}
    for idx, pdir in prompt_dirs:
        if idx >= len(prompts_all):
            print(f"Skipping {pdir}: idx {idx} out of range for dataset prompts.")
            continue
        paths, images_np = load_step_images(pdir)
        if not paths:
            print(f"Skipping {pdir}: no step_*.png found.")
            continue
        prompt_data[idx] = {
            "prompt_text": prompts_all[idx],
            "paths": paths,
            "images_np": images_np,
            "dir": pdir,
        }

    if not prompt_data:
        raise RuntimeError("No usable prompt directories found.")

    # Initialize per-prompt result skeletons.
    results = {
        idx: {
            "prompt_idx": idx,
            "prompt": data["prompt_text"],
            "target_dir": data["dir"],
            "steps": [os.path.basename(p) for p in data["paths"]],
            "scores": {},
        }
        for idx, data in prompt_data.items()
    }

    # Outer loop: reward — load scorer once, score every prompt, then free.
    for reward_name in rewards:
        fmt = REWARD_FORMATS[reward_name]
        print(f"\n=== Reward: {reward_name} (format={fmt}) ===")
        scoring_fn = multi_score(device, {reward_name: 1.0})

        for idx, data in tqdm(prompt_data.items(), desc=f"[{reward_name}]"):
            n = len(data["paths"])
            if fmt == "paths":
                images_input = data["paths"]
            else:
                images_input = (
                    torch.from_numpy(data["images_np"]).permute(0, 3, 1, 2).float() / 255.0
                )

            prompts_batch = [data["prompt_text"]] * n
            metadatas = [{} for _ in range(n)]

            # Chunk the per-prompt batch to avoid OOM for heavy rewards (hpsv3).
            bs = args.batch_size if args.batch_size and args.batch_size > 0 else n

            try:
                chunked = []
                for start in range(0, n, bs):
                    end = min(start + bs, n)
                    img_chunk = images_input[start:end]
                    p_chunk = prompts_batch[start:end]
                    m_chunk = metadatas[start:end]
                    sd, _ = scoring_fn(img_chunk, p_chunk, m_chunk)
                    vals = _to_jsonable(sd[reward_name])
                    if isinstance(vals, list):
                        chunked.extend(vals)
                    else:
                        chunked.append(vals)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                scores = chunked
            except Exception as e:
                print(f"  prompt_{idx}: {reward_name} failed: {e}")
                scores = None

            results[idx]["scores"][reward_name] = scores

            if isinstance(scores, list) and scores and isinstance(scores[0], (int, float)):
                trend = ", ".join(f"{s:.3f}" for s in scores)
                print(f"  prompt_{idx}: {trend}")

        # Free scorer before loading the next one.
        del scoring_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save per-prompt and summary JSONs. Merge with any existing file so
    # per-reward invocations (in different conda envs) accumulate.
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, res in results.items():
        out_json = os.path.join(args.output_dir, f"prompt_{idx}_rewards.json")
        _merge_per_prompt_json(out_json, res)

    summary_path = os.path.join(args.output_dir, "all_prompts_rewards.json")
    _merge_summary_json(
        summary_path,
        {
            "dataset": args.dataset,
            "split": args.split,
            "rewards": rewards,
            "input_dir": args.input_dir,
            "results": {f"prompt_{k}": v for k, v in results.items()},
        },
    )
    print(f"\nSaved per-prompt JSONs and summary to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-step trends for each reward independently, "
                    "across all prompt_* directories produced by the-evolution-of-generated-images.py."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/the-evoluation-of-generated-images",
        help="Root directory containing prompt_0, prompt_1, ... subdirs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPDv3",
        help="Dataset name under ../dataset/ to read {split}.txt from.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of prompts loaded from {split}.txt (matches the-evolution-of-generated-images.py).",
    )
    parser.add_argument(
        "--rewards",
        type=str,
        default="",
        help=(
            "Comma-separated reward names to analyze independently. "
            "Empty string uses the default set: "
            "pickscore,imagereward,hpsv3,deqa,visualquality_r1,omniaid_remote,aesthetic."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/reward-evolution",
        help="Directory to save per-prompt reward JSONs and summary.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Chunk size for per-prompt scoring (0 = full batch). Set a small value for memory-heavy rewards like hpsv3.",
    )
    args = parser.parse_args()
    main(args)

# Example:
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 python reward-evolution-of-generated-images.py
#   # (runs the 7 default rewards; omniaid_remote requires the scoring server running on 127.0.0.1:18092)
