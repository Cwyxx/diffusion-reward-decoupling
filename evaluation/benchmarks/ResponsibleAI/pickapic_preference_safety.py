"""Audit Pick-a-Pic v2 preference pairs for unsafe images with ShieldGemma 2.

For every preference row in the visualization-data dump, this scores the
`win_image-*.png` and `lose_image-*.png` with the same `shieldgemma_fp16`
judge used elsewhere in this repo (google/shieldgemma-2-4b-it, float32 forward,
unsafe = OR over sexually-explicit / dangerous / violence-gore). It then reports
whether win/lose images are flagged unsafe and how the two sides of each pair
compare.

Layout assumed (one preference pair per `row_*` directory):

    <data-root>/
      train-00000-of-00645-.../
        row_0/
          caption.txt
          info.json
          win_image-<uuid>.png
          lose_image-<uuid>.png
        row_1/ ...
      train-00001-of-00645-.../ ...

Scoring is sharded across all visible GPUs: one ShieldGemma instance per GPU,
each pinned to `cuda:<id>`. Per-image verdicts are streamed to
`shieldgemma_fp16_per_image.jsonl` (one JSON object per line), which makes the
run resumable -- on restart, images already present in that file are skipped.
The aggregate report (`summary.json`) is recomputed from the JSONL and can be
rebuilt without a GPU via `--report-only`.

Run (8x3090):

    conda activate visualquality       # needs transformers>=4.50 + HF gemma access
    python evaluation/benchmarks/ResponsibleAI/pickapic_preference_safety.py \
        --data-root /data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick-a-pic-v2/visualization-data \
        --num-gpus 8

Smoke test on a few train shards first:

    python .../pickapic_preference_safety.py --limit-shards 1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

# The scorer lives next to this file and has no package-relative imports, so a
# plain sibling import keeps this script runnable as a standalone file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_ROOT = (
    "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/"
    "pick-a-pic-v2/visualization-data"
)
PER_IMAGE_FILENAME = "shieldgemma_fp16_per_image.jsonl"
SUMMARY_FILENAME = "shieldgemma_fp16_summary.json"

UNSAFE_FLAG_KEY = "shieldgemma_fp16-unsafe"
POLICY_FLAG_KEYS = (
    "shieldgemma_fp16-sexually-explicit",
    "shieldgemma_fp16-dangerous",
    "shieldgemma_fp16-violence-gore",
)
ROLES = ("win", "lose")


# --------------------------------------------------------------------------- #
# Task enumeration
# --------------------------------------------------------------------------- #
def _find_role_image(row_dir: str, role: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(row_dir, f"{role}_image-*.png")))
    return matches[0] if matches else None


def enumerate_tasks(data_root: str, limit_shards: int | None) -> list[dict]:
    """Return one task per (row, role) image found under the data root."""
    shard_dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "train-*")) if os.path.isdir(d)
    )
    if limit_shards is not None:
        shard_dirs = shard_dirs[:limit_shards]

    tasks: list[dict] = []
    for shard_dir in shard_dirs:
        shard = os.path.basename(shard_dir)
        row_dirs = sorted(
            d for d in glob.glob(os.path.join(shard_dir, "row_*")) if os.path.isdir(d)
        )
        for row_dir in row_dirs:
            row = os.path.basename(row_dir)
            row_id = f"{shard}/{row}"
            for role in ROLES:
                image_path = _find_role_image(row_dir, role)
                if image_path is None:
                    print(f"[warn] missing {role}_image in {row_id}", file=sys.stderr)
                    continue
                tasks.append(
                    {"row_id": row_id, "role": role, "image_path": image_path}
                )
    return tasks


def _load_done_paths(per_image_path: str) -> set[str]:
    """Image paths already scored, read back from the streamed JSONL."""
    if not os.path.exists(per_image_path):
        return set()
    done: set[str] = set()
    with open(per_image_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["image_path"])
            except (json.JSONDecodeError, KeyError):
                # tolerate a partially written trailing line from a hard kill
                continue
    return done


# --------------------------------------------------------------------------- #
# GPU worker
# --------------------------------------------------------------------------- #
def _worker(gpu_id, shard_tasks, out_path, batch_size, threshold):
    """Score one shard of images on cuda:<gpu_id>, streaming JSONL to out_path."""
    import torch  # noqa: F401  (import inside the spawned process)
    from PIL import Image
    from shieldgemma_scorer import score_images

    device = f"cuda:{gpu_id}"
    n = len(shard_tasks)
    print(f"[gpu{gpu_id}] {n} images, batch_size={batch_size}, device={device}")

    with open(out_path, "a", buffering=1) as fh:  # line-buffered for resume safety
        for start in range(0, n, batch_size):
            batch = shard_tasks[start : start + batch_size]
            images = [Image.open(t["image_path"]).convert("RGB") for t in batch]
            scores = score_images(
                images, device=device, threshold=threshold, batch_size=batch_size
            )
            if len(scores) != len(batch):
                raise RuntimeError(
                    f"[gpu{gpu_id}] scorer returned {len(scores)} for {len(batch)} images"
                )
            for task, score in zip(batch, scores):
                record = {
                    "row_id": task["row_id"],
                    "role": task["role"],
                    "image_path": task["image_path"],
                    **score,
                }
                fh.write(json.dumps(record) + "\n")
            if (start // batch_size) % 50 == 0:
                done = min(start + batch_size, n)
                print(f"[gpu{gpu_id}] {done}/{n}", flush=True)
    print(f"[gpu{gpu_id}] done ({n} images)", flush=True)


def score_all(tasks, out_path, num_gpus, batch_size, threshold):
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        raise SystemExit("[error] CUDA not available; ShieldGemma scan needs GPUs")
    available = torch.cuda.device_count()
    num_gpus = min(num_gpus, available)
    if num_gpus < 1:
        raise SystemExit("[error] no visible CUDA devices")
    print(f"[info] scoring {len(tasks)} images across {num_gpus} GPU(s)")

    # Round-robin so each worker gets an interleaved, balanced slice.
    shards = [tasks[i::num_gpus] for i in range(num_gpus)]
    # Each worker appends to its own file so processes never contend on one fd.
    worker_paths = [f"{out_path}.gpu{i}" for i in range(num_gpus)]

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, shards[gpu_id], worker_paths[gpu_id], batch_size, threshold),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"[error] a GPU worker exited with code {p.exitcode}")

    # Concatenate per-worker partials into the canonical JSONL.
    with open(out_path, "a") as out_fh:
        for wp in worker_paths:
            if not os.path.exists(wp):
                continue
            with open(wp, "r") as in_fh:
                for line in in_fh:
                    out_fh.write(line)
            os.remove(wp)


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #
def aggregate(per_image_path: str) -> dict:
    """Build the win/lose unsafe report from the per-image JSONL."""
    # role -> flag_key -> unsafe count, plus per-role totals
    role_unsafe = {role: Counter() for role in ROLES}
    role_total = Counter()
    # pair-level: row_id -> {win: bool, lose: bool}
    pair_flag: dict[str, dict[str, int]] = {}

    with open(per_image_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            role = rec["role"]
            role_total[role] += 1
            for key in (UNSAFE_FLAG_KEY, *POLICY_FLAG_KEYS):
                role_unsafe[role][key] += int(rec.get(key, 0))
            pair_flag.setdefault(rec["row_id"], {})[role] = int(
                rec.get(UNSAFE_FLAG_KEY, 0)
            )

    def rate(role, key):
        tot = role_total[role]
        return (role_unsafe[role][key] / tot) if tot else 0.0

    per_role = {}
    for role in ROLES:
        per_role[role] = {
            "n_images": role_total[role],
            "unsafe_count": role_unsafe[role][UNSAFE_FLAG_KEY],
            "unsafe_rate": rate(role, UNSAFE_FLAG_KEY),
            "policy_rates": {k: rate(role, k) for k in POLICY_FLAG_KEYS},
            "policy_counts": {k: role_unsafe[role][k] for k in POLICY_FLAG_KEYS},
        }

    # Pair-level 2x2 over rows where BOTH images were scored.
    complete_pairs = {
        rid: f for rid, f in pair_flag.items() if "win" in f and "lose" in f
    }
    cells = Counter()
    for f in complete_pairs.values():
        cells[(f["win"], f["lose"])] += 1
    n_pairs = len(complete_pairs)

    def cell_rate(c):
        return (cells[c] / n_pairs) if n_pairs else 0.0

    joint = {
        "n_complete_pairs": n_pairs,
        "both_safe": {"count": cells[(0, 0)], "rate": cell_rate((0, 0))},
        "win_unsafe_only": {"count": cells[(1, 0)], "rate": cell_rate((1, 0))},
        "lose_unsafe_only": {"count": cells[(0, 1)], "rate": cell_rate((0, 1))},
        "both_unsafe": {"count": cells[(1, 1)], "rate": cell_rate((1, 1))},
        "either_unsafe": {
            "count": n_pairs - cells[(0, 0)],
            "rate": 1.0 - cell_rate((0, 0)) if n_pairs else 0.0,
        },
    }

    return {
        "detector": "shieldgemma_fp16 (google/shieldgemma-2-4b-it, OR of 3 policies)",
        "per_role": per_role,
        "pair_level": joint,
    }


def print_report(summary: dict) -> None:
    print("\n================ ShieldGemma_fp32 Pick-a-Pic v2 audit ================")
    for role in ROLES:
        r = summary["per_role"][role]
        print(
            f"\n[{role.upper()}]  n={r['n_images']}  "
            f"unsafe={r['unsafe_count']} ({r['unsafe_rate']*100:.2f}%)"
        )
        for k in POLICY_FLAG_KEYS:
            short = k.replace("shieldgemma_fp16-", "")
            print(
                f"    {short:<18} {r['policy_counts'][k]:>7}  "
                f"({r['policy_rates'][k]*100:.2f}%)"
            )
    j = summary["pair_level"]
    print(f"\n[PAIR-LEVEL]  complete pairs = {j['n_complete_pairs']}")
    for k in ("both_safe", "win_unsafe_only", "lose_unsafe_only", "both_unsafe",
              "either_unsafe"):
        print(f"    {k:<18} {j[k]['count']:>7}  ({j[k]['rate']*100:.2f}%)")
    print("======================================================================\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="visualization-data dir containing train-* shards")
    ap.add_argument("--output-dir", default=None,
                    help="where to write JSONL + summary (default: <data-root>/"
                         "unsafe-shieldgemma_fp16)")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("SHIELDGEMMA_BATCH_SIZE", "1")))
    ap.add_argument("--threshold", type=float,
                    default=float(os.environ.get("SHIELDGEMMA_THRESHOLD", "0.5")))
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="only scan the first N train-* shards (smoke test)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip scoring; just (re)aggregate the existing JSONL")
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.join(args.data_root, "unsafe-shieldgemma_fp16")
    os.makedirs(out_dir, exist_ok=True)
    per_image_path = os.path.join(out_dir, PER_IMAGE_FILENAME)
    summary_path = os.path.join(out_dir, SUMMARY_FILENAME)

    if not args.report_only:
        tasks = enumerate_tasks(args.data_root, args.limit_shards)
        if not tasks:
            raise SystemExit(f"[error] no images found under {args.data_root}")
        done = _load_done_paths(per_image_path)
        if done:
            before = len(tasks)
            tasks = [t for t in tasks if t["image_path"] not in done]
            print(f"[info] resume: {len(done)} already scored, "
                  f"{len(tasks)}/{before} remaining")
        if tasks:
            score_all(tasks, per_image_path, args.num_gpus,
                      args.batch_size, args.threshold)
        else:
            print("[info] nothing to score; all images already in JSONL")

    if not os.path.exists(per_image_path):
        raise SystemExit(f"[error] no per-image results at {per_image_path}")
    summary = aggregate(per_image_path)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print_report(summary)
    print(f"[info] per-image JSONL: {per_image_path}")
    print(f"[info] summary JSON:    {summary_path}")


if __name__ == "__main__":
    main()
