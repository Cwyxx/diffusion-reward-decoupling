"""Generate N images per prompt for Best-of-N ceiling eval (SD-v1.5 / SDXL).

User-facing CLI: --gpus 0,1,2,3 dispatches to those GPUs in parallel.

Internally: launcher process forks one subprocess per GPU. Each subprocess
runs in worker mode (detected by the BESTOFN_RANK env var set by the
launcher) and handles items[rank::world_size]. Workers write to per-rank
evaluation_results.rankR.jsonl; the launcher merges them into
evaluation_results.jsonl when all workers exit successfully.
"""
import argparse
import glob
import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DATASET_ROOT = os.path.join(_REPO_ROOT, "dataset")


# ---------- Dataset loading ----------

def _load_txt(path):
    with open(path, "r") as f:
        return [{"prompt": ln.strip(), "metadata": None}
                for ln in f if ln.strip()]


def _load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            items.append({"prompt": row["prompt"], "metadata": row})
    return items


_DATASET_LOADERS = {
    "drawbench-unique": ("test.txt", _load_txt),
    "ocr":              ("test.txt", _load_txt),
    "geneval":          ("test_metadata.jsonl", _load_jsonl),
}


def load_prompts(dataset_name):
    if dataset_name not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset_name!r}; known: {sorted(_DATASET_LOADERS)}")
    fname, loader = _DATASET_LOADERS[dataset_name]
    return loader(os.path.join(DATASET_ROOT, dataset_name, fname))


# ---------- jsonl helpers ----------

def main_jsonl_path(out_dir):
    return os.path.join(out_dir, "evaluation_results.jsonl")


def rank_jsonl_path(out_dir, rank):
    return os.path.join(out_dir, f"evaluation_results.rank{rank}.jsonl")


def load_rows(jsonl_path):
    """Return dict keyed by (sample_id, seed_index)."""
    rows = {}
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            rows[(r["sample_id"], r["seed_index"])] = r
    return rows


def append_row(jsonl_path, row):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def merge_rank_files(out_dir):
    """Merge any evaluation_results.rank*.jsonl into the main jsonl, then delete them.

    Idempotent: safe to call when no rank files exist or the main jsonl is partial.
    """
    rank_paths = sorted(glob.glob(os.path.join(out_dir, "evaluation_results.rank*.jsonl")))
    if not rank_paths:
        return

    rows = load_rows(main_jsonl_path(out_dir))
    for rp in rank_paths:
        for k, v in load_rows(rp).items():
            rows[k] = v  # rank subsets are disjoint by prompt; no real conflict

    with open(main_jsonl_path(out_dir), "w") as f:
        for key in sorted(rows.keys()):
            f.write(json.dumps(rows[key]) + "\n")

    for rp in rank_paths:
        os.remove(rp)


# ---------- Launcher ----------

def get_scheduler_class(recipe):
    """Discover the scheduler class name without loading the full pipeline."""
    from diffusers import DiffusionPipeline
    pretrained = recipe.repo_id if recipe.load_kind == "full" else recipe.base_model_id
    config = DiffusionPipeline.load_config(pretrained)
    # config["scheduler"] is e.g. ["diffusers", "PNDMScheduler"]
    return config["scheduler"][1]


def build_manifest(args, scheduler_class):
    from evaluation.checkpoints.registry import get_recipe
    from evaluation.manifest import GenerationManifest
    recipe = get_recipe(args.method)
    return GenerationManifest(
        method=args.method,
        dataset=args.dataset,
        checkpoint_id=recipe.repo_id,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        scheduler_class=scheduler_class,
        max_seed_generated=-1,
    )


def run_launcher(args):
    from evaluation.checkpoints.registry import get_recipe
    from evaluation.manifest import check_consistency, read_manifest, write_manifest

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        sys.exit("--gpus must be a non-empty comma-separated list of GPU ids")
    if len(gpus) != len(set(gpus)):
        sys.exit(f"--gpus has duplicates: {gpus}")
    world_size = len(gpus)

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    # Consolidate any stale rank files from a prior interrupted run before
    # writing the new manifest.
    merge_rank_files(out_dir)

    recipe = get_recipe(args.method)
    scheduler_class = get_scheduler_class(recipe)
    incoming = build_manifest(args, scheduler_class)

    existing = read_manifest(out_dir)
    if existing is not None and not args.force_regenerate:
        check_consistency(existing, incoming)
        incoming.max_seed_generated = max(existing.max_seed_generated, incoming.max_seed_generated)
    write_manifest(out_dir, incoming)

    procs = []
    for rank, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["BESTOFN_RANK"] = str(rank)
        env["BESTOFN_WORLD_SIZE"] = str(world_size)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--method", args.method,
            "--dataset", args.dataset,
            "--output_dir", args.output_dir,
            "--n_max", str(args.n_max),
            "--num_inference_steps", str(args.num_inference_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--resolution", str(args.resolution),
            "--gpus", args.gpus,  # passed through; worker ignores it (env var wins)
        ]
        procs.append(subprocess.Popen(cmd, env=env))

    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        sys.exit(f"Worker exit codes: {rcs}; not merging or finalizing manifest.")

    merge_rank_files(out_dir)

    final = read_manifest(out_dir)
    final.max_seed_generated = max(final.max_seed_generated, args.n_max - 1)
    write_manifest(out_dir, final)


# ---------- Worker ----------

def run_worker(args, rank, world_size):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    from evaluation.checkpoints import load_pipeline

    out_dir = args.output_dir
    images_dir = os.path.join(out_dir, "images")
    rank_jsonl = rank_jsonl_path(out_dir, rank)

    device = torch.device("cuda")
    dtype = torch.float16 if args.method.endswith("-sdxl") else torch.float32
    pipeline = load_pipeline(args.method, device=device, dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)

    items = load_prompts(args.dataset)
    my_indices = list(range(rank, len(items), world_size))

    done = set()
    done.update(load_rows(main_jsonl_path(out_dir)).keys())
    done.update(load_rows(rank_jsonl).keys())

    pbar = tqdm(
        total=len(my_indices) * args.n_max,
        desc=f"rank{rank}",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for sample_id in my_indices:
        item = items[sample_id]
        prompt = item["prompt"]
        metadata = item["metadata"]
        sid_dir = os.path.join(images_dir, f"{sample_id:05d}")
        os.makedirs(sid_dir, exist_ok=True)

        for seed_index in range(args.n_max):
            img_path = os.path.join(sid_dir, f"{seed_index:05d}.png")
            row_key = (sample_id, seed_index)

            image_exists = os.path.exists(img_path)
            row_exists = row_key in done

            if image_exists and row_exists:
                pbar.update(1)
                continue

            if not image_exists:
                generator = torch.Generator(device).manual_seed(seed_index)
                with torch.no_grad():
                    result = pipeline(
                        prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.resolution,
                        width=args.resolution,
                        output_type="pt",
                        generator=generator,
                    )
                img_tensor = result.images[0]
                arr = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(arr).save(img_path)

            if not row_exists:
                row = {
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "prompt": prompt,
                    "image_path": img_path,
                    "scores": {},
                }
                if metadata is not None:
                    row["metadata"] = metadata
                append_row(rank_jsonl, row)
                done.add(row_key)

            pbar.update(1)

    pbar.close()


# ---------- main ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate N images per prompt for Best-of-N ceiling eval (SD-v1.5 / SDXL).",
    )
    ap.add_argument("--method", required=True,
                    help="SD-v1.5: base, dpo, kto, spo, smpo, dro, inpo. "
                         "SDXL: base-sdxl, dpo-sdxl, spo-sdxl, inpo-sdxl.")
    ap.add_argument("--dataset", required=True,
                    help="One of: drawbench-unique, ocr, geneval")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpus", required=True, metavar="0,1,2,3",
                    help="Comma-separated GPU IDs to dispatch to.")
    ap.add_argument("--n_max", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--force-regenerate", dest="force_regenerate", action="store_true",
                    help="Bypass manifest consistency check (still skips already-generated "
                         "files; delete output_dir manually to truly start over).")
    return ap.parse_args()


def main():
    args = parse_args()
    rank_env = os.environ.get("BESTOFN_RANK")
    if rank_env is not None:
        rank = int(rank_env)
        world_size = int(os.environ["BESTOFN_WORLD_SIZE"])
        run_worker(args, rank, world_size)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
