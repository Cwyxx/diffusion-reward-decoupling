# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flow_grpo.rewards import multi_score


AVAILABLE_METRICS = [
    "pickscore", "imagereward", "aesthetic", "hpsv3", "deqa", "visualquality_r1",
    "ocr", "geneval", "wise",
]

# Metrics whose scoring functions require small batches.
SMALL_BATCH_METRICS = {"hpsv3", "visualquality_r1"}


def prepare_images(metric, image_paths):
    """Load images in the format each scorer expects."""
    if metric in {"hpsv3", "deqa", "visualquality_r1"}:
        return image_paths  # accept file paths directly
    if metric == "aesthetic":
        # aesthetic_score expects ndarray NHWC uint8.
        return np.stack([np.array(Image.open(p).convert("RGB")) for p in image_paths])
    # pickscore / imagereward / ocr accept PIL images.
    return [Image.open(p).convert("RGB") for p in image_paths]


def run_metric(metric, image_paths, prompts, metadatas, batch_size, device):
    if metric == "geneval":
        # Bypass multi_score / reward-server: run the official scorer in-process.
        from flow_grpo.geneval_local import score as geneval_score_local
        return [float(v) for v in geneval_score_local(image_paths, metadatas)]

    if metric == "wise":
        # WISE judging is HTTP-bound (vLLM remote endpoint), not GPU-bound,
        # so it doesn't share multi_score's batched-on-cuda contract.
        # main() routes it to _score_wise_in_place directly; reaching this
        # branch means an unexpected caller bypassed that dispatch.
        raise RuntimeError("metric=wise must be routed via _score_wise_in_place")

    scoring_fn = multi_score(device, {metric: 1.0})
    all_scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=metric):
        batch_paths = image_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        images = prepare_images(metric, batch_paths)
        score_details, _ = scoring_fn(images, batch_prompts, batch_meta)
        values = score_details[metric]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        all_scores.extend(float(v) for v in values)
    return all_scores


# ------------------------- WISE_Verified judge -------------------------
# The judge prompt template and binary score parsing are duplicated from
# evaluation/benchmarks/WISE/vllm_eval.py:88-177. They live inline here so
# Best-of-N scoring is self-contained and so the protocol applied to each
# image is reviewable in this file. If the WISE upstream protocol changes,
# update both copies (the standalone WISE evaluator at vllm_eval.py is for
# flat single-image-per-prompt eval; this one is for Best-of-N).

_WISE_USER_PROMPT_TEMPLATE = """Please evaluate this generated image for the WISE benchmark and return ONLY one binary score.

# WISE Text-to-Image Evaluation Protocol

## What WISE Is Evaluating
WISE is a knowledge-intensive text-to-image benchmark. Many prompts do not directly state the final visual answer. Instead, the model must use commonsense, cultural, scientific, spatial, or temporal knowledge to infer what should appear in the image.

Your job is not to judge whether the image is beautiful. Your job is to judge whether the generated image correctly realizes the knowledge-based meaning of the prompt and is visually usable.

## Input Fields

**PROMPT**
The original text-to-image prompt given to the image generation model. It may contain an implicit clue rather than the explicit final answer.

**EXPLANATION**
The reference interpretation used for judging. It explains the intended answer, the required knowledge reasoning chain, and the visual evidence that should appear in a correct image. Treat EXPLANATION as the ground-truth judging guide.

For example:
- If PROMPT says "the round pastry commonly shared during Mid-Autumn Festival family gatherings", EXPLANATION may specify mooncakes. A correct image should show mooncakes, not just any festival food.
- If PROMPT says "a plant kept for many days beside a bright one-sided window", EXPLANATION may specify phototropism. A correct image should show the plant bending toward the light source.
- If PROMPT says "a street in New York when it is midnight in Beijing", EXPLANATION may specify the corresponding local time and expected lighting/activity. A correct image should reflect that inferred local time, not simply show Beijing or generic night.

## How To Judge

Evaluate the image using these checks:
1. Does the image contain the main objects or scene required by the PROMPT?
2. Does it satisfy the intended knowledge-based answer described in the EXPLANATION?
3. Are important relations correct, such as spatial layout, temporal state, physical effect, biological behavior, cultural object, or scientific phenomenon?
4. Is the image visually usable for judging, without obvious collapse, severe deformation, unreadable main objects, or major artifacts?

## Binary Score

**Score: 1**
Give 1 only when both conditions are met:
- The image is semantically correct according to both PROMPT and EXPLANATION.
- The image has no obvious generation failure that prevents reliable judging.

Minor aesthetic weakness, ordinary composition, non-photorealistic style, or lack of artistic beauty should not by itself cause rejection if the semantic target is correct and the image is clear.

**Score: 0**
Give 0 if any of the following applies:
- The image misses the intended answer in EXPLANATION.
- The image only follows surface words in PROMPT but fails the required knowledge inference.
- Key objects, attributes, states, behaviors, or relations are missing or wrong.
- The image contradicts the prompt or explanation.
- The main visual evidence is ambiguous enough that a human judge could not confidently verify correctness.
- The image has obvious visual collapse, severe deformation, garbled main objects, impossible structure, or artifacts that interfere with evaluation.

If there is serious doubt, return 0.

## Output Format

Return exactly one line and nothing else:

Score: 0

or

Score: 1

---

PROMPT: "{prompt}"
EXPLANATION: "{explanation}"

Return only `Score: 0` or `Score: 1`."""


def _wise_build_messages(prompt: str, explanation: str, image_base64: str) -> list:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional text-to-image quality auditor. Evaluate the image strictly according to the protocol.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _WISE_USER_PROMPT_TEMPLATE.format(prompt=prompt, explanation=explanation),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        },
    ]


def _wise_extract_score(txt: str):
    match = re.search(r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b", txt, re.IGNORECASE)
    if match:
        return float(match.group(1))
    nums = re.findall(r"(?m)^\s*([01])\s*$", txt)
    if len(nums) == 1:
        return float(nums[0])
    return None


def _wise_chat_completion(messages, *, api_base, api_key, model, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Some Qwen variants still emit <think>...</think> blocks; strip them.
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"</think>\s*", "", content)
    return content.strip()


def _wise_judge_one(image_path, metadata, *, api_base, api_key, model, timeout, max_retries):
    if "Prompt" not in metadata or "Explanation" not in metadata:
        raise KeyError(
            f"WISE judge needs metadata.Prompt and metadata.Explanation; "
            f"row for {image_path} has keys {sorted(metadata.keys())}"
        )
    with open(image_path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()
    messages = _wise_build_messages(metadata["Prompt"], metadata["Explanation"], img64)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            txt = _wise_chat_completion(
                messages, api_base=api_base, api_key=api_key, model=model, timeout=timeout,
            )
            score = _wise_extract_score(txt)
            if score is not None:
                return float(score)
            last_err = f"score parse failed; raw={txt[:200]!r}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(
        f"WISE judge failed after {max_retries} attempts on {image_path}: {last_err}"
    )


def _score_wise_in_place(todo_rows):
    """Fill todo_rows[i]['scores']['wise'] in place via vLLM judge.

    Reads vLLM endpoint config from env vars (VLLM_API_BASE,
    VLLM_API_KEY, JUDGE_MODEL, WISE_MAX_WORKERS, WISE_TIMEOUT,
    WISE_MAX_RETRIES). On any unrecoverable judge error this raises
    and main()'s end-of-run rewrite never runs, dropping all in-memory
    scores; re-run resumes from whatever was previously on disk.
    """
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("JUDGE_MODEL", "Qwen3.5-35B-A3B")
    max_workers = int(os.environ.get("WISE_MAX_WORKERS", "32"))
    timeout = int(os.environ.get("WISE_TIMEOUT", "300"))
    max_retries = int(os.environ.get("WISE_MAX_RETRIES", "3"))

    n = len(todo_rows)
    if n == 0:
        return
    print(f"[wise] {n} images to score; api_base={api_base} model={model} workers={max_workers}")

    def task(i):
        r = todo_rows[i]
        score = _wise_judge_one(
            r["image_path"], r.get("metadata") or {},
            api_base=api_base, api_key=api_key, model=model,
            timeout=timeout, max_retries=max_retries,
        )
        return i, score

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, i) for i in range(n)]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=n, desc="wise"):
            i, score = fut.result()  # raises if judge gave up on any image
            todo_rows[i]["scores"]["wise"] = score


def main(args):
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f if line.strip()]
    # Sort key handles both legacy (sample_id only) and BoN ((sample_id, seed_index)) schemas.
    results.sort(key=lambda r: (r["sample_id"], r.get("seed_index", 0)))

    for metric in args.metrics:
        if args.force:
            todo = results
        else:
            todo = [r for r in results if metric not in r["scores"]]
        print(f"\n=== Scoring with {metric}: {len(todo)}/{len(results)} rows todo (force={args.force}) ===")
        if not todo:
            continue

        if metric == "wise":
            _score_wise_in_place(todo)
            continue

        image_paths = [r["image_path"] for r in todo]
        prompts = [r["prompt"] for r in todo]
        metadatas = [r.get("metadata") or {} for r in todo]

        bs = 1 if metric in SMALL_BATCH_METRICS else args.batch_size
        scores = run_metric(metric, image_paths, prompts, metadatas, bs, "cuda")
        assert len(scores) == len(todo)
        for r, s in zip(todo, scores):
            r["scores"][metric] = s
        torch.cuda.empty_cache()

    # Atomic rewrite of the jsonl (multi-row file; partial-write loss is bad).
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, results_path)

    agg = defaultdict(list)
    for r in results:
        for name, value in r["scores"].items():
            if isinstance(value, (int, float)):
                agg[name].append(value)
    averages = {name: float(np.mean(v)) for name, v in agg.items()}

    avg_path = os.path.join(args.output_dir, "average_scores.json")
    with open(avg_path, "w") as f:
        json.dump(averages, f, indent=4)

    print("\n--- Average Scores (all rows) ---")
    for name, avg in sorted(averages.items()):
        print(f"{name:<20}: {avg:.6f}")
    print(f"Saved to {avg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics on images generated by generate-images.py "
                    "or generate-images-bestofn.py, via flow_grpo.rewards.multi_score."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing evaluation_results.jsonl and images/.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        choices=AVAILABLE_METRICS,
        help="One or more metrics to compute. Scores are accumulated into evaluation_results.jsonl.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score rows that already have the requested metric (default: skip them).",
    )
    main(parser.parse_args())
