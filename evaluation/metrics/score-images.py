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
    "ocr", "geneval", "wise", "dpg-score", "dpg-score-mplug", "spatial-geneval",
    "sd-safety-checker", "shieldgemma",
    "dalleval-bias-gender", "dalleval-bias-attribute", "dalleval-bias-skintone",
]

# Metrics whose scoring functions require small batches.
SMALL_BATCH_METRICS = {"hpsv3", "visualquality_r1"}


def _dalleval_attr_keys():
    # Importing the scorer just to read ATTR_SLUGS would pull lavis in; keep
    # this list inline and aligned with blip2_scorer.ATTRIBUTES (asserted there).
    return tuple(f"dalleval-attr-{slug}" for slug in (
        "boots", "slippers", "jeans", "shorts", "slacks",
        "dress", "skirt", "suit", "shirt", "uniform",
        "jacket", "hat", "tie", "mask", "gloves",
    ))


# Some metric names dispatch to scorers that write benchmark-specific score keys.
METRIC_OUTPUT_KEYS = {
    "sd-safety-checker": ("sd-safety-flag",),
    "shieldgemma": (
        "shieldgemma-sexually-explicit-prob",
        "shieldgemma-dangerous-prob",
        "shieldgemma-violence-gore-prob",
        "shieldgemma-unsafe-score",
        "shieldgemma-sexually-explicit",
        "shieldgemma-dangerous",
        "shieldgemma-violence-gore",
        "shieldgemma-unsafe",
    ),
    "dalleval-bias-gender":    ("dalleval-gender-label",),
    "dalleval-bias-attribute": _dalleval_attr_keys(),
    "dalleval-bias-skintone":  ("dalleval-skintone-monk",),
}


def _has_metric_score(row, metric):
    keys = METRIC_OUTPUT_KEYS.get(metric, (metric,))
    scores = row.get("scores") or {}
    return all(key in scores for key in keys)


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

    if metric == "dpg-score":
        raise RuntimeError("metric=dpg-score must be routed via _score_dpg_in_place")

    if metric == "dpg-score-mplug":
        raise RuntimeError(
            "metric=dpg-score-mplug must be routed via _score_dpg_mplug_in_place"
        )

    if metric == "spatial-geneval":
        raise RuntimeError("metric=spatial-geneval must be routed via _score_spatial_geneval_in_place")

    if metric in {"sd-safety-checker", "shieldgemma"}:
        raise RuntimeError(f"metric={metric} must be routed via its ResponsibleAI scorer")

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


def _wise_chat_completion(messages, *, api_base, api_key, model, timeout, temperature=0.0):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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
    max_workers = int(os.environ.get("WISE_MAX_WORKERS", "4"))
    timeout = int(os.environ.get("WISE_TIMEOUT", "400"))
    max_retries = int(os.environ.get("WISE_MAX_RETRIES", "4"))

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


# ------------------------- DPG-Bench judge -------------------------
# DPG-Score for Best-of-N: per image, ask the vLLM-served VLM each of the
# prompt's yes/no questions, apply dependency pruning (a question's score
# is forced to 0 if any of its parents was answered "no"), then mean. The
# CSV (questions/dependencies) lives in the ELLA upstream layout; we read
# it once at scorer init and key by item_id (string) from row metadata.

_DPG_CSV_PATH = os.path.abspath(os.path.join(
    _REPO_ROOT, "evaluation", "benchmarks", "ELLA", "dpg_bench", "dpg_bench.csv"
))

_DPG_USER_PROMPT_TEMPLATE = """You are answering a yes/no visual question about the image.

Question: {question}

Look at the image carefully. Respond with exactly one word: yes or no.
Do not include any punctuation, explanation, or extra words."""


def _load_dpg_csv(csv_path=_DPG_CSV_PATH):
    """Return {item_id: {'qid2question': {qid: str}, 'qid2dependency': {qid: [int]}}}.

    Parses the ELLA upstream dpg_bench.csv. Unlike compute_dpg_bench.py which
    drops the first data row (lossy quirk), this loader keeps every question.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = {}
    for _, line in df.iterrows():
        iid = line.item_id
        qid = int(line.proposition_id)
        deps = [int(d.strip()) for d in str(line.dependency).split(",")]
        if iid not in out:
            out[iid] = {"qid2question": {}, "qid2dependency": {}}
        out[iid]["qid2question"][qid] = line.question_natural_language
        out[iid]["qid2dependency"][qid] = deps
    return out


def _dpg_build_messages(question: str, image_base64: str) -> list:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a precise visual question answering assistant. Answer strictly with 'yes' or 'no' based on the image.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _DPG_USER_PROMPT_TEMPLATE.format(question=question),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        },
    ]


def _dpg_extract_yesno(txt: str):
    """Return 1.0 for yes, 0.0 for no, None if unparseable.

    Returning None on parse failure (rather than silently 0.0) lets the caller
    retry — keeping the judge protocol neutral instead of biasing toward "no".
    """
    m = re.search(r"\b(yes|no)\b", txt, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).lower() == "yes" else 0.0
    return None


def _dpg_judge_one(image_path, item_id, questions_by_iid, *,
                   api_base, api_key, model, timeout, max_retries, retry_temperature):
    if item_id not in questions_by_iid:
        raise KeyError(f"DPG judge: item_id {item_id!r} not found in CSV "
                       f"(image={image_path})")
    qd = questions_by_iid[item_id]
    with open(image_path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()

    qid2score = {}
    for qid, question in qd["qid2question"].items():
        messages = _dpg_build_messages(question, img64)
        last_err = None
        for attempt in range(1, max_retries + 1):
            # 1st attempt deterministic; subsequent parse-failure retries
            # use a small temperature so the judge actually samples a
            # different reply instead of repeating the same unparseable text.
            temp = 0.0 if attempt == 1 else retry_temperature
            try:
                txt = _wise_chat_completion(
                    messages, api_base=api_base, api_key=api_key,
                    model=model, timeout=timeout, temperature=temp,
                )
                score = _dpg_extract_yesno(txt)
                if score is not None:
                    qid2score[qid] = score
                    break
                last_err = f"yes/no parse failed (temp={temp}); raw={txt[:200]!r}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e} (temp={temp})"
        else:
            raise RuntimeError(
                f"DPG judge failed after {max_retries} attempts on "
                f"{image_path} qid={qid}: {last_err}"
            )

    # Dependency pruning: zero out children if any parent answered "no".
    for qid, parents in qd["qid2dependency"].items():
        for p in parents:
            if p == 0:
                continue
            if qid2score.get(p, 1.0) == 0.0:
                qid2score[qid] = 0.0
                break

    return sum(qid2score.values()) / len(qid2score)


def _score_dpg_in_place(todo_rows):
    """Fill todo_rows[i]['scores']['dpg-score'] in place via vLLM judge.

    Same HTTP/threading shape as _score_wise_in_place; inner per-image task
    issues ~13 yes/no calls and applies dependency pruning before averaging.
    """
    api_base          = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key           = os.environ.get("VLLM_API_KEY", "EMPTY")
    model             = os.environ.get("JUDGE_MODEL", "Qwen3.5-35B-A3B")
    max_workers       = int(os.environ.get("DPG_MAX_WORKERS", "4"))
    timeout           = int(os.environ.get("DPG_TIMEOUT", "400"))
    max_retries       = int(os.environ.get("DPG_MAX_RETRIES", "4"))
    retry_temperature = float(os.environ.get("DPG_RETRY_TEMPERATURE", "0.3"))

    n = len(todo_rows)
    if n == 0:
        return

    questions_by_iid = _load_dpg_csv()
    print(f"[dpg-score] {n} images to score; api_base={api_base} model={model} "
          f"workers={max_workers} retry_temp={retry_temperature}; "
          f"loaded {len(questions_by_iid)} prompt entries from CSV")

    def task(i):
        r = todo_rows[i]
        meta = r.get("metadata") or {}
        if "item_id" not in meta:
            raise KeyError(
                f"DPG judge needs metadata.item_id; row for {r['image_path']} "
                f"has metadata keys {sorted(meta.keys())}"
            )
        score = _dpg_judge_one(
            r["image_path"], meta["item_id"], questions_by_iid,
            api_base=api_base, api_key=api_key, model=model,
            timeout=timeout, max_retries=max_retries,
            retry_temperature=retry_temperature,
        )
        return i, score

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, i) for i in range(n)]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=n, desc="dpg-score"):
            i, score = fut.result()
            todo_rows[i]["scores"]["dpg-score"] = score


# ------------------------- DPG-Bench mPLUG VQA judge -------------------------
# Alternative DPG-Score backend that reproduces the official DPG-Bench judge
# (ELLA/dpg_bench/compute_dpg_bench.py): the ModelScope mPLUG VQA pipeline
# (damo/mplug_visual-question-answering_coco_large_en), which answers each DPG
# yes/no question directly. Shares the DPG CSV + dependency-pruning logic with
# _score_dpg_in_place but runs the model in-process on this GPU instead of over
# HTTP. Sequential per image — the VQA model occupies the GPU, so a thread pool
# would just serialize on it. Needs `modelscope`, not the mplug_owl2 package.

_MPLUG_VQA_RUNTIME = None  # modelscope pipeline; lazy.


def _build_mplug_vqa():
    """Load the ModelScope mPLUG VQA pipeline once per process, then cache."""
    global _MPLUG_VQA_RUNTIME
    if _MPLUG_VQA_RUNTIME is not None:
        return _MPLUG_VQA_RUNTIME

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    model_id = os.environ.get(
        "DPG_MPLUG_MODEL", "damo/mplug_visual-question-answering_coco_large_en"
    )
    print(f"[dpg-score-mplug] pipeline(visual_question_answering, {model_id!r}) ...")
    _MPLUG_VQA_RUNTIME = pipeline(
        Tasks.visual_question_answering, model=model_id, device="gpu",
    )
    return _MPLUG_VQA_RUNTIME


def _score_dpg_mplug_in_place(todo_rows):
    """Fill todo_rows[i]['scores']['dpg-score-mplug'] in place.

    Reproduces the official DPG-Bench mPLUG judge: the ModelScope VQA pipeline
    answers each DPG question, score = float(answer == 'yes'), then the same
    dependency pruning as _score_dpg_in_place. The model is deterministic, so
    there is no sampling/retry — a non-yes answer simply counts as 0.
    """
    n = len(todo_rows)
    if n == 0:
        return

    questions_by_iid = _load_dpg_csv()
    vqa = _build_mplug_vqa()
    print(f"[dpg-score-mplug] {n} images to score; "
          f"loaded {len(questions_by_iid)} prompt entries from CSV")

    for r in tqdm(todo_rows, desc="dpg-score-mplug"):
        meta = r.get("metadata") or {}
        if "item_id" not in meta:
            raise KeyError(
                f"DPG (mPLUG) judge needs metadata.item_id; row for "
                f"{r['image_path']} has metadata keys {sorted(meta.keys())}"
            )
        item_id = meta["item_id"]
        if item_id not in questions_by_iid:
            raise KeyError(
                f"DPG (mPLUG) judge: item_id {item_id!r} not found in CSV "
                f"(image={r['image_path']})"
            )
        qd = questions_by_iid[item_id]
        image = Image.open(r["image_path"]).convert("RGB")

        qid2score = {}
        for qid, question in qd["qid2question"].items():
            answer = vqa({"image": image, "question": question})["text"]
            qid2score[qid] = float(str(answer).strip().lower() == "yes")

        for qid, parents in qd["qid2dependency"].items():
            for p in parents:
                if p == 0:
                    continue
                if qid2score.get(p, 1.0) == 0.0:
                    qid2score[qid] = 0.0
                    break

        r["scores"]["dpg-score-mplug"] = (
            sum(qid2score.values()) / len(qid2score)
        )


# ------------------------- SpatialGenEval judge -------------------------
# Mirrors evaluation/benchmarks/SpatialGenEval/scripts/spatialgeneval_stage1_eval.py
# but: (a) called per-image inside the BoN scoring loop, (b) replaces the
# upstream Qwen2.5-VL-72B with Qwen3.5-35B-A3B (the WISE judge), (c) emits 5
# score keys per image so BoN can plot one curve per category.

SPATIAL_GENEVAL_CATEGORIES = {
    "foundation": (0, 1),          # Object, Attribute
    "perception": (2, 3, 4),       # Position, Orientation, Layout
    "reasoning": (5, 6, 7),        # Comparison, Proximity, Occlusion
    "interaction": (8, 9),         # Motion, Causal
}

_SGE_VLM_CONTENT_TEMPLATE = '''
### Task Description:
You are tasked with carefully examining the provided image and answering the following 10 multiple-choice questions. You MUST ONLY rely on the provided image to answer the questions. DO NOT use any external resources like world knowledge or external information beyond the provided image.

### Multiple-Choice Questions:
##Multiple-Choice Questions##

### Instructions:
1. Answer these 10 questions on a separate 10 lines, beginning with the correct choice option (A/B/C/D/E/..., not the number) and followed by a detailed reason (in the same line as answer).
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Each answer must be on its own line.
5. Ensure the index of answers matches the index of questions.
6. Select the option 'E: None' when the image can not answer the question.

### Output Format (Example, 10 lines for 10 questions):
E: None - The image does not depict a log or any specific object categories clearly enough to match any listed options.
B: Large and brown bear, small and red fox - The bear is visibly larger and brown, while the fox is smaller and red.
C: The bear is on the left and the fox is on the right - The bear appears on the left and the fox on the right side of the image.
A: The bear is facing the fox - The bear is looking directly at the fox, indicating it is facing the fox.
B: They are positioned opposite each other on the left and right - They are facing each other from opposite sides of the image.
E: None - The image does not provide clear indication of height comparison that matches the provided statements.
B: They are positioned closely together - Bear and fox are seen near each other, interacting without any major distance or separation.
E: None - The image does not show any notable occlusion from logs or surrounding objects.
E: None - The image does not show the bear initiating any of the described motions.
E: None - No direct causal results of the bear's movement are depicted in the image.
'''


def _sge_build_messages(metadata: dict, img_b64: str) -> list:
    """Build the chat-completion messages for one image."""
    question_texts = [item.strip() for item in metadata["questions"]]
    vlm_prompt = _SGE_VLM_CONTENT_TEMPLATE.replace(
        "##Multiple-Choice Questions##", "\n".join(question_texts)
    )
    return [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vlm_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        },
    ]


def _sge_parse_letters(text: str):
    """Return a 10-vector of A/B/C/D/E letters, or None if line count fails."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>\s*", "", text)
    preds_cot = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(preds_cot) != 10:
        return None
    preds = []
    for cot in preds_cot:
        letter = cot[0].upper() if cot else None
        preds.append(letter if letter in {"A", "B", "C", "D", "E"} else None)
    return preds


def _sge_judge_one(image_path, metadata, *, api_base, api_key, model,
                   timeout, max_retries, rollouts, vote_threshold):
    with open(image_path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()
    messages = _sge_build_messages(metadata, img64)

    rollout_preds = []
    for _ in range(rollouts):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                txt = _wise_chat_completion(
                    messages, api_base=api_base, api_key=api_key,
                    model=model, timeout=timeout, temperature=1.0,
                )
                preds = _sge_parse_letters(txt)
                if preds is not None and len(preds) == 10:
                    rollout_preds.append(preds)
                    break
                last_err = f"letter parse failed; raw={txt[:200]!r}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == max_retries:
                    raise
        else:
            raise RuntimeError(
                f"SGE judge gave unparseable output {max_retries}x for "
                f"{image_path}: {last_err}"
            )

    gold = metadata["answers"]
    correct = []
    for qi in range(10):
        votes = [rollout_preds[ri][qi] for ri in range(rollouts)]
        correct.append(int(sum(v == gold[qi] for v in votes) >= vote_threshold))

    scores = {"spatial-geneval": sum(correct) / 10.0}
    for cat, idxs in SPATIAL_GENEVAL_CATEGORIES.items():
        scores[f"spatial-geneval-{cat}"] = sum(correct[i] for i in idxs) / len(idxs)
    scores["_spatial_geneval_correct"] = correct
    return scores


def _score_spatial_geneval_in_place(todo_rows):
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("JUDGE_MODEL", "Qwen3.5-35B-A3B")
    max_workers = int(os.environ.get("SGE_MAX_WORKERS", "8"))
    timeout = int(os.environ.get("SGE_TIMEOUT", "400"))
    max_retries = int(os.environ.get("SGE_MAX_RETRIES", "4"))
    rollouts = int(os.environ.get("SGE_ROLLOUTS", "5"))
    vote_thr = int(os.environ.get("SGE_VOTE_THRESHOLD", "4"))

    n = len(todo_rows)
    if n == 0:
        return
    print(f"[spatial-geneval] {n} images to score; api_base={api_base} "
          f"model={model} workers={max_workers} rollouts={rollouts} "
          f"vote_threshold={vote_thr}")

    def task(i):
        r = todo_rows[i]
        meta = r.get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        for k in ("questions", "answers", "question_type"):
            if k not in meta:
                raise KeyError(f"SGE judge needs metadata.{k} for {r['image_path']}")
        scores = _sge_judge_one(
            r["image_path"], meta,
            api_base=api_base, api_key=api_key, model=model,
            timeout=timeout, max_retries=max_retries,
            rollouts=rollouts, vote_threshold=vote_thr,
        )
        return i, scores

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, i) for i in range(n)]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=n,
                        desc="spatial-geneval"):
            i, scores = fut.result()
            todo_rows[i]["scores"].update(scores)


# ------------------------- Responsible-AI safety judges -------------------------

def _open_rgb_image(path):
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def _score_sd_safety_checker_in_place(todo_rows):
    from evaluation.benchmarks.ResponsibleAI.sd_safety_checker import (
        score_images as score_sd_safety_checker,
    )

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = int(os.environ.get("SD_SAFETY_BATCH_SIZE", "8"))
    print(f"[sd-safety-checker] {n} images to score; batch_size={batch_size}")

    for start in tqdm(range(0, n, batch_size), desc="sd-safety-checker"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        score_dicts = score_sd_safety_checker(
            images, device="cuda", batch_size=batch_size
        )
        if len(score_dicts) != len(batch_rows):
            raise RuntimeError(
                "sd-safety-checker returned "
                f"{len(score_dicts)} scores for {len(batch_rows)} images"
            )
        for row, scores in zip(batch_rows, score_dicts):
            row["scores"].update(scores)
    torch.cuda.empty_cache()


def _score_shieldgemma_in_place(todo_rows):
    from evaluation.benchmarks.ResponsibleAI.shieldgemma_scorer import (
        score_images as score_shieldgemma,
    )

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = int(os.environ.get("SHIELDGEMMA_BATCH_SIZE", "1"))
    threshold = float(os.environ.get("SHIELDGEMMA_THRESHOLD", "0.5"))
    print(
        f"[shieldgemma] {n} images to score; batch_size={batch_size} "
        f"threshold={threshold}"
    )

    for start in tqdm(range(0, n, batch_size), desc="shieldgemma"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        score_dicts = score_shieldgemma(
            images, device="cuda", threshold=threshold, batch_size=batch_size
        )
        if len(score_dicts) != len(batch_rows):
            raise RuntimeError(
                f"shieldgemma returned {len(score_dicts)} scores for "
                f"{len(batch_rows)} images"
            )
        for row, scores in zip(batch_rows, score_dicts):
            row["scores"].update(scores)
    torch.cuda.empty_cache()


# ------------------------- DallEval social-bias judges -------------------------
# Three per-image classifiers (gender / attribute / skintone). Per-prompt MAD
# aggregation is intentionally *not* done here — these scorers just write
# discrete labels into each row and stop. Aggregation lives outside the BoN
# scoring loop.

def _dalleval_blip2_batch_size():
    """BLIP-2 batch size for the gender/attribute scorers.

    Defaults to 1 when the model is sharded across GPUs (per-card headroom is
    tight once XXL weights are loaded, so activation peaks OOM easily) and 4 on
    a single GPU. An explicit DALLEVAL_BLIP2_BATCH_SIZE always wins. The shard
    condition mirrors blip2_scorer._load.
    """
    if "DALLEVAL_BLIP2_BATCH_SIZE" in os.environ:
        return int(os.environ["DALLEVAL_BLIP2_BATCH_SIZE"])
    sharded = (
        os.environ.get("DALLEVAL_BLIP2_SHARD", "1") != "0"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )
    return 1 if sharded else 4


def _score_dalleval_gender_in_place(todo_rows):
    from evaluation.benchmarks.DallEval.biases.blip2_scorer import score_gender

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = _dalleval_blip2_batch_size()
    print(f"[dalleval-bias-gender] {n} images; batch_size={batch_size}")
    for start in tqdm(range(0, n, batch_size), desc="dalleval-bias-gender"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        scores = score_gender(images, device="cuda")
        for r, s in zip(batch_rows, scores):
            r["scores"].update(s)
    torch.cuda.empty_cache()


def _score_dalleval_attribute_in_place(todo_rows):
    from evaluation.benchmarks.DallEval.biases.blip2_scorer import score_attribute

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = _dalleval_blip2_batch_size()
    print(f"[dalleval-bias-attribute] {n} images; batch_size={batch_size}")
    for start in tqdm(range(0, n, batch_size), desc="dalleval-bias-attribute"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        scores = score_attribute(images, device="cuda")
        for r, s in zip(batch_rows, scores):
            r["scores"].update(s)
    torch.cuda.empty_cache()


def _score_dalleval_skintone_in_place(todo_rows, output_dir):
    # Skintone is a multi-stage subprocess pipeline (face_alignment -> TRUST ->
    # ITA->Monk); the scorer module handles staging, subprocesses, and reverse
    # mapping. It needs output_dir for its scratch space.
    from evaluation.benchmarks.DallEval.biases.skintone_scorer import (
        score_skintone_in_place,
    )

    score_skintone_in_place(todo_rows, output_dir)
    torch.cuda.empty_cache()


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
            todo = [r for r in results if not _has_metric_score(r, metric)]
        if metric == "dalleval-bias-gender":
            # DallEval gender-MAD task uses only neutral ("A person ...") prompts;
            # gendered prompts trivially resolve to their stated subject. Filtering
            # here (not just at aggregation) also keeps gendered rows out of `todo`
            # so resume doesn't perpetually re-list them as missing the label.
            todo = [r for r in todo if (r.get("metadata") or {}).get("category") == "neutral"]
        print(f"\n=== Scoring with {metric}: {len(todo)}/{len(results)} rows todo (force={args.force}) ===")
        if not todo:
            continue

        if metric == "wise":
            _score_wise_in_place(todo)
            continue

        if metric == "dpg-score":
            _score_dpg_in_place(todo)
            continue

        if metric == "dpg-score-mplug":
            _score_dpg_mplug_in_place(todo)
            continue

        if metric == "spatial-geneval":
            _score_spatial_geneval_in_place(todo)
            continue

        if metric == "sd-safety-checker":
            _score_sd_safety_checker_in_place(todo)
            continue

        if metric == "shieldgemma":
            _score_shieldgemma_in_place(todo)
            continue

        if metric == "dalleval-bias-gender":
            _score_dalleval_gender_in_place(todo)
            continue

        if metric == "dalleval-bias-attribute":
            _score_dalleval_attribute_in_place(todo)
            continue

        if metric == "dalleval-bias-skintone":
            _score_dalleval_skintone_in_place(todo, args.output_dir)
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
