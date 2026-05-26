# DiffDoctor aigi-detector Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `diffdoctor` scoring metric that runs the DiffDoctor SegFormer artifact detector on the `aigi-detector` Best-of-N dataset, emitting per-image `diffdoctor-clean-rate` and `diffdoctor-clean-area` scores that flat-average into `average_scores.json`.

**Architecture:** A standalone scorer module under `evaluation/benchmarks/DiffDoctor/` loads the SegFormer detector and turns each image into two `[0,1]` higher-is-better scores; `score-images.py` gains a thin in-place dispatch branch (mirroring `sd-safety-checker`/`shieldgemma`); `run-bestofn.sh` wires the metric to the aigi-detector dataset in the `visualquality` conda env.

**Tech Stack:** PyTorch, HuggingFace `transformers` (`SegformerForSemanticSegmentation`, `SegformerImageProcessor`), `torchvision`, OpenCV, NumPy.

**Spec:** `docs/superpowers/specs/2026-05-26-diffdoctor-aigi-detector-metric-design.md`

---

## File Structure

- **Create** `evaluation/benchmarks/DiffDoctor/__init__.py` — package marker.
- **Create** `evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py` — model load, preprocessing (replicating `ad_inference.py`), heatmap→score aggregation, public `score_images`.
- **Create** `evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py` — CPU unit tests for the pure aggregation function (`heatmaps_to_scores`), runnable via `python`.
- **Modify** `evaluation/metrics/score-images.py` — register `diffdoctor` in `AVAILABLE_METRICS`, `METRIC_OUTPUT_KEYS`, add `_score_diffdoctor_in_place`, dispatch in `main()`.
- **Modify** `evaluation/run-bestofn.sh` — aigi-detector `metric_list=(diffdoctor)`, `metric_env[diffdoctor]=visualquality`, update Stage 3 comment.
- **No change** `evaluation/run-bestofn-batch.sh` — only loops datasets; picks up the new scorer automatically.

---

## Task 1: Pure aggregation function + CPU unit tests (TDD)

**Files:**
- Create: `evaluation/benchmarks/DiffDoctor/__init__.py`
- Create: `evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py`
- Create: `evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py` (partial — only `heatmaps_to_scores` in this task)

- [ ] **Step 1: Create the package marker**

Create `evaluation/benchmarks/DiffDoctor/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the failing test**

Create `evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py`:

```python
"""CPU unit tests for the DiffDoctor heatmap->score aggregation.

Runnable directly (no pytest required):
    python evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.benchmarks.DiffDoctor.diffdoctor_scorer import heatmaps_to_scores


def test_all_clean():
    hm = torch.zeros(1, 1, 8, 8)
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 1.0
    assert abs(s["diffdoctor-clean-area"] - 1.0) < 1e-6


def test_one_artifact_pixel():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 0, 0] = 0.9
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 0.0  # max 0.9 >= 0.5
    assert abs(s["diffdoctor-clean-area"] - (1.0 - 1.0 / 64)) < 1e-6


def test_half_artifact():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, :4, :] = 0.9  # half the pixels above threshold
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 0.0
    assert abs(s["diffdoctor-clean-area"] - 0.5) < 1e-6


def test_batch_and_3d_input():
    hm = torch.zeros(2, 8, 8)  # 3D [N,H,W] should be accepted
    hm[1] = 0.9
    out = heatmaps_to_scores(hm, threshold=0.5)
    assert len(out) == 2
    assert out[0]["diffdoctor-clean-rate"] == 1.0
    assert out[1]["diffdoctor-clean-rate"] == 0.0


if __name__ == "__main__":
    test_all_clean()
    test_one_artifact_pixel()
    test_half_artifact()
    test_batch_and_3d_input()
    print("OK: all heatmaps_to_scores tests passed")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'heatmaps_to_scores'` (the scorer module/function does not exist yet).

- [ ] **Step 4: Write the minimal implementation**

Create `evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py` with just the pure function for now:

```python
"""DiffDoctor artifact-detector scorer for the aigi-detector eval dataset.

Wraps the DiffDoctor SegFormer artifact detector (ICCV 2025). Per image it
emits two higher-is-better scores in [0, 1]:

  diffdoctor-clean-rate : 1.0 if the image has no artifact pixel above the
                          threshold (heatmap.max() < tau), else 0.0
  diffdoctor-clean-area : 1 - fraction of pixels above the threshold

Preprocessing and inference replicate flow_grpo/DiffDoctor/ad_inference.py.
"""
from __future__ import annotations

import torch


def heatmaps_to_scores(heatmaps, threshold: float = 0.5) -> list[dict]:
    """Aggregate per-pixel artifact heatmaps into per-image scores.

    heatmaps: tensor of shape [N, 1, H, W] or [N, H, W], values in [0, 1].
    Returns one dict per image with the two diffdoctor-* keys.
    """
    if heatmaps.dim() == 3:
        heatmaps = heatmaps.unsqueeze(1)
    n = heatmaps.shape[0]
    flat = heatmaps.reshape(n, -1)
    max_vals = flat.max(dim=1).values
    area_ratio = (flat > threshold).float().mean(dim=1)
    return [
        {
            "diffdoctor-clean-rate": float(max_vals[i].item() < threshold),
            "diffdoctor-clean-area": float(1.0 - area_ratio[i].item()),
        }
        for i in range(n)
    ]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py`
Expected: `OK: all heatmaps_to_scores tests passed`

- [ ] **Step 6: Commit**

```bash
git add evaluation/benchmarks/DiffDoctor/__init__.py \
        evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py \
        evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py
git commit -m "Add DiffDoctor heatmap->score aggregation with CPU tests"
```

---

## Task 2: Model loading, preprocessing, and `score_images`

**Files:**
- Modify: `evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py`

- [ ] **Step 1: Add imports and module-level config/cache**

At the top of `diffdoctor_scorer.py`, replace the import block (`from __future__ import annotations` / `import torch`) with:

```python
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# DiffDoctor artifact-detector checkpoint (already present on the server).
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/"
    "ad_pytorch_model.bin"
)
BACKBONE = "nvidia/mit-b5"

_model = None
_preprocessor = None
_loaded_device = None
```

- [ ] **Step 2: Add the loader and helpers**

Append to `diffdoctor_scorer.py` (after `heatmaps_to_scores`):

```python
def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("DIFFDOCTOR_CKPT", DEFAULT_CKPT)


def _threshold() -> float:
    return float(os.environ.get("DIFFDOCTOR_THRESHOLD", "0.5"))


def _load(device: str):
    """Load the SegFormer artifact detector once per process, then cache."""
    global _model, _preprocessor, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, _preprocessor, device

    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"DiffDoctor checkpoint not found at {ckpt!r}. Set DIFFDOCTOR_CKPT "
            f"to the ad_pytorch_model.bin path (the ~339 MB weights), or git "
            f"lfs pull the in-repo copy under flow_grpo/DiffDoctor/checkpoints/."
        )

    preprocessor = SegformerImageProcessor.from_pretrained(BACKBONE)
    model = SegformerForSemanticSegmentation.from_pretrained(BACKBONE)
    # DiffDoctor replaces the segmentation head with a single-channel conv.
    model.decode_head.classifier = nn.Conv2d(
        model.decode_head.classifier.in_channels, 1, kernel_size=1
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()

    _model, _preprocessor, _loaded_device = model, preprocessor, device
    return model, preprocessor, device


def _preprocess(pil_images, preprocessor, device):
    """Replicate ad_inference.py: resize 512, RGB, ToTensor, ImageNet-normalize."""
    from torchvision import transforms

    tensors = []
    for img in pil_images:
        arr = np.array(img.convert("RGB"))          # HWC RGB uint8
        arr = cv2.resize(arr, (512, 512))           # same as ad_inference.py
        tensors.append(transforms.ToTensor()(arr))  # CHW float in [0, 1]
    # do_rescale=False: ToTensor already scaled to [0, 1]; processor only
    # resizes (already 512) and applies ImageNet mean/std normalization.
    processed = preprocessor(tensors, return_tensors="pt", do_rescale=False)
    return processed["pixel_values"].to(device)


def _infer_heatmaps(processed, model):
    with torch.no_grad():
        pred = model(processed)
        logits = nn.functional.interpolate(
            pred.logits, size=processed.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        heatmaps = torch.sigmoid(logits)  # [N, 1, 512, 512] in [0, 1]
    return heatmaps


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {diffdoctor-clean-rate, diffdoctor-clean-area} dict per PIL image."""
    model, preprocessor, device = _load(device)
    threshold = _threshold()
    results: list[dict] = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        processed = _preprocess(batch, preprocessor, device)
        heatmaps = _infer_heatmaps(processed, model)
        assert heatmaps.min() >= 0 and heatmaps.max() <= 1
        results.extend(heatmaps_to_scores(heatmaps.cpu(), threshold))
    return results
```

- [ ] **Step 3: Verify the module imports cleanly and unit tests still pass**

The aggregation tests must keep passing, and importing the module must not require a GPU or the checkpoint (loading is lazy, inside `_load`).

Run:
```bash
python evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py
python -c "import sys; sys.path.insert(0,'.'); from evaluation.benchmarks.DiffDoctor.diffdoctor_scorer import score_images; print('import OK')"
```
Expected:
```
OK: all heatmaps_to_scores tests passed
import OK
```
(If `cv2` import fails, the `visualquality` env is missing `opencv-python` — install it there; it is in `flow_grpo/DiffDoctor/requirements.txt`.)

- [ ] **Step 4: Commit**

```bash
git add evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py
git commit -m "Add DiffDoctor SegFormer loading, preprocessing, and score_images"
```

---

## Task 3: Wire `diffdoctor` into score-images.py

**Files:**
- Modify: `evaluation/metrics/score-images.py`

- [ ] **Step 1: Register the metric name**

In `AVAILABLE_METRICS` (around line 26-31), add `"diffdoctor"`. After the edit the list ends like:

```python
AVAILABLE_METRICS = [
    "pickscore", "imagereward", "aesthetic", "hpsv3", "deqa", "visualquality_r1",
    "ocr", "geneval", "wise", "dpg-score", "dpg-score-mplug", "spatial-geneval",
    "sd-safety-checker", "shieldgemma",
    "dalleval-bias-gender", "dalleval-bias-attribute", "dalleval-bias-skintone",
    "diffdoctor",
]
```

- [ ] **Step 2: Register the output keys**

In the `METRIC_OUTPUT_KEYS` dict (around line 48-63), add a `diffdoctor` entry so resume / `_has_metric_score` checks both keys. Add this entry before the closing brace:

```python
    "diffdoctor": ("diffdoctor-clean-rate", "diffdoctor-clean-area"),
```

- [ ] **Step 3: Add the in-place scorer function**

Immediately after the `_score_shieldgemma_in_place` function (it ends with `torch.cuda.empty_cache()` around line 798), insert:

```python
def _score_diffdoctor_in_place(todo_rows):
    from evaluation.benchmarks.DiffDoctor.diffdoctor_scorer import (
        score_images as score_diffdoctor,
    )

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = int(os.environ.get("DIFFDOCTOR_BATCH_SIZE", "8"))
    print(f"[diffdoctor] {n} images to score; batch_size={batch_size}")

    for start in tqdm(range(0, n, batch_size), desc="diffdoctor"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        score_dicts = score_diffdoctor(images, device="cuda", batch_size=batch_size)
        if len(score_dicts) != len(batch_rows):
            raise RuntimeError(
                f"diffdoctor returned {len(score_dicts)} scores for "
                f"{len(batch_rows)} images"
            )
        for row, scores in zip(batch_rows, score_dicts):
            row["scores"].update(scores)
    torch.cuda.empty_cache()
```

- [ ] **Step 4: Add the dispatch branch in `main()`**

In `main()`, right after the `shieldgemma` dispatch block:

```python
        if metric == "shieldgemma":
            _score_shieldgemma_in_place(todo)
            continue
```

insert:

```python
        if metric == "diffdoctor":
            _score_diffdoctor_in_place(todo)
            continue
```

- [ ] **Step 5: Verify the CLI accepts the metric and keys are registered**

Run:
```bash
python -c "import sys; sys.path.insert(0,'.'); import importlib.util as u; spec=u.spec_from_file_location('si','evaluation/metrics/score-images.py'); m=u.module_from_spec(spec); spec.loader.exec_module(m); assert 'diffdoctor' in m.AVAILABLE_METRICS; assert m.METRIC_OUTPUT_KEYS['diffdoctor']==('diffdoctor-clean-rate','diffdoctor-clean-area'); print('score-images wiring OK')"
```
Expected: `score-images wiring OK`

- [ ] **Step 6: Commit**

```bash
git add evaluation/metrics/score-images.py
git commit -m "Wire diffdoctor metric into score-images.py"
```

---

## Task 4: Wire `diffdoctor` into run-bestofn.sh

**Files:**
- Modify: `evaluation/run-bestofn.sh`

- [ ] **Step 1: Set the aigi-detector metric list**

Replace the aigi-detector case block (currently lines ~81-84):

```bash
    # aigi-detector: 1000 image-level MSCOCO val2014 prompts, generation only for
    # now. Score model is TBD, so metric_list is empty -> Stage 2/3 are skipped.

    aigi-detector)    metric_list=() ;;
```

with:

```bash
    # aigi-detector: 1000 image-level MSCOCO val2014 prompts. Scored by the
    # DiffDoctor artifact detector; results flat-averaged into average_scores.json.
    aigi-detector)    metric_list=(diffdoctor) ;;
```

- [ ] **Step 2: Map the metric to the visualquality env**

In the `metric_env` associative array (currently ending at `[dpg-score-mplug]=dpg-bench`), add a line before the closing `)`:

```bash
    [diffdoctor]=visualquality
```

- [ ] **Step 3: Update the Stage 3 aigi-detector comment**

Replace the aigi-detector Stage 3 branch (currently lines ~177-181):

```bash
elif [[ "${dataset}" == "aigi-detector" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped (aigi-detector: generation only, scorer TBD)"
    echo "  Images: ${output_dir}/images/"
    echo "============================================"
```

with:

```bash
elif [[ "${dataset}" == "aigi-detector" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped (aigi-detector: flat-average metric)"
    echo "  See ${output_dir}/average_scores.json"
    echo "============================================"
```

- [ ] **Step 4: Verify the script still parses and the wiring is present**

Run:
```bash
bash -n evaluation/run-bestofn.sh && echo "syntax OK"
grep -n "aigi-detector)    metric_list=(diffdoctor)" evaluation/run-bestofn.sh
grep -n "\[diffdoctor\]=visualquality" evaluation/run-bestofn.sh
```
Expected: `syntax OK` plus one matching line from each grep.

- [ ] **Step 5: Commit**

```bash
git add evaluation/run-bestofn.sh
git commit -m "Wire diffdoctor as the aigi-detector scorer in run-bestofn.sh"
```

---

## Task 5: Server smoke test (manual, requires GPU + checkpoint)

This task runs only on the server where the GPU and the DiffDoctor checkpoint are available. It is verification — it produces a commit only if it surfaces a bug to fix.

**Files:** none (runtime verification)

- [ ] **Step 1: Confirm the checkpoint is reachable**

Run:
```bash
ls -la /data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/ad_pytorch_model.bin
```
Expected: a ~339 MB file (not a 134-byte pointer). If absent, set `DIFFDOCTOR_CKPT` to the correct path before continuing.

- [ ] **Step 2: Score the DiffDoctor demo images and check ranges**

In the `visualquality` env, run a self-contained check over the bundled demo images:

```bash
conda activate visualquality
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from PIL import Image
from evaluation.benchmarks.DiffDoctor.diffdoctor_scorer import score_images

d = "flow_grpo/DiffDoctor/asset/input"
paths = [os.path.join(d, p) for p in os.listdir(d) if p.lower().endswith((".jpg", ".png"))]
imgs = [Image.open(p).convert("RGB") for p in paths]
scores = score_images(imgs, device="cuda", batch_size=4)
assert len(scores) == len(imgs)
for p, s in zip(paths, scores):
    cr, ca = s["diffdoctor-clean-rate"], s["diffdoctor-clean-area"]
    assert cr in (0.0, 1.0) and 0.0 <= ca <= 1.0, (p, s)
    print(f"{os.path.basename(p):40s} clean-rate={cr} clean-area={ca:.4f}")
print("OK: diffdoctor smoke test passed")
PY
```
Expected: one line per image with `clean-rate` in {0.0, 1.0} and `clean-area` in [0, 1], ending with `OK: diffdoctor smoke test passed`.

- [ ] **Step 3: Cross-check against `ad_inference.py` (parity)**

Confirm the scorer's `clean-area` is consistent with the official heatmap. For one image, the fraction of `sigmoid` pixels `> 0.5` produced by `ad_inference.py` (run it per its readme; it writes heatmaps to `flow_grpo/DiffDoctor/asset/output/`) should match `1 - clean-area` for that same image within small resampling tolerance. Spot-check one image visually: a clearly artifact-heavy image should have lower `clean-area` than a clean one.

- [ ] **Step 4: End-to-end dry check on a few aigi-detector rows (optional)**

If an `aigi-detector` generation output dir already exists, score a copy of its `evaluation_results.jsonl` to confirm the full path works and `average_scores.json` gains both keys:

```bash
conda activate visualquality
CUDA_VISIBLE_DEVICES=0 python evaluation/metrics/score-images.py \
    --output_dir <existing_aigi_output_dir> --metrics diffdoctor
python -c "import json; a=json.load(open('<existing_aigi_output_dir>/average_scores.json')); print({k:a[k] for k in a if k.startswith('diffdoctor')})"
```
Expected: a dict containing `diffdoctor-clean-rate` and `diffdoctor-clean-area`, both in [0, 1].

---

## Self-Review

- **Spec coverage:**
  - Scorer module (spec a) → Tasks 1-2.
  - `score-images.py` edits: AVAILABLE_METRICS, METRIC_OUTPUT_KEYS, in-place branch, dispatch (spec b) → Task 3.
  - `run-bestofn.sh` edits: metric_list, metric_env, Stage 3 comment (spec c) → Task 4.
  - `run-bestofn-batch.sh` (spec d) → confirmed no-op (File Structure note).
  - Data flow / flat-average口径 → Task 4 (Stage 3 stays skipped) + average_scores.json verified in Task 5 Step 4.
  - Error handling: missing checkpoint FileNotFoundError → Task 2 Step 2; heatmap range assert → Task 2 Step 2; length mismatch RuntimeError → Task 3 Step 3.
  - Testing: CPU unit test → Task 1; server smoke + parity → Task 5.
  - Preprocessing replicates ad_inference.py → Task 2 Step 2.
  - Env overrides DIFFDOCTOR_CKPT / DIFFDOCTOR_THRESHOLD / DIFFDOCTOR_BATCH_SIZE → Tasks 2-3.
- **Placeholder scan:** no TBD/TODO; every code step shows full code.
- **Type consistency:** `heatmaps_to_scores(heatmaps, threshold)` and `score_images(pil_images, device, batch_size)` signatures and the two score keys (`diffdoctor-clean-rate`, `diffdoctor-clean-area`) are identical across Tasks 1-5.
