# Effort aigi-detector Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `effort` scoring metric that runs the official Effort AIGI detector (CLIP ViT-L/14 + SVD-residual adapter + 2-class head, GenImage(sdv1.4) checkpoint) on the `aigi-detector` Best-of-N dataset, emitting a per-image `effort-real-score = 1 - fake_prob` that flat-averages into `average_scores.json`, alongside the existing DiffDoctor metric.

**Architecture:** A standalone, self-contained scorer module under `evaluation/benchmarks/Effort/` copies the official Effort model definition (so the published checkpoint's parameter names match), preprocesses with CLIP normalization, and turns each image into one `[0,1]` higher-is-better score; `score-images.py` gains a thin in-place dispatch branch (mirroring `diffdoctor`); `run-bestofn.sh` appends the metric to the aigi-detector dataset in the `visualquality` conda env. The heavy DeepfakeBench framework is NOT imported (it pins torch 1.12).

**Tech Stack:** PyTorch (2.x), HuggingFace `transformers` (`CLIPModel`), `torchvision`.

**Spec:** `docs/superpowers/specs/2026-05-26-effort-aigi-detector-metric-design.md`

---

## File Structure

- **Create** `evaluation/benchmarks/Effort/__init__.py` — package marker.
- **Create** `evaluation/benchmarks/Effort/effort_scorer.py` — SVD-residual model classes (copied from the official detector), `EffortDetector`, checkpoint loading (official recipe), CLIP-normalized preprocessing, fake-prob→score aggregation, public `score_images`.
- **Create** `evaluation/benchmarks/Effort/test_effort_scorer.py` — CPU unit tests for the pure aggregation function (`probs_to_scores`), runnable via `python`.
- **Modify** `evaluation/metrics/score-images.py` — register `effort` in `AVAILABLE_METRICS`, `METRIC_OUTPUT_KEYS`, add `_score_effort_in_place`, dispatch in `main()`.
- **Modify** `evaluation/run-bestofn.sh` — aigi-detector `metric_list=(diffdoctor effort)`, `metric_env[effort]=visualquality`, update Stage 3 comment.
- **No change** `evaluation/run-bestofn-batch.sh` — only loops datasets; picks up the new scorer automatically.

---

## Task 1: Pure aggregation function + CPU unit tests (TDD)

**Files:**
- Create: `evaluation/benchmarks/Effort/__init__.py`
- Create: `evaluation/benchmarks/Effort/test_effort_scorer.py`
- Create: `evaluation/benchmarks/Effort/effort_scorer.py` (partial — only `probs_to_scores` in this task)

- [ ] **Step 1: Create the package marker**

Create `evaluation/benchmarks/Effort/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the failing test**

Create `evaluation/benchmarks/Effort/test_effort_scorer.py`:

```python
"""CPU unit tests for the Effort fake-prob -> score aggregation.

Runnable directly (no pytest required):
    python evaluation/benchmarks/Effort/test_effort_scorer.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.benchmarks.Effort.effort_scorer import probs_to_scores


def test_real_image():
    out = probs_to_scores(torch.tensor([0.0]))[0]
    assert abs(out["effort-real-score"] - 1.0) < 1e-6


def test_fake_image():
    out = probs_to_scores(torch.tensor([1.0]))[0]
    assert abs(out["effort-real-score"] - 0.0) < 1e-6


def test_midrange():
    out = probs_to_scores(torch.tensor([0.3]))[0]
    assert abs(out["effort-real-score"] - 0.7) < 1e-6


def test_batch():
    out = probs_to_scores(torch.tensor([0.0, 0.25, 1.0]))
    assert len(out) == 3
    assert abs(out[0]["effort-real-score"] - 1.0) < 1e-6
    assert abs(out[1]["effort-real-score"] - 0.75) < 1e-6
    assert abs(out[2]["effort-real-score"] - 0.0) < 1e-6


if __name__ == "__main__":
    test_real_image()
    test_fake_image()
    test_midrange()
    test_batch()
    print("OK: all probs_to_scores tests passed")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python evaluation/benchmarks/Effort/test_effort_scorer.py`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'probs_to_scores'` (the scorer module/function does not exist yet).

- [ ] **Step 4: Write the minimal implementation**

Create `evaluation/benchmarks/Effort/effort_scorer.py` with just the docstring and the pure function for now:

```python
"""Effort AIGI-detector scorer for the aigi-detector eval dataset.

Wraps the official Effort detector (ICML 2025): a CLIP ViT-L/14 vision tower
with an orthogonal SVD-residual adapter on every self_attn linear layer plus a
2-class head. Per image it emits a single higher-is-better score in [0, 1]:

  effort-real-score : 1 - fake_prob, where
                      fake_prob = softmax(head(pooled_feature))[:, 1].

Preprocessing and the model definition replicate the official demo
(flow_grpo/Effort-AIGI-Detection/DeepfakeBench/training/demo.py and
detectors/effort_detector.py), including CLIP mean/std normalization, so the
official GenImage(sdv1.4) checkpoint loads and behaves identically.
"""
from __future__ import annotations

import torch


def probs_to_scores(fake_probs) -> list[dict]:
    """Aggregate per-image fake probabilities into per-image scores.

    fake_probs: 1-D tensor of values in [0, 1].
    Returns one dict per image with the effort-real-score key.
    """
    return [{"effort-real-score": float(1.0 - p.item())} for p in fake_probs]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python evaluation/benchmarks/Effort/test_effort_scorer.py`
Expected: `OK: all probs_to_scores tests passed`

- [ ] **Step 6: Commit**

```bash
git add evaluation/benchmarks/Effort/__init__.py \
        evaluation/benchmarks/Effort/effort_scorer.py \
        evaluation/benchmarks/Effort/test_effort_scorer.py
git commit -m "Add Effort fake-prob->score aggregation with CPU tests"
```

---

## Task 2: Model definition, checkpoint loading, preprocessing, and `score_images`

**Files:**
- Modify: `evaluation/benchmarks/Effort/effort_scorer.py`

- [ ] **Step 1: Replace the import block with imports + module-level config/cache**

In `effort_scorer.py`, replace the import block (`from __future__ import annotations` / `import torch`) with:

```python
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Official Effort GenImage(sdv1.4) checkpoint (already present on the server).
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/"
    "effort_clip_L14_trainOn_sdv14.pth"
)
# CLIP ViT-L/14 backbone (architecture + pretrained encoder). Fetched via the
# HF_ENDPOINT mirror already exported by the run scripts; override with EFFORT_CLIP.
DEFAULT_BACKBONE = "openai/clip-vit-large-patch14"

# CLIP image normalization (must match training; see effort.yaml mean/std).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_model = None
_loaded_device = None
```

- [ ] **Step 2: Append the model classes (copied verbatim from the official detector)**

Append to `effort_scorer.py` (after `probs_to_scores`). This is the official Effort model definition; copy it exactly so the published checkpoint's parameter names match under `load_state_dict`:

```python
class SVDResidualLinear(nn.Module):
    """nn.Linear whose weight = frozen top-r SVD reconstruction + trainable residual."""

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to keep frozen

        # Original (frozen) main weight
        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def apply_svd_residual_to_self_attn(model, r):
    """Replace every nn.Linear inside any self_attn submodule with SVDResidualLinear."""
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        replace_with_svd_residual(sub_module, r),
                    )
        else:
            apply_svd_residual_to_self_attn(module, r)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def replace_with_svd_residual(module, r):
    """Build an SVDResidualLinear from an nn.Linear via SVD of its weight."""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = SVDResidualLinear(
            in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone()
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))  # do not exceed the number of singular values

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main.data.copy_(weight_main)

        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module


class EffortDetector(nn.Module):
    """CLIP ViT-L/14 vision tower + SVD-residual adapter + 2-class head.

    Parameter names mirror the official EffortDetector so the published
    checkpoint loads exactly. forward() takes a preprocessed image batch and
    returns the per-image fake probability.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = self._build_backbone(backbone_name)
        self.head = nn.Linear(1024, 2)

    def _build_backbone(self, backbone_name: str):
        from transformers import CLIPModel

        clip_model = CLIPModel.from_pretrained(backbone_name)
        # SVD-residual on self_attn only; ViT-L/14 keeps the top 1024-1 components.
        clip_model.vision_model = apply_svd_residual_to_self_attn(
            clip_model.vision_model, r=1024 - 1
        )
        return clip_model.vision_model

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)["pooler_output"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(images))
        return torch.softmax(logits, dim=1)[:, 1]  # fake probability
```

- [ ] **Step 3: Append the loader, preprocessing, and `score_images`**

Append to `effort_scorer.py` (after `EffortDetector`):

```python
def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("EFFORT_CKPT", DEFAULT_CKPT)


def _backbone_name() -> str:
    return os.environ.get("EFFORT_CLIP", DEFAULT_BACKBONE)


def _load(device: str):
    """Build the Effort detector and load the official checkpoint once, then cache."""
    global _model, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, device

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Effort checkpoint not found at {ckpt!r}. Set EFFORT_CKPT to the "
            f"official GenImage(sdv1.4) weights (effort_clip_L14_trainOn_sdv14.pth)."
        )

    model = EffortDetector(_backbone_name())
    # Official load recipe: unwrap state_dict, strip DataParallel 'module.' prefix.
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    # strict=False tolerates benign mismatches (e.g. position_ids), but the
    # trained head + SVD residuals MUST have matched the model. Guard the exact
    # silent-failure mode that would otherwise yield a randomly-initialized head.
    critical_unloaded = [
        k for k in state
        if (k.startswith("head.") or k.endswith("_residual"))
        and k in result.unexpected_keys
    ]
    if critical_unloaded:
        raise RuntimeError(
            f"Effort checkpoint loaded but critical weights did not match the "
            f"model (first few): {critical_unloaded[:5]}. Check the backbone/arch."
        )
    model.to(device).eval()

    _model, _loaded_device = model, device
    return model, device


def _preprocess(pil_images, device):
    """Replicate the official demo: resize 224, RGB, ToTensor, CLIP-normalize."""
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    batch = torch.stack([tf(img.convert("RGB")) for img in pil_images], dim=0)
    return batch.to(device)


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {effort-real-score} dict per PIL image."""
    model, device = _load(device)
    results: list[dict] = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        processed = _preprocess(batch, device)
        with torch.no_grad():
            fake_prob = model(processed)
        assert fake_prob.min() >= 0 and fake_prob.max() <= 1
        results.extend(probs_to_scores(fake_prob.cpu()))
    return results
```

- [ ] **Step 4: Verify the module imports cleanly and unit tests still pass**

The aggregation tests must keep passing, and importing the module must not require a GPU or the checkpoint (loading is lazy, inside `_load`).

Run:
```bash
python evaluation/benchmarks/Effort/test_effort_scorer.py
python -c "import sys; sys.path.insert(0,'.'); from evaluation.benchmarks.Effort.effort_scorer import score_images; print('import OK')"
```
Expected:
```
OK: all probs_to_scores tests passed
import OK
```
(If `transformers`/`torchvision` import fails, the `visualquality` env is missing them — install there.)

- [ ] **Step 5: Commit**

```bash
git add evaluation/benchmarks/Effort/effort_scorer.py
git commit -m "Add Effort model definition, checkpoint loading, preprocessing, score_images"
```

---

## Task 3: Wire `effort` into score-images.py

**Files:**
- Modify: `evaluation/metrics/score-images.py`

- [ ] **Step 1: Register the metric name**

In `AVAILABLE_METRICS` (lines 26-32), add `"effort"` after `"diffdoctor",`. After the edit the list ends like:

```python
AVAILABLE_METRICS = [
    "pickscore", "imagereward", "aesthetic", "hpsv3", "deqa", "visualquality_r1",
    "ocr", "geneval", "wise", "dpg-score", "dpg-score-mplug", "spatial-geneval",
    "sd-safety-checker", "shieldgemma",
    "dalleval-bias-gender", "dalleval-bias-attribute", "dalleval-bias-skintone",
    "diffdoctor", "effort",
]
```

- [ ] **Step 2: Register the output keys**

In `METRIC_OUTPUT_KEYS` (the diffdoctor entry is at line 64), add an `effort` entry right after it:

```python
    "diffdoctor": ("diffdoctor-clean-rate", "diffdoctor-clean-area"),
    "effort": ("effort-real-score",),
```

- [ ] **Step 3: Add the in-place scorer function**

Immediately after `_score_diffdoctor_in_place` (it ends with `torch.cuda.empty_cache()` at line 827), insert:

```python
def _score_effort_in_place(todo_rows):
    from evaluation.benchmarks.Effort.effort_scorer import (
        score_images as score_effort,
    )

    n = len(todo_rows)
    if n == 0:
        return
    batch_size = int(os.environ.get("EFFORT_BATCH_SIZE", "8"))
    print(f"[effort] {n} images to score; batch_size={batch_size}")

    for start in tqdm(range(0, n, batch_size), desc="effort"):
        batch_rows = todo_rows[start : start + batch_size]
        images = [_open_rgb_image(r["image_path"]) for r in batch_rows]
        score_dicts = score_effort(images, device="cuda", batch_size=batch_size)
        if len(score_dicts) != len(batch_rows):
            raise RuntimeError(
                f"effort returned {len(score_dicts)} scores for "
                f"{len(batch_rows)} images"
            )
        for row, scores in zip(batch_rows, score_dicts):
            row["scores"].update(scores)
    torch.cuda.empty_cache()
```

- [ ] **Step 4: Add the dispatch branch in `main()`**

In `main()`, right after the `diffdoctor` dispatch block (lines 943-945):

```python
        if metric == "diffdoctor":
            _score_diffdoctor_in_place(todo)
            continue
```

insert:

```python
        if metric == "effort":
            _score_effort_in_place(todo)
            continue
```

- [ ] **Step 5: Verify the CLI accepts the metric and keys are registered**

Run:
```bash
python -c "import sys; sys.path.insert(0,'.'); import importlib.util as u; spec=u.spec_from_file_location('si','evaluation/metrics/score-images.py'); m=u.module_from_spec(spec); spec.loader.exec_module(m); assert 'effort' in m.AVAILABLE_METRICS; assert m.METRIC_OUTPUT_KEYS['effort']==('effort-real-score',); print('score-images wiring OK')"
```
Expected: `score-images wiring OK`

- [ ] **Step 6: Commit**

```bash
git add evaluation/metrics/score-images.py
git commit -m "Wire effort metric into score-images.py"
```

---

## Task 4: Wire `effort` into run-bestofn.sh

**Files:**
- Modify: `evaluation/run-bestofn.sh`

- [ ] **Step 1: Append effort to the aigi-detector metric list**

Replace the aigi-detector case block (lines ~81-83):

```bash
    # aigi-detector: 1000 image-level MSCOCO val2014 prompts. Scored by the
    # DiffDoctor artifact detector; results flat-averaged into average_scores.json.
    aigi-detector)    metric_list=(diffdoctor) ;;
```

with:

```bash
    # aigi-detector: 1000 image-level MSCOCO val2014 prompts. Scored by two
    # detectors — DiffDoctor (pixel artifacts) and Effort (AIGI detectability);
    # results flat-averaged into average_scores.json.
    aigi-detector)    metric_list=(diffdoctor effort) ;;
```

- [ ] **Step 2: Map the metric to the visualquality env**

In the `metric_env` associative array, add a line right after `[diffdoctor]=visualquality` (line 101):

```bash
    [diffdoctor]=visualquality
    [effort]=visualquality
```

- [ ] **Step 3: Update the Stage 3 aigi-detector comment**

Replace the aigi-detector Stage 3 branch (lines ~177-181):

```bash
elif [[ "${dataset}" == "aigi-detector" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped (aigi-detector: flat-average metric)"
    echo "  See ${output_dir}/average_scores.json"
    echo "============================================"
```

with:

```bash
elif [[ "${dataset}" == "aigi-detector" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped (aigi-detector: flat-average metrics)"
    echo "  DiffDoctor + Effort scores in ${output_dir}/average_scores.json"
    echo "============================================"
```

- [ ] **Step 4: Verify the script still parses and the wiring is present**

Run:
```bash
bash -n evaluation/run-bestofn.sh && echo "syntax OK"
grep -n "aigi-detector)    metric_list=(diffdoctor effort)" evaluation/run-bestofn.sh
grep -n "\[effort\]=visualquality" evaluation/run-bestofn.sh
```
Expected: `syntax OK` plus one matching line from each grep.

- [ ] **Step 5: Commit**

```bash
git add evaluation/run-bestofn.sh
git commit -m "Wire effort as a second aigi-detector scorer in run-bestofn.sh"
```

---

## Task 5: Server smoke test (manual, requires GPU + checkpoint)

This task runs only on the server where the GPU and the Effort checkpoint are available. It is verification — it produces a commit only if it surfaces a bug to fix.

**Files:** none (runtime verification)

- [ ] **Step 1: Confirm the checkpoint is reachable**

Run:
```bash
ls -la /data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/effort_clip_L14_trainOn_sdv14.pth
```
Expected: the weights file exists. If absent, set `EFFORT_CKPT` to the correct path before continuing.

- [ ] **Step 2: Score a few images and check ranges + load integrity**

In the `visualquality` env, run a self-contained check. Use two clearly-different images if available (a real photo and an obviously-AI image); otherwise any aigi-detector outputs. The key assertions: scores are in `[0,1]`, and `_load` did NOT raise (the critical-weights guard passed → head + residuals loaded, not random):

```bash
conda activate visualquality
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from PIL import Image
from evaluation.benchmarks.Effort.effort_scorer import score_images

# Point this at any folder with a few images (e.g. an aigi-detector images/ dir).
d = os.environ.get("EFFORT_SMOKE_DIR", "flow_grpo/DiffDoctor/asset/input")
paths = [os.path.join(d, p) for p in os.listdir(d)
         if p.lower().endswith((".jpg", ".jpeg", ".png"))][:8]
imgs = [Image.open(p).convert("RGB") for p in paths]
scores = score_images(imgs, device="cuda", batch_size=4)
assert len(scores) == len(imgs)
for p, s in zip(paths, scores):
    rs = s["effort-real-score"]
    assert 0.0 <= rs <= 1.0, (p, s)
    print(f"{os.path.basename(p):40s} effort-real-score={rs:.4f}")
print("OK: effort smoke test passed")
PY
```
Expected: one line per image with `effort-real-score` in `[0, 1]`, ending with `OK: effort smoke test passed`. (If `_load` raises `RuntimeError: ...critical weights did not match...`, the backbone/arch is wrong — check `EFFORT_CLIP` resolves to CLIP ViT-L/14.)

- [ ] **Step 3: Sanity-check discrimination**

Effort's score is only meaningful on real generated content. On a handful of real photos vs. obviously-AI / heavily-artifacted images, real photos should get a noticeably higher `effort-real-score` than the AI ones. If all images return near-identical mid-range scores, suspect a silent load issue (re-check Step 2's guard did not get bypassed) or a preprocessing mismatch (the CLIP normalization must be present).

- [ ] **Step 4: End-to-end dry check on a few aigi-detector rows (optional)**

If an `aigi-detector` generation output dir already exists, score it to confirm the full path works and `average_scores.json` gains the effort key:

```bash
conda activate visualquality
CUDA_VISIBLE_DEVICES=0 python evaluation/metrics/score-images.py \
    --output_dir <existing_aigi_output_dir> --metrics effort
python -c "import json; a=json.load(open('<existing_aigi_output_dir>/average_scores.json')); print({k:a[k] for k in a if k.startswith('effort')})"
```
Expected: a dict containing `effort-real-score` in `[0, 1]`.

---

## Self-Review

- **Spec coverage:**
  - Scorer module (spec a) → Tasks 1-2. CLIP normalization + `[:,1]` fake-prob + `1-fake` → Task 2 Steps 2-3. Official load recipe (`get("state_dict")`, strip `module.`, `strict=False`) + critical-weights guard → Task 2 Step 3. Lazy cache, FileNotFoundError, range assert → Task 2 Step 3. `EFFORT_CKPT` / `EFFORT_CLIP` env overrides → Task 2 Step 1/3.
  - `score-images.py` edits: AVAILABLE_METRICS, METRIC_OUTPUT_KEYS, in-place branch, dispatch (spec b) → Task 3. `EFFORT_BATCH_SIZE` → Task 3 Step 3.
  - `run-bestofn.sh` edits: metric_list `(diffdoctor effort)`, `metric_env[effort]=visualquality`, Stage 3 comment (spec c) → Task 4.
  - `run-bestofn-batch.sh` (spec d) → confirmed no-op (File Structure note).
  - Data flow / flat-average口径 → Task 4 (Stage 3 stays skipped) + average_scores.json verified in Task 5 Step 4.
  - Testing: CPU unit test → Task 1; server smoke + discrimination sanity → Task 5.
- **Placeholder scan:** no TBD/TODO; every code step shows full code. The `<existing_aigi_output_dir>` in Task 5 Step 4 is a runtime path the operator fills, not a code placeholder.
- **Type consistency:** `probs_to_scores(fake_probs)` and `score_images(pil_images, device, batch_size)` signatures and the single score key (`effort-real-score`) are identical across Tasks 1-5. `_load` returns `(model, device)` (2-tuple) and `score_images` unpacks exactly that — no `preprocessor` third element (unlike DiffDoctor, Effort needs none).
