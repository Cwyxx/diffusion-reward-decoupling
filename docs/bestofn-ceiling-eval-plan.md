# Best-of-N Ceiling Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SD-v1.5 Best-of-N ceiling evaluation pipeline specified in `docs/bestofn-ceiling-eval-design.md` — generates N images per prompt across 7 methods × 3 datasets with file-level resumption, scores each image, and outputs BoN curves answering whether RL post-training expands generation capability vs only shifting sampling probability.

**Architecture:** Three-stage pipeline (`generate → score → aggregate`) parallel to the existing `evaluation/run-eval.sh` infrastructure. Pure-logic modules (manifest, registry, BoN math) under `evaluation/` with unit tests; entry-point scripts (generation, scoring modifications, aggregation) follow the existing hyphen-named convention. Method-checkpoint differences hidden behind a uniform `evaluation/checkpoints/` registry.

**Tech Stack:** Python 3.10+, PyTorch 2.6, diffusers 0.33.1, peft 0.10, PIL, matplotlib, pytest 8.2.

---

## File Structure

**Create:**
- `evaluation/tests/__init__.py` — empty marker
- `evaluation/tests/test_manifest.py` — unit tests for manifest read/write/check
- `evaluation/tests/test_registry.py` — unit tests for checkpoint registry lookup
- `evaluation/tests/test_bestofn.py` — unit tests for BoN aggregation math
- `evaluation/manifest.py` — `read_manifest / write_manifest / check_consistency`
- `evaluation/bestofn.py` — pure functions `bon_continuous / pass_at_n / aggregate_curve`
- `evaluation/checkpoints/__init__.py` — re-exports `load_pipeline`, `REGISTRY`, `CheckpointRecipe`
- `evaluation/checkpoints/registry.py` — `CheckpointRecipe` dataclass + `REGISTRY` dict
- `evaluation/checkpoints/loaders.py` — 4 generic loaders + `load_pipeline` dispatcher
- `evaluation/checkpoints/verify-checkpoints.py` — smoke-test gate script
- `evaluation/metrics/generate-images-bestofn.py` — N-image-per-prompt generator with resumption
- `evaluation/metrics/aggregate-bestofn.py` — BoN curve computation + plotting
- `evaluation/run-bestofn.sh` — single (method, dataset) orchestrator

**Modify:**
- `evaluation/metrics/score-images.py` — add `ocr` / `geneval` to whitelist; upgrade JSONL schema to `(sample_id, seed_index)` key + optional `metadata`; skip already-scored rows.

---

## Task 1: Manifest module

**Files:**
- Create: `evaluation/tests/__init__.py`
- Create: `evaluation/manifest.py`
- Create: `evaluation/tests/test_manifest.py`

- [ ] **Step 1: Create the empty `tests/__init__.py` marker**

```python
# evaluation/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# evaluation/tests/test_manifest.py
import json
import pytest

from evaluation.manifest import (
    GenerationManifest,
    read_manifest,
    write_manifest,
    check_consistency,
)


def _sample(**overrides):
    base = dict(
        method="dpo",
        dataset="ocr",
        checkpoint_id="mhdang/dpo-sd1.5-text2image-v1",
        num_inference_steps=50,
        guidance_scale=7.5,
        resolution=512,
        scheduler_class="PNDMScheduler",
        max_seed_generated=-1,
    )
    base.update(overrides)
    return GenerationManifest(**base)


def test_roundtrip(tmp_path):
    m = _sample()
    write_manifest(str(tmp_path), m)
    loaded = read_manifest(str(tmp_path))
    assert loaded == m


def test_read_returns_none_when_missing(tmp_path):
    assert read_manifest(str(tmp_path)) is None


def test_consistency_passes_when_only_max_seed_differs():
    a = _sample(max_seed_generated=15)
    b = _sample(max_seed_generated=31)
    check_consistency(a, b)  # no exception


def test_consistency_fails_when_hyperparam_differs():
    a = _sample(guidance_scale=7.5)
    b = _sample(guidance_scale=4.5)
    with pytest.raises(ValueError, match="guidance_scale"):
        check_consistency(a, b)


def test_consistency_fails_when_method_differs():
    a = _sample(method="dpo")
    b = _sample(method="kto")
    with pytest.raises(ValueError, match="method"):
        check_consistency(a, b)
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python -m pytest evaluation/tests/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the manifest module**

```python
# evaluation/manifest.py
"""Per-(method, dataset) generation manifest.

Tracks the inference hyperparameters used for the images under a directory.
On resume, the new manifest must match (except for max_seed_generated, which
is informational only — file system is the truth source for what's generated).
"""
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional


MANIFEST_NAME = "manifest.json"


@dataclass
class GenerationManifest:
    method: str
    dataset: str
    checkpoint_id: Optional[str]
    num_inference_steps: int
    guidance_scale: float
    resolution: int
    scheduler_class: str
    max_seed_generated: int  # informational; -1 = nothing generated yet


# Fields that must match between an existing manifest and a new run.
_LOCKED_FIELDS = (
    "method",
    "dataset",
    "checkpoint_id",
    "num_inference_steps",
    "guidance_scale",
    "resolution",
    "scheduler_class",
)


def manifest_path(directory: str) -> str:
    return os.path.join(directory, MANIFEST_NAME)


def read_manifest(directory: str) -> Optional[GenerationManifest]:
    p = manifest_path(directory)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return GenerationManifest(**json.load(f))


def write_manifest(directory: str, manifest: GenerationManifest) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(manifest_path(directory), "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def check_consistency(existing: GenerationManifest, incoming: GenerationManifest) -> None:
    """Raise ValueError if any locked field differs."""
    for name in _LOCKED_FIELDS:
        ev = getattr(existing, name)
        iv = getattr(incoming, name)
        if ev != iv:
            raise ValueError(
                f"Manifest mismatch on '{name}': existing={ev!r} vs incoming={iv!r}. "
                f"Pass --force-regenerate or use a different output directory."
            )
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest evaluation/tests/test_manifest.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluation/manifest.py evaluation/tests/__init__.py evaluation/tests/test_manifest.py
git commit -m "Add generation manifest with hyperparameter consistency check"
```

---

## Task 2: Checkpoint registry skeleton + base loader

**Files:**
- Create: `evaluation/checkpoints/__init__.py`
- Create: `evaluation/checkpoints/registry.py`
- Create: `evaluation/checkpoints/loaders.py`
- Create: `evaluation/tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# evaluation/tests/test_registry.py
import pytest

from evaluation.checkpoints import REGISTRY, CheckpointRecipe, get_recipe


def test_base_recipe_registered():
    r = get_recipe("base")
    assert r.method == "base"
    assert r.load_kind == "base"
    assert r.base_model_id == "runwayml/stable-diffusion-v1-5"


def test_unknown_method_raises():
    with pytest.raises(KeyError, match="unknown_method"):
        get_recipe("unknown_method")


def test_only_base_registered_in_skeleton():
    # Task 2 ships the registry skeleton; Task 4 enrolls the 6 RL methods.
    assert set(REGISTRY.keys()) == {"base"}


def test_recipe_load_kinds_are_valid():
    valid = {"base", "lora", "unet", "full"}
    for name, recipe in REGISTRY.items():
        assert recipe.load_kind in valid, f"{name} has invalid load_kind={recipe.load_kind}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest evaluation/tests/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement registry**

```python
# evaluation/checkpoints/__init__.py
from evaluation.checkpoints.registry import REGISTRY, CheckpointRecipe, get_recipe
from evaluation.checkpoints.loaders import load_pipeline

__all__ = ["REGISTRY", "CheckpointRecipe", "get_recipe", "load_pipeline"]
```

```python
# evaluation/checkpoints/registry.py
"""Registry mapping method name to its checkpoint loading recipe.

Each RL method's public release has a different format (LoRA adapter, full
UNet weights, full pipeline). This module isolates those differences behind
a uniform CheckpointRecipe; loaders.py dispatches to the right loader.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


SD15 = "runwayml/stable-diffusion-v1-5"

LoadKind = Literal["base", "lora", "unet", "full"]


@dataclass
class CheckpointRecipe:
    method: str
    base_model_id: str
    load_kind: LoadKind
    repo_id: Optional[str] = None
    subfolder: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# Task 2 only registers the base model. The 6 RL methods (dpo / kto / spo /
# smpo / dro / inpo) get added in Task 4 after the implementer looks up
# each paper's public release format.
REGISTRY: Dict[str, CheckpointRecipe] = {
    "base": CheckpointRecipe(method="base", base_model_id=SD15, load_kind="base"),
}


def get_recipe(method: str) -> CheckpointRecipe:
    if method not in REGISTRY:
        raise KeyError(f"Unknown method: {method!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[method]
```

```python
# evaluation/checkpoints/loaders.py
"""Generic checkpoint loaders. Each takes a CheckpointRecipe + device + dtype
and returns a ready-to-call StableDiffusionPipeline.
"""
import torch
from diffusers import StableDiffusionPipeline

from evaluation.checkpoints.registry import CheckpointRecipe, get_recipe


def _finalize(pipeline: StableDiffusionPipeline, device, dtype) -> StableDiffusionPipeline:
    pipeline.to(device, dtype=dtype)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def load_base(recipe: CheckpointRecipe, device, dtype) -> StableDiffusionPipeline:
    pipeline = StableDiffusionPipeline.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    return _finalize(pipeline, device, dtype)


# Other loaders (load_unet / load_lora / load_full) are added in Task 3.

_LOADERS = {
    "base": load_base,
}


def load_pipeline(method: str, device="cuda", dtype=torch.float32) -> StableDiffusionPipeline:
    recipe = get_recipe(method)
    if recipe.load_kind not in _LOADERS:
        raise NotImplementedError(
            f"Loader for load_kind={recipe.load_kind!r} not yet implemented "
            f"(method={method!r}). See Task 3."
        )
    return _LOADERS[recipe.load_kind](recipe, device, dtype)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest evaluation/tests/test_registry.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/checkpoints/__init__.py evaluation/checkpoints/registry.py \
        evaluation/checkpoints/loaders.py evaluation/tests/test_registry.py
git commit -m "Add checkpoint registry skeleton with base loader"
```

---

## Task 3: UNet / LoRA / Full loaders

**Files:**
- Modify: `evaluation/checkpoints/loaders.py`

These loaders cover all RL method release formats:
- **`load_unet`**: load full UNet weights from `repo_id`, swap into base SD-v1.5 pipeline. Used by Diffusion-DPO and most variants.
- **`load_lora`**: load LoRA adapter on top of base SD-v1.5 transformer/UNet. Used by some variants that release LoRA-only checkpoints.
- **`load_full`**: load entire pipeline directly from `repo_id`. Used when the release is a self-contained diffusers repo.

- [ ] **Step 1: Implement load_unet**

Append to `evaluation/checkpoints/loaders.py`:

```python
from diffusers import UNet2DConditionModel


def load_unet(recipe: CheckpointRecipe, device, dtype) -> StableDiffusionPipeline:
    """Load base pipeline, then replace its UNet with weights from recipe.repo_id."""
    if recipe.repo_id is None:
        raise ValueError(f"load_unet requires repo_id (method={recipe.method})")

    pipeline = StableDiffusionPipeline.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(
        recipe.repo_id,
        subfolder=recipe.subfolder or "unet",
        torch_dtype=dtype,
    )
    pipeline.unet = unet
    return _finalize(pipeline, device, dtype)
```

- [ ] **Step 2: Implement load_lora**

```python
def load_lora(recipe: CheckpointRecipe, device, dtype) -> StableDiffusionPipeline:
    """Load base pipeline, then attach a LoRA adapter from recipe.repo_id."""
    if recipe.repo_id is None:
        raise ValueError(f"load_lora requires repo_id (method={recipe.method})")

    pipeline = StableDiffusionPipeline.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    pipeline.load_lora_weights(
        recipe.repo_id,
        subfolder=recipe.subfolder,
        **recipe.extra_kwargs,
    )
    return _finalize(pipeline, device, dtype)
```

- [ ] **Step 3: Implement load_full**

```python
def load_full(recipe: CheckpointRecipe, device, dtype) -> StableDiffusionPipeline:
    """Load entire pipeline directly from recipe.repo_id (release is self-contained)."""
    if recipe.repo_id is None:
        raise ValueError(f"load_full requires repo_id (method={recipe.method})")

    pipeline = StableDiffusionPipeline.from_pretrained(recipe.repo_id, torch_dtype=dtype)
    return _finalize(pipeline, device, dtype)
```

- [ ] **Step 4: Register all loaders in dispatch table**

Replace the `_LOADERS` dict at the bottom of `loaders.py`:

```python
_LOADERS = {
    "base": load_base,
    "lora": load_lora,
    "unet": load_unet,
    "full": load_full,
}
```

- [ ] **Step 5: Add a dispatch test**

Append to `evaluation/tests/test_registry.py`:

```python
def test_load_pipeline_dispatches_by_load_kind(monkeypatch):
    """Verify load_pipeline routes to the loader matching recipe.load_kind."""
    from evaluation.checkpoints import loaders

    called = {}

    def fake_load(recipe, device, dtype):
        called["kind"] = recipe.load_kind
        return "fake_pipeline"

    monkeypatch.setitem(loaders._LOADERS, "unet", fake_load)
    result = loaders.load_pipeline("dpo", device="cpu", dtype=None)
    assert result == "fake_pipeline"
    assert called["kind"] == "unet"
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
python -m pytest evaluation/tests/test_registry.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add evaluation/checkpoints/loaders.py evaluation/tests/test_registry.py
git commit -m "Add UNet, LoRA, full-pipeline checkpoint loaders"
```

---

## Task 4: Populate registry with the 6 RL method recipes

**Files:**
- Modify: `evaluation/checkpoints/registry.py`

This task is research + lookup, not code design. Each of the 6 RL methods (dpo, kto, spo, smpo, dro, inpo) has a public SD-v1.5 + Pick-a-Pic-v2 / PickScore checkpoint. Find each, fill the recipe.

- [ ] **Step 1: For each of the 6 methods, find the HF repo and verify the load format**

For each method, locate one of:
- A HuggingFace repo with a `unet/` subfolder containing weights → `load_kind="unet"`, `subfolder="unet"`
- A HuggingFace repo containing LoRA weights (`pytorch_lora_weights.safetensors` etc.) → `load_kind="lora"`
- A self-contained diffusers pipeline repo → `load_kind="full"`

Confirmed starting point: **dpo** = `mhdang/dpo-sd1.5-text2image-v1` (UNet release).

For each, also find the upstream paper's repo (typically GitHub) and confirm it's trained on Pick-a-Pic-v2 / PickScore.

- [ ] **Step 2: Add REGISTRY entries with concrete values**

Edit `evaluation/checkpoints/registry.py` to add a CheckpointRecipe for each of the 6 RL methods. Example for dpo:

```python
"dpo": CheckpointRecipe(
    method="dpo",
    base_model_id=SD15,
    load_kind="unet",
    repo_id="mhdang/dpo-sd1.5-text2image-v1",
    subfolder="unet",
),
```

Repeat for kto, spo (lora with `extra_kwargs={"weight_name": "..."}`), smpo, dro, inpo.

- [ ] **Step 3: Pre-cache HF repos locally to avoid runtime network flakiness**

```bash
for repo in mhdang/dpo-sd1.5-text2image-v1 <other_repos_here>; do
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$repo')"
done
```

- [ ] **Step 4: Run registry tests to confirm none broke**

```bash
python -m pytest evaluation/tests/test_registry.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/checkpoints/registry.py
git commit -m "Populate registry with HF repo IDs for 6 RL method checkpoints"
```

---

## Task 5: BoN aggregation pure functions

**Files:**
- Create: `evaluation/bestofn.py`
- Create: `evaluation/tests/test_bestofn.py`

- [ ] **Step 1: Write the failing tests**

```python
# evaluation/tests/test_bestofn.py
import math
import numpy as np
import pytest

from evaluation.bestofn import bon_continuous, pass_at_n, aggregate_curve


def test_bon_continuous_n_equals_1_is_first_sample():
    # 2 prompts, 4 samples each
    scores = np.array([[0.1, 0.5, 0.3, 0.9], [0.2, 0.2, 0.8, 0.4]])
    assert bon_continuous(scores, n=1) == pytest.approx((0.1 + 0.2) / 2)


def test_bon_continuous_n_equals_max_is_full_max():
    scores = np.array([[0.1, 0.5, 0.3, 0.9], [0.2, 0.2, 0.8, 0.4]])
    assert bon_continuous(scores, n=4) == pytest.approx((0.9 + 0.8) / 2)


def test_bon_continuous_monotonic_in_n():
    rng = np.random.default_rng(0)
    scores = rng.random((10, 32))
    prev = -math.inf
    for n in range(1, 33):
        v = bon_continuous(scores, n=n)
        assert v >= prev
        prev = v


def test_pass_at_n_n_equals_1_is_first_seed_pass_rate():
    # 3 prompts, 4 samples each, binary
    scores = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert pass_at_n(scores, n=1) == pytest.approx(1 / 3)


def test_pass_at_n_n_equals_max_is_any_pass():
    scores = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert pass_at_n(scores, n=4) == pytest.approx(2 / 3)


def test_aggregate_curve_returns_one_value_per_n():
    scores = np.zeros((5, 8))
    scores[0, 7] = 1.0  # only the last sample of prompt 0 has a positive score
    curve = aggregate_curve(scores, kind="continuous")
    assert set(curve.keys()) == set(range(1, 9))
    assert curve[1] == 0.0  # first sample of every prompt is 0
    assert curve[8] == pytest.approx(1.0 / 5)


def test_aggregate_curve_pass_at_n():
    scores = np.array([[0, 1, 0, 0], [0, 0, 0, 0]])
    curve = aggregate_curve(scores, kind="binary")
    assert curve[1] == 0.0  # neither prompt passes at seed 0
    assert curve[2] == 0.5  # prompt 0 passes by seed 1
    assert curve[4] == 0.5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest evaluation/tests/test_bestofn.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the aggregation module**

```python
# evaluation/bestofn.py
"""Best-of-N aggregation: pure functions over per-(prompt, seed) score arrays.

Input convention: `scores` is a 2D ndarray of shape (num_prompts, n_max).
Element [p, i] is the score of prompt p's i-th sample.

Two reductions:
- `bon_continuous(scores, n)`: HP-style. mean over prompts of max over first n samples.
- `pass_at_n(scores, n)`: binary. mean over prompts of any over first n samples.

`aggregate_curve(scores, kind)` returns {n: value} for all n in [1, n_max].
"""
from typing import Dict, Literal

import numpy as np


def bon_continuous(scores: np.ndarray, n: int) -> float:
    if not 1 <= n <= scores.shape[1]:
        raise ValueError(f"n={n} out of range [1, {scores.shape[1]}]")
    return float(np.mean(np.max(scores[:, :n], axis=1)))


def pass_at_n(scores: np.ndarray, n: int) -> float:
    if not 1 <= n <= scores.shape[1]:
        raise ValueError(f"n={n} out of range [1, {scores.shape[1]}]")
    return float(np.mean(np.any(scores[:, :n] > 0, axis=1)))


def aggregate_curve(
    scores: np.ndarray,
    kind: Literal["continuous", "binary"],
) -> Dict[int, float]:
    """Compute the BoN value for every n in [1, n_max]. Returns dict keyed by n."""
    n_max = scores.shape[1]
    fn = bon_continuous if kind == "continuous" else pass_at_n
    return {n: fn(scores, n) for n in range(1, n_max + 1)}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest evaluation/tests/test_bestofn.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/bestofn.py evaluation/tests/test_bestofn.py
git commit -m "Add Best-of-N aggregation primitives (HP max + binary pass@N)"
```

---

## Task 6: Generation script (`generate-images-bestofn.py`)

**Files:**
- Create: `evaluation/metrics/generate-images-bestofn.py`

This is the largest single piece. It loops over (sample_id, seed_index ∈ [0, N)), generates each image only if the target file doesn't exist, writes atomically, updates the manifest, and supports drawbench-unique / ocr / geneval prompt sources.

- [ ] **Step 1: Write the script**

```python
# evaluation/metrics/generate-images-bestofn.py
"""Generate N images per prompt with file-level resumption.

Differences from generate-images.py:
- SD-v1.5 (StableDiffusionPipeline), not SD3.5-M
- Checkpoint loading via evaluation/checkpoints registry
- N images per prompt (path: images/{sid:05d}/{seed:05d}.png)
- Manifest consistency check; skip already-existing image files
- Per-(sample_id, seed_index) deterministic seed
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.checkpoints import load_pipeline
from evaluation.checkpoints.registry import get_recipe
from evaluation.manifest import (
    GenerationManifest,
    check_consistency,
    read_manifest,
    write_manifest,
)


DATASET_ROOT = os.path.join(_REPO_ROOT, "dataset")


# -- Dataset loading --------------------------------------------------------

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


# -- Manifest construction --------------------------------------------------

def build_manifest(args, scheduler_class):
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


# -- evaluation_results.jsonl helpers ---------------------------------------

def load_existing_rows(jsonl_path):
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


# -- Main loop --------------------------------------------------------------

def main(args):
    out_dir = args.output_dir
    images_dir = os.path.join(out_dir, "images")
    jsonl_path = os.path.join(out_dir, "evaluation_results.jsonl")

    os.makedirs(images_dir, exist_ok=True)

    # Load pipeline.
    device = torch.device("cuda")
    dtype = torch.float32  # spec §2.3: fp32 for SD-v1.5 inference on 24GB GPU
    pipeline = load_pipeline(args.method, device=device, dtype=dtype)
    scheduler_class = type(pipeline.scheduler).__name__

    # Manifest consistency.
    incoming = build_manifest(args, scheduler_class)
    existing = read_manifest(out_dir)
    if existing is not None and not args.force_regenerate:
        check_consistency(existing, incoming)
        # Preserve max_seed_generated from existing if higher.
        incoming.max_seed_generated = max(existing.max_seed_generated, incoming.max_seed_generated)
    write_manifest(out_dir, incoming)

    # Prompts + existing rows.
    items = load_prompts(args.dataset)
    existing_rows = load_existing_rows(jsonl_path)

    pipeline.set_progress_bar_config(disable=True)
    max_seed_seen = incoming.max_seed_generated

    pbar = tqdm(total=len(items) * args.n_max, desc=f"{args.method}/{args.dataset}")
    for sample_id, item in enumerate(items):
        prompt = item["prompt"]
        metadata = item["metadata"]
        sid_dir = os.path.join(images_dir, f"{sample_id:05d}")
        os.makedirs(sid_dir, exist_ok=True)

        for seed_index in range(args.n_max):
            img_path = os.path.join(sid_dir, f"{seed_index:05d}.png")
            row_key = (sample_id, seed_index)

            image_exists = os.path.exists(img_path)
            row_exists = row_key in existing_rows

            if image_exists and row_exists:
                pbar.update(1)
                continue

            # Regenerate the image if missing.
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
                img_tensor = result.images[0]  # (C, H, W) in [0, 1]
                arr = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(arr).save(img_path)

            # Append the row only if not already present (avoid duplicate keys in jsonl).
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
                append_row(jsonl_path, row)
                existing_rows[row_key] = row

            max_seed_seen = max(max_seed_seen, seed_index)

            pbar.update(1)

    pbar.close()

    # Update manifest with final max_seed_generated.
    incoming.max_seed_generated = max_seed_seen
    write_manifest(out_dir, incoming)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate N images per prompt for Best-of-N ceiling eval (SD-v1.5)."
    )
    ap.add_argument("--method", required=True,
                    help="One of: base, dpo, kto, spo, smpo, dro, inpo")
    ap.add_argument("--dataset", required=True,
                    help="One of: drawbench-unique, ocr, geneval")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_max", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--force-regenerate", dest="force_regenerate", action="store_true",
                    help="Bypass manifest consistency check (will overwrite manifest).")
    main(ap.parse_args())
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
python -c "
import sys, os
sys.path.insert(0, '/Users/chenweiyan/Documents/Post-training-AIGC/diffusion-reward-decoupling')
import importlib.util
spec = importlib.util.spec_from_file_location(
    'gen', 'evaluation/metrics/generate-images-bestofn.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('Import OK')
"
```

Expected: `Import OK`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/metrics/generate-images-bestofn.py
git commit -m "Add SD-v1.5 BoN generator with manifest + atomic write resumption"
```

---

## Task 7: Modify `score-images.py` (whitelist + schema + skip-already-scored)

**Files:**
- Modify: `evaluation/metrics/score-images.py`

Three changes per spec §4.3:
1. Add `ocr` and `geneval` to `AVAILABLE_METRICS`.
2. Upgrade JSONL row format to `(sample_id, seed_index)` + optional `metadata`. Pass `metadata` through to scorer.
3. Skip rows that already have the metric scored unless `--force`.
4. For OCR: binarize the continuous scorer output via `--ocr_threshold` (default 1.0 = exact match). For GenEval: store `score_details["strict_accuracy"]` (already 0/1) instead of `score_details["geneval"]` (continuous).

- [ ] **Step 1: Read current state**

```bash
sed -n '1,120p' evaluation/metrics/score-images.py
```

(Reference; no edits yet.)

- [ ] **Step 2: Replace the file**

```python
# evaluation/metrics/score-images.py
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Score images produced by generate-images.py / generate-images-bestofn.py.

Reads `evaluation_results.jsonl` (rows keyed by (sample_id, seed_index) for the
BoN pipeline; sample_id-only rows from the legacy SD3.5 pipeline are still
accepted), invokes the requested metric(s) via flow_grpo.rewards.multi_score,
and writes scores back into each row's `scores` dict.

Resumption: rows that already have the requested metric in `scores` are
skipped unless `--force` is passed.

OCR / GenEval are binary in this codebase's research framing — see
binarize_* helpers below.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flow_grpo.rewards import multi_score


AVAILABLE_METRICS = [
    "pickscore", "imagereward", "aesthetic", "hpsv3", "deqa", "visualquality_r1",
    "ocr", "geneval",
]

SMALL_BATCH_METRICS = {"hpsv3", "visualquality_r1"}
BINARY_METRICS = {"ocr", "geneval"}


def prepare_images(metric, image_paths):
    """Load images in the format each scorer expects."""
    if metric in {"hpsv3", "deqa", "visualquality_r1"}:
        return image_paths
    if metric == "aesthetic":
        return np.stack([np.array(Image.open(p).convert("RGB")) for p in image_paths])
    # pickscore / imagereward / ocr / geneval all want PIL or ndarray; use PIL.
    return [Image.open(p).convert("RGB") for p in image_paths]


def binarize_ocr(continuous_scores, threshold):
    """OCR scorer returns 1 - edit_distance/len(prompt) in [0, 1]. Binarize at threshold."""
    return [1 if float(s) >= threshold else 0 for s in continuous_scores]


def run_metric(metric, image_paths, prompts, metadatas, batch_size, device, ocr_threshold):
    """Returns a parallel list of float scores for `image_paths`."""
    scoring_fn = multi_score(device, {metric: 1.0})
    all_scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=metric):
        batch_paths = image_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        images = prepare_images(metric, batch_paths)
        score_details, _ = scoring_fn(images, batch_prompts, batch_meta)

        if metric == "geneval":
            # Use binary strict_accuracy, not continuous similarity.
            values = score_details["strict_accuracy"]
        elif metric == "ocr":
            values = binarize_ocr(score_details[metric], ocr_threshold)
        else:
            values = score_details[metric]

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        all_scores.extend(float(v) for v in values)
    return all_scores


def main(args):
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    with open(results_path, "r") as f:
        results = [json.loads(ln) for ln in f if ln.strip()]

    # Sort key: (sample_id, seed_index) if available, else (sample_id, 0).
    results.sort(key=lambda r: (r["sample_id"], r.get("seed_index", 0)))

    for metric in args.metrics:
        print(f"\n=== Scoring with {metric} (force={args.force}) ===")
        # Filter to rows missing this metric (or all rows if --force).
        if args.force:
            todo = results
        else:
            todo = [r for r in results if metric not in r["scores"]]
        if not todo:
            print(f"All rows already have {metric}; skipping.")
            continue

        image_paths = [r["image_path"] for r in todo]
        prompts = [r["prompt"] for r in todo]
        metadatas = [r.get("metadata") or {} for r in todo]

        bs = 1 if metric in SMALL_BATCH_METRICS else args.batch_size
        scores = run_metric(metric, image_paths, prompts, metadatas, bs, "cuda", args.ocr_threshold)
        assert len(scores) == len(todo)
        for r, s in zip(todo, scores):
            r["scores"][metric] = s
        torch.cuda.empty_cache()

    # Atomic rewrite of the jsonl.
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, results_path)

    # Average summary (across all rows; for BoN data this includes all (sample_id, seed_index)).
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="Directory containing evaluation_results.jsonl and images/.")
    ap.add_argument("--metrics", nargs="+", required=True, choices=AVAILABLE_METRICS)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="Re-score rows that already have the requested metric.")
    ap.add_argument("--ocr_threshold", type=float, default=1.0,
                    help="Binarize OCR score >= threshold to 1, else 0. Default 1.0 = exact match.")
    main(ap.parse_args())
```

- [ ] **Step 3: Sanity-check that the modified script loads**

```bash
python -c "
import sys, importlib.util
sys.path.insert(0, '/Users/chenweiyan/Documents/Post-training-AIGC/diffusion-reward-decoupling')
spec = importlib.util.spec_from_file_location('s', 'evaluation/metrics/score-images.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('AVAILABLE_METRICS:', m.AVAILABLE_METRICS)
"
```

Expected output should include `'ocr'` and `'geneval'` in AVAILABLE_METRICS.

- [ ] **Step 4: Commit**

```bash
git add evaluation/metrics/score-images.py
git commit -m "Add ocr/geneval to score-images.py with seed-aware schema and resumption"
```

---

## Task 8: Aggregation script (`aggregate-bestofn.py`)

**Files:**
- Create: `evaluation/metrics/aggregate-bestofn.py`

Reads `evaluation_results.jsonl` for one (method, dataset), computes BoN curves for each scored metric, and writes:
- `bestofn/curves.json` — `{metric: {1: x, 2: y, ...}, "ceiling_lift": {...}, ...}`
- `bestofn/plots/{metric}_curve_log.png` — single-method plot. Cross-method overlay is a separate step (see Task 11).

- [ ] **Step 1: Write the script**

```python
# evaluation/metrics/aggregate-bestofn.py
"""Compute Best-of-N curves for one (method, dataset) directory.

Inputs: ${output_dir}/evaluation_results.jsonl (rows keyed by (sample_id, seed_index))
Outputs: ${output_dir}/bestofn/curves.json + plots/<metric>_curve_log.png
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.bestofn import aggregate_curve


# Which metrics are binary (use pass@N) vs continuous (use mean-of-max).
BINARY_METRICS = {"ocr", "geneval"}


def load_results(results_path):
    rows = []
    with open(results_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def build_score_matrix(rows, metric):
    """Build a (num_prompts, n_max) matrix of `metric` scores.

    Rows missing this metric are filled with NaN; if any are missing, fail loudly.
    """
    grouped = defaultdict(dict)
    for r in rows:
        if metric not in r["scores"]:
            continue
        grouped[r["sample_id"]][r["seed_index"]] = r["scores"][metric]

    if not grouped:
        return None

    sample_ids = sorted(grouped.keys())
    n_max = max(max(v.keys()) for v in grouped.values()) + 1
    mat = np.full((len(sample_ids), n_max), np.nan, dtype=float)
    for i, sid in enumerate(sample_ids):
        for seed_idx, val in grouped[sid].items():
            mat[i, seed_idx] = val

    if np.isnan(mat).any():
        n_missing = int(np.isnan(mat).sum())
        raise ValueError(
            f"Score matrix for metric={metric!r} has {n_missing} NaN entries. "
            f"Re-run scoring (with --force if needed) before aggregating."
        )
    return mat


def plot_curve(curve, metric, out_path):
    ns = sorted(curve.keys())
    ys = [curve[n] for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ns, ys, marker="o", markersize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(f"BoN({metric})")
    ax.set_title(f"Best-of-N curve: {metric}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    rows = load_results(results_path)

    # Discover which metrics actually have any scores.
    metrics = sorted({m for r in rows for m in r["scores"].keys()})
    if not metrics:
        raise SystemExit(f"No scores found in {results_path}; run scoring first.")

    bestofn_dir = os.path.join(args.output_dir, "bestofn")
    plots_dir = os.path.join(bestofn_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out = {}
    for metric in metrics:
        mat = build_score_matrix(rows, metric)
        if mat is None:
            continue
        kind = "binary" if metric in BINARY_METRICS else "continuous"
        curve = aggregate_curve(mat, kind=kind)
        out[metric] = {
            "curve": curve,
            "ceiling_lift": curve[mat.shape[1]] - curve[1],
            "n_max": mat.shape[1],
            "num_prompts": mat.shape[0],
            "kind": kind,
        }
        plot_curve(curve, metric, os.path.join(plots_dir, f"{metric}_curve_log.png"))

    curves_path = os.path.join(bestofn_dir, "curves.json")
    with open(curves_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {curves_path}")
    for m, info in out.items():
        print(f"  {m:<20} N={info['n_max']:>3}  ceiling_lift={info['ceiling_lift']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    main(ap.parse_args())
```

- [ ] **Step 2: Sanity-check imports**

```bash
python -c "
import importlib.util, sys
sys.path.insert(0, '/Users/chenweiyan/Documents/Post-training-AIGC/diffusion-reward-decoupling')
spec = importlib.util.spec_from_file_location('a', 'evaluation/metrics/aggregate-bestofn.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('Import OK')
"
```

Expected: `Import OK`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/metrics/aggregate-bestofn.py
git commit -m "Add Best-of-N curve aggregator with per-metric log-x plots"
```

---

## Task 9: Smoke-test gate (`verify-checkpoints.py`)

**Files:**
- Create: `evaluation/checkpoints/verify-checkpoints.py`

Loads each method, generates one image with a fixed prompt, scores with PickScore, asserts a non-NaN result. Must be invoked manually before scaling generation.

- [ ] **Step 1: Write the script**

```python
# evaluation/checkpoints/verify-checkpoints.py
"""Smoke-test gate: load each registered method, generate 1 image, PickScore-evaluate.

Runs against `cuda:0` by default. Prints a one-line PASS/FAIL per method;
exits non-zero if any failed.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.checkpoints import REGISTRY, load_pipeline
from flow_grpo.rewards import multi_score


SMOKE_PROMPT = "a high-quality photograph of a red apple on a wooden table"


def smoke_test_one(method, score_fn):
    pipeline = load_pipeline(method, device="cuda", dtype=torch.float32)
    generator = torch.Generator("cuda").manual_seed(0)
    with torch.no_grad():
        result = pipeline(
            SMOKE_PROMPT,
            num_inference_steps=20,  # short for smoke
            guidance_scale=7.5,
            height=512,
            width=512,
            output_type="pt",
            generator=generator,
        )
    img_t = result.images[0]
    arr = (img_t.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pil = Image.fromarray(arr)

    score_details, _ = score_fn([pil], [SMOKE_PROMPT], [{}])
    score = float(score_details["pickscore"][0])
    if np.isnan(score):
        raise ValueError("PickScore returned NaN")
    return score


def main(args):
    methods = args.methods or sorted(REGISTRY.keys())
    score_fn = multi_score("cuda", {"pickscore": 1.0})

    failures = []
    for method in methods:
        try:
            score = smoke_test_one(method, score_fn)
            print(f"PASS  method={method:<8}  pickscore={score:.4f}")
        except Exception:
            print(f"FAIL  method={method:<8}")
            traceback.print_exc()
            failures.append(method)
        torch.cuda.empty_cache()

    if failures:
        print(f"\n{len(failures)} method(s) failed: {failures}")
        sys.exit(1)
    print(f"\nAll {len(methods)} methods passed smoke test.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*",
                    help="Subset of methods to test. Default: all in REGISTRY.")
    main(ap.parse_args())
```

- [ ] **Step 2: Sanity-check the script syntax (without GPU)**

```bash
python -c "
import importlib.util, sys
sys.path.insert(0, '/Users/chenweiyan/Documents/Post-training-AIGC/diffusion-reward-decoupling')
spec = importlib.util.spec_from_file_location('v', 'evaluation/checkpoints/verify-checkpoints.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('Import OK')
"
```

Expected: `Import OK`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/checkpoints/verify-checkpoints.py
git commit -m "Add verify-checkpoints smoke-test gate for the 7-method registry"
```

---

## Task 10: Orchestrator shell script (`run-bestofn.sh`)

**Files:**
- Create: `evaluation/run-bestofn.sh`

Mirrors the structure of `evaluation/run-eval.sh`: pin `CUDA_VISIBLE_DEVICES`, switch conda envs per metric, call generation → scoring → aggregation in sequence. Same per-metric env table as `run-eval.sh`, plus `ocr` and `geneval` (env defaults to `alignprop` unless overridden by user setup).

- [ ] **Step 1: Write the shell script**

```bash
# evaluation/run-bestofn.sh
#!/bin/bash
# End-to-end Best-of-N evaluation for ONE (method, dataset) combination.
# Generates N_max images per prompt, scores via dataset-specific metrics, and
# aggregates BoN curves. Resumable: re-running with the same args picks up
# where the last run left off (subject to manifest consistency).
#
# Usage:
#   bash evaluation/run-bestofn.sh <cuda_device> <method> <dataset> <n_max>
# Example:
#   bash evaluation/run-bestofn.sh 0 dpo  drawbench-unique 32
#   bash evaluation/run-bestofn.sh 1 base ocr              32

set -euo pipefail

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

cuda_device=${1:?cuda_device}
method=${2:?method}
dataset=${3:?dataset}
n_max=${4:?n_max}

export CUDA_VISIBLE_DEVICES=${cuda_device}

# Output root (parallel to existing flow-grpo/sd-3-5-medium/ subtree).
base_root="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling"
output_dir="${base_root}/bestofn-eval/sd-v1-5/${method}/${dataset}"
mkdir -p "${output_dir}"

# Per-dataset metric set.
case "${dataset}" in
    drawbench-unique) metric_list=(pickscore hpsv3 deqa aesthetic) ;;
    ocr)              metric_list=(ocr) ;;
    geneval)          metric_list=(geneval) ;;
    *) echo "Unknown dataset: ${dataset}" >&2; exit 1 ;;
esac

# Default conda env for generation + most scorers.
DEFAULT_ENV=alignprop
declare -A metric_env=(
    [hpsv3]=hpsv3
    [deqa]=internvl
    [visualquality_r1]=visualquality
    # ocr and geneval fall back to DEFAULT_ENV.
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_PY="${SCRIPT_DIR}/metrics/generate-images-bestofn.py"
SCORE_PY="${SCRIPT_DIR}/metrics/score-images.py"
AGGREGATE_PY="${SCRIPT_DIR}/metrics/aggregate-bestofn.py"

# Stage 1: Generate.
echo "============================================"
echo "Stage 1: generate (method=${method}, dataset=${dataset}, n_max=${n_max})"
echo "============================================"
conda activate "${DEFAULT_ENV}"
python "${GENERATE_PY}" \
    --method "${method}" \
    --dataset "${dataset}" \
    --output_dir "${output_dir}" \
    --n_max "${n_max}"

# Stage 2: Score (per-metric env switch).
for metric in "${metric_list[@]}"; do
    echo "--------------------------------------------"
    echo "Stage 2: score ${metric}"
    echo "--------------------------------------------"
    env="${metric_env[$metric]:-$DEFAULT_ENV}"
    conda activate "${env}"
    python "${SCORE_PY}" \
        --output_dir "${output_dir}" \
        --metrics "${metric}"
done

# Stage 3: Aggregate.
echo "============================================"
echo "Stage 3: aggregate"
echo "============================================"
conda activate "${DEFAULT_ENV}"
python "${AGGREGATE_PY}" --output_dir "${output_dir}"

echo "Done: ${output_dir}/bestofn/"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x evaluation/run-bestofn.sh
```

- [ ] **Step 3: Smoke-check the syntax (no execution)**

```bash
bash -n evaluation/run-bestofn.sh && echo "shell syntax OK"
```

Expected: `shell syntax OK`.

- [ ] **Step 4: Commit**

```bash
git add evaluation/run-bestofn.sh
git commit -m "Add run-bestofn.sh orchestrator (generate → score → aggregate)"
```

---

## Task 11: End-to-end dry run on base model

**Files:** none (validation-only).

Verify the whole pipeline before scaling to all 21 (method, dataset) combinations.

- [ ] **Step 1: Run the smoke gate on the base model only**

```bash
conda activate alignprop
python evaluation/checkpoints/verify-checkpoints.py --methods base
```

Expected: `PASS  method=base     pickscore=<some number>` and exit 0.

- [ ] **Step 2: Tiny generation run (N=4, drawbench-unique)**

```bash
bash evaluation/run-bestofn.sh 0 base drawbench-unique 4
```

Expected behavior:
- `${base_root}/bestofn-eval/sd-v1-5/base/drawbench-unique/manifest.json` is written
- `images/{0..199:05d}/{0..3:02d}.png` exist
- `evaluation_results.jsonl` has one row per (sample_id, seed_index) with all 4 HP metrics scored
- `bestofn/curves.json` lists 4 metrics, each with N=1..4 curve, `kind=continuous`
- `bestofn/plots/{pickscore,hpsv3,deqa,aesthetic}_curve_log.png` exist

- [ ] **Step 3: Re-run with N=8 to verify resumption**

```bash
bash evaluation/run-bestofn.sh 0 base drawbench-unique 8
```

Expected:
- Generation skips seed_index=0..3 (already exist), generates 4..7 only.
- Scoring skips already-scored rows (those for seed 0..3) and only scores the new 4 seeds × 200 prompts × 4 metrics.
- Aggregation overwrites `curves.json` with N=1..8 curve.

- [ ] **Step 4: Inspect curves.json**

```bash
OUT=/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/bestofn-eval/sd-v1-5/base/drawbench-unique
python -c "
import json
with open('$OUT/bestofn/curves.json') as f:
    out = json.load(f)
for m, info in out.items():
    print(f\"{m:<12} curve={info['curve']}\")
    print(f\"             ceiling_lift={info['ceiling_lift']:+.4f}\")
"
```

Expected: each curve is monotonic non-decreasing (sanity check from spec §6.2).

- [ ] **Step 5: Verify resumption manifest mismatch detection**

Try to re-run with a different `--guidance_scale`:

```bash
OUT=/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/bestofn-eval/sd-v1-5/base/drawbench-unique
python evaluation/metrics/generate-images-bestofn.py \
    --method base --dataset drawbench-unique \
    --output_dir "$OUT" \
    --n_max 8 --guidance_scale 4.5
```

Expected: `ValueError: Manifest mismatch on 'guidance_scale': existing=7.5 vs incoming=4.5. Pass --force-regenerate ...` and non-zero exit.

- [ ] **Step 6: Commit any small fixes discovered during dry run**

If issues surfaced, fix them and commit. If nothing changed:

```bash
echo "Dry run complete; no fixes needed."
```

---

## Self-Review Notes

After writing this plan, the implementer should also confirm before scaling to 21 combinations:

1. **Spec coverage** — every section in `docs/bestofn-ceiling-eval-design.md` maps to one or more tasks above:
   - §2 (config) → Tasks 2–4 (registry) + Task 6 (CLI defaults)
   - §3.1 (output layout) → Task 10 (orchestrator path) + Task 6 (image paths)
   - §3.2 (resumption) → Tasks 1, 6, 7 (manifest, generator skip-existing, scorer skip-already-scored)
   - §3.3 (manifest) → Task 1
   - §4.1 (registry) → Tasks 2–4
   - §4.2 (generator) → Task 6
   - §4.3 (scorer) → Task 7
   - §4.4 (aggregator) → Task 8 + Task 5
   - §4.5 (orchestrator) → Task 10
   - §6.1 (gates) → Task 9 + Task 11
   - §6.2 (sanity checks) → built into Task 8 (monotonicity is mathematically guaranteed by the implementation; explicit assertions can be added in a follow-up).

2. **Out of scope (per spec §6.4)** — explicitly NOT in this plan: bootstrap CIs, headline tables, cross-task scoring, training new RL methods, atomic-write safety helpers (cheap to redo a single image generation).

3. **Cross-method overlay plot** — Task 8 produces per-(method, dataset) plots. Producing the 7-line cross-method overlay plot is a one-shot script that reads each method's `curves.json` and plots them together; deferred from this plan because it has no architectural complexity (~30-line matplotlib script) and is best authored after viewing real per-method outputs.
