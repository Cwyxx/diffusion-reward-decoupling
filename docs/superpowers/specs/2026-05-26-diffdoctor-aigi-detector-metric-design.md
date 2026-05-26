# DiffDoctor as an aigi-detector Evaluation Metric — Design

Date: 2026-05-26

## Goal

Use the DiffDoctor artifact detector (ICCV 2025, SegFormer-based) as a scoring
metric for the `aigi-detector` Best-of-N dataset, which currently generates
images but has no scorer (`metric_list=()`, "scorer TBD"). The metric quantifies
how artifact-free a generated image is.

## Background

DiffDoctor's artifact detector is a SegFormer (`nvidia/mit-b5` backbone) with the
decode head's classifier replaced by a single-channel `Conv2d`. Given an RGB
image it outputs a per-pixel artifact heatmap in `[0, 1]` (`sigmoid` of the
logits): higher = more likely an artifact. The official inference demo lives at
`flow_grpo/DiffDoctor/ad_inference.py`.

We aggregate each image's heatmap into two image-level scores (threshold
`τ = 0.5`, the same line used for both):

- `diffdoctor-clean-rate` = `1.0` if `heatmap.max() < τ` else `0.0`
  — image-level pass: the whole image has no single artifact spot.
- `diffdoctor-clean-area` = `1 - (heatmap > τ).mean()`
  — fraction of the image that is clean (continuous, severity-aware).

Both are higher-is-better and in `[0, 1]`.

## Integration approach (chosen)

**Standalone scorer module + in-place dispatch branch in `score-images.py`**,
mirroring the existing `sd-safety-checker` / `shieldgemma` ResponsibleAI scorers.

Rejected alternatives:
- Inlining the SegFormer load directly in `score-images.py` (already 987 lines;
  violates the "benchmark-specific code lives under `evaluation/benchmarks/<Name>/`"
  convention).
- Registering in `flow_grpo/rewards.py` `multi_score` (its contract is one scalar
  per image; we emit two keys and depend on a local checkpoint + SegFormer not in
  the registry).

## Components

### (a) `evaluation/benchmarks/DiffDoctor/diffdoctor_scorer.py` (new)

Public API:

```python
def score_images(images, device="cuda", batch_size=8) -> list[dict]:
    # one dict per image:
    # {"diffdoctor-clean-rate": 0.0|1.0, "diffdoctor-clean-area": float in [0,1]}
```

- Lazy-load the model once per process and cache in a module global (mirrors
  `_build_mplug_vqa`): `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")`
  for the architecture, replace `model.decode_head.classifier` with
  `nn.Conv2d(in_channels, 1, kernel_size=1)`, then
  `load_state_dict(torch.load(ckpt))`, `.to(device)`, `.eval()`.
- Checkpoint path: constant default
  `/data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/ad_pytorch_model.bin`,
  overridable via env var `DIFFDOCTOR_CKPT`.
- Threshold: constant `0.5`, overridable via env var `DIFFDOCTOR_THRESHOLD`.
- Preprocessing **replicates `ad_inference.py` exactly**: resize to 512×512, RGB,
  `transforms.ToTensor()` (→ `[0,1]`), `SegformerImageProcessor(..., do_rescale=False)`,
  forward, `nn.functional.interpolate(pred.logits, size=512, mode="bilinear",
  align_corners=False)`, `sigmoid`.
- Heatmap→score aggregation is a separate pure function so it can be unit-tested
  on CPU without the checkpoint.

### (b) `evaluation/metrics/score-images.py` (3 small edits)

- Add `"diffdoctor"` to `AVAILABLE_METRICS`.
- Add `METRIC_OUTPUT_KEYS["diffdoctor"] = ("diffdoctor-clean-rate",
  "diffdoctor-clean-area")` so `_has_metric_score` / resume works.
- Add a `main()` dispatch branch `_score_diffdoctor_in_place(todo)` modeled on
  `_score_sd_safety_checker_in_place`: iterate in batches, open RGB images, call
  `score_images`, `row["scores"].update(...)`, then `torch.cuda.empty_cache()`.

### (c) `evaluation/run-bestofn.sh` (3 small edits)

- `aigi-detector) metric_list=(diffdoctor) ;;` (replace the empty list).
- Add `[diffdoctor]=visualquality` to the `metric_env` map.
- Keep the Stage 3 `aigi-detector` branch as a skip; update its comment from
  "scorer TBD" to point at `average_scores.json`.

### (d) `evaluation/run-bestofn-batch.sh`

Update any aigi-detector metric_list / comment references to stay consistent
with (c).

## Data flow

```
run-bestofn.sh (aigi-detector)
  Stage1: generate → evaluation_results.jsonl + images/
  Stage2: conda activate visualquality
          score-images.py --metrics diffdoctor
            → _score_diffdoctor_in_place(todo)
                per image: heatmap = sigmoid(model(img))   # [1,512,512]
                           clean-rate = 1.0 if heatmap.max() < 0.5 else 0.0
                           clean-area = 1 - (heatmap > 0.5).mean()
                row["scores"].update(...)
            → main() averages all numeric scores → average_scores.json
  Stage3: skip (read average_scores.json)
```

Final aggregates in `average_scores.json`:
- `diffdoctor-clean-rate` = fraction of images with `max < 0.5`
  = clean_rate@(max<0.5).
- `diffdoctor-clean-area` = mean over images of `1 - area_ratio`.

Statistic: flat average over all generated images (every prompt × every seed).
No Best-of-N max-over-N curve — Stage 3 stays skipped.

## Error handling

- Missing checkpoint → raise a clear `FileNotFoundError` naming the path and
  noting it may need `git lfs pull` or a corrected `DIFFDOCTOR_CKPT`.
- Follow the existing scorer convention: errors propagate; `main()`'s end-of-run
  jsonl rewrite never runs on failure, so a re-run resumes from on-disk scores.
- Assert each heatmap is in `[0,1]` and both output keys are present.

## Testing

- **CPU unit test** (no GPU/checkpoint): exercise the heatmap→score pure
  function with synthetic heatmaps:
  - all-zeros → clean-rate=1, clean-area=1
  - one pixel > 0.5 → clean-rate=0, clean-area≈1
  - half the pixels > 0.5 → clean-rate=0, clean-area≈0.5
- **Server smoke test** (with checkpoint): run on the images in
  `flow_grpo/DiffDoctor/asset/input`; verify the two keys exist, values are in
  range, and the `max`/`area` match the heatmap from `ad_inference.py` under the
  same preprocessing.

## Prerequisites

- The DiffDoctor checkpoint is already present on the server at the configured
  default path
  `/data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/ad_pytorch_model.bin`
  (~339 MB), so no `git lfs pull` is needed there. (Note: the in-repo copy
  `flow_grpo/DiffDoctor/checkpoints/ad_pytorch_model.bin` is only a 134-byte
  git-lfs pointer and is NOT used; the scorer loads from the server path above,
  overridable via `DIFFDOCTOR_CKPT`.)
- The `visualquality` conda env must have `transformers` (SegFormer) and
  `torchvision`.
- `nvidia/mit-b5` config + image processor are fetched from Hugging Face
  (`HF_ENDPOINT` mirror is already exported in the run scripts).

## Out of scope

- Best-of-N max-over-N aggregation curve for artifacts.
- Using DiffDoctor as a training reward (already implemented upstream in
  `flow_grpo/DiffDoctor/train_diffusion_model.py`).
- The RichHF baseline checkpoint (selectable later via `DIFFDOCTOR_CKPT`).
