# Effort as a Second aigi-detector Evaluation Metric — Design

Date: 2026-05-26

## Goal

Add the **Effort** AI-generated-image detector (ICML 2025 Oral, "Orthogonal
Subspace Decomposition for Generalizable AIGI Detection") as a second scoring
metric for the `aigi-detector` Best-of-N dataset, alongside the existing
DiffDoctor artifact metric. Effort quantifies how *detectable-as-AI* a generated
image is; we report its complement so that more realistic images score higher.

## Background

Effort inserts an orthogonal-subspace (SVD-residual) adapter into a CLIP
`ViT-L/14` vision tower: each `self_attn` linear layer is SVD-decomposed, the top
`r = 1024-1 = 1023` singular components are frozen (main weight) and the
remaining residual components are trainable. A `nn.Linear(1024, 2)` head on top
of the pooled feature classifies `0 = Real` vs `1 = Fake`. The official inference
demo lives at `flow_grpo/Effort-AIGI-Detection/DeepfakeBench/training/demo.py`
(+ `detectors/effort_detector.py`).

We use the official **GenImage (sdv1.4)** checkpoint (CLIP-L14 + Effort, trained
to detect natural AI-generated images). Per image we emit a single
higher-is-better score in `[0, 1]`:

- `effort-real-score` = `1 - fake_prob`, where
  `fake_prob = softmax(head(feature))[:, 1]`.

Higher = the detector judges the image more real / less obviously AI-generated.

## Decisions locked during brainstorming

- **Checkpoint**: official **GenImage (sdv1.4)**, already on the server at
  `/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/effort_clip_L14_trainOn_sdv14.pth`.
- **Output semantics (Option A)**: replicate the *official* pipeline —
  preprocessing **includes CLIP normalization**, and the score uses the
  **`[:, 1]` (fake) probability**, then `1 - fake_prob` to make it
  higher-is-better. We do NOT follow the simplified "RealGen" variant found on
  GitHub, which drops normalization and uses `[:, 0]`; that variant is bound to a
  different, separately-retrained checkpoint and is incompatible with the
  official weights.
- **Single metric**: only the continuous `effort-real-score`. No threshold
  pass-rate this round (deferred — see Out of scope).
- **Statistic**: flat average over all generated images. Stage 3 aggregation
  stays skipped (same as DiffDoctor).

## Integration approach (chosen)

**Lightweight self-contained scorer module + in-place dispatch branch in
`score-images.py`**, mirroring the DiffDoctor integration (and the existing
`sd-safety-checker` / `shieldgemma` ResponsibleAI scorers).

The scorer copies the official Effort model definition (the `EffortDetector`,
`SVDResidualLinear`, `apply_svd_residual_to_self_attn`,
`replace_with_svd_residual` code) into a single file so it does NOT import the
DeepfakeBench framework.

Rejected alternatives:
- Importing the full DeepfakeBench framework (`detectors`/`networks`/`loss`/
  `trainer`/`utils.registry`). Its `install.sh` pins `torch==1.12`,
  `timm==0.6.12`, openai-CLIP — incompatible with the repo's `torch 2.6` /
  `transformers 4.40` and the `visualquality` env.
- Registering in `flow_grpo/rewards.py` `multi_score` (its contract differs and
  it depends on the registry; out of scope for an eval-only metric).

## Components

### (a) `evaluation/benchmarks/Effort/effort_scorer.py` (new)

Copied verbatim from the official model definition (so checkpoint key names match
exactly under `strict=False`):
- `SVDResidualLinear`, `apply_svd_residual_to_self_attn`,
  `replace_with_svd_residual` — unchanged from
  `flow_grpo/Effort-AIGI-Detection/DeepfakeBench/training/detectors/effort_detector.py`.
- `EffortDetector`: `build_backbone` loads
  `CLIPModel.from_pretrained(BACKBONE).vision_model`, applies SVD residual with
  `r = 1024 - 1`; `features` = `backbone(x)["pooler_output"]`; `head =
  nn.Linear(1024, 2)`.

Module-level constants / env overrides:
- `DEFAULT_CKPT = "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/effort_clip_L14_trainOn_sdv14.pth"`,
  overridable via env `EFFORT_CKPT`.
- `BACKBONE = "openai/clip-vit-large-patch14"` (fetched via the `HF_ENDPOINT`
  mirror already exported in the run scripts), overridable via env `EFFORT_CLIP`.

Public API:

```python
def score_images(images, device="cuda", batch_size=8) -> list[dict]:
    # one dict per PIL image: {"effort-real-score": float in [0, 1]}
```

- **Lazy-load + cache** the model once per process in module globals (mirrors
  DiffDoctor's `_load`): build `EffortDetector`, then load the official
  checkpoint with the official recipe —
  `ckpt = torch.load(path, map_location="cpu")`,
  `state = ckpt.get("state_dict", ckpt)`,
  `state = {k.replace("module.", ""): v for k, v in state.items()}`,
  `model.load_state_dict(state, strict=False)`, `.to(device).eval()`.
- **Preprocessing replicates the official demo exactly**: per image
  `convert("RGB")` → `transforms.Resize((224, 224))` → `transforms.ToTensor()`
  → `transforms.Normalize(CLIP_MEAN, CLIP_STD)`, then `torch.stack` into a batch.
  `CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]`,
  `CLIP_STD = [0.26862954, 0.26130258, 0.27577711]`.
- **Forward**: `feat = backbone(batch)["pooler_output"]`; `logits = head(feat)`;
  `fake_prob = torch.softmax(logits, dim=1)[:, 1]`. Batches over `batch_size`.
- **Score aggregation is a separate pure function** so it can be unit-tested on
  CPU without the checkpoint:

```python
def probs_to_scores(fake_probs) -> list[dict]:
    # fake_probs: 1-D tensor in [0, 1]
    return [{"effort-real-score": float(1.0 - p.item())} for p in fake_probs]
```

- Missing checkpoint → `FileNotFoundError` naming the path and the `EFFORT_CKPT`
  override.
- Assert `fake_prob` is in `[0, 1]` before aggregation.

### (b) `evaluation/metrics/score-images.py` (3 small edits)

- Add `"effort"` to `AVAILABLE_METRICS`.
- Add `METRIC_OUTPUT_KEYS["effort"] = ("effort-real-score",)` so
  `_has_metric_score` / resume works.
- Add a `main()` dispatch branch `_score_effort_in_place(todo)` modeled on
  `_score_diffdoctor_in_place`: iterate in batches (env `EFFORT_BATCH_SIZE`,
  default 8), open RGB images via `_open_rgb_image`, call `score_images`,
  `row["scores"].update(...)`, raise on length mismatch, then
  `torch.cuda.empty_cache()`.

### (c) `evaluation/run-bestofn.sh` (2 small edits)

- `aigi-detector) metric_list=(diffdoctor effort) ;;` (append `effort`).
- Add `[effort]=visualquality` to the `metric_env` map.
- Stage 3 `aigi-detector` branch stays a skip (already points at
  `average_scores.json`); update its comment to mention both detectors.

### (d) `evaluation/run-bestofn-batch.sh`

Update any aigi-detector metric_list / comment references to stay consistent
with (c) if present.

## Data flow

```
run-bestofn.sh (aigi-detector)
  Stage1: generate → evaluation_results.jsonl + images/
  Stage2: conda activate visualquality
          score-images.py --metrics diffdoctor   (existing)
          score-images.py --metrics effort
            → _score_effort_in_place(todo)
                per batch: feat = clip_vision(norm(resize224(img)))["pooler_output"]
                           logits = head(feat)
                           fake_prob = softmax(logits)[:, 1]
                           effort-real-score = 1 - fake_prob
                row["scores"].update(...)
            → main() averages all numeric scores → average_scores.json
  Stage3: skip (read average_scores.json)
```

Final aggregate in `average_scores.json`:
- `effort-real-score` = mean over all images of `1 - fake_prob`.

## Error handling

- Missing checkpoint → clear `FileNotFoundError` naming the path + `EFFORT_CKPT`.
- Follow the existing scorer convention: errors propagate; `main()`'s end-of-run
  jsonl rewrite never runs on failure, so a re-run resumes from on-disk scores.
- Assert each `fake_prob` is in `[0, 1]` and the output key is present.

## Testing

- **CPU unit test** (no GPU/checkpoint): exercise `probs_to_scores` with
  synthetic fake-prob tensors:
  - `fake_prob = 0.0` → `effort-real-score = 1.0`
  - `fake_prob = 1.0` → `effort-real-score = 0.0`
  - `fake_prob = 0.3` → `effort-real-score ≈ 0.7`
  - batch of mixed values → element-wise `1 - p`.
- **Server smoke test** (with checkpoint, `visualquality` env): run
  `score-images.py --metrics effort` on an existing aigi-detector output dir;
  verify the key exists, values are in `[0, 1]`, and that visibly-realistic
  images score noticeably higher than obviously-AI / artifact-heavy ones (sanity
  check that `strict=False` actually loaded the trained residual + head, not a
  randomly-initialized head).

## Prerequisites

- The official GenImage (sdv1.4) checkpoint is already present on the server at
  `/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/effort_clip_L14_trainOn_sdv14.pth`,
  so no download is needed there.
- The `visualquality` conda env must have `transformers` (CLIPModel),
  `torchvision`, and `torch` (2.x is fine — the model is plain `nn.Module` +
  `torch.linalg.svd`, no DeepfakeBench / torch-1.12 pins).
- `openai/clip-vit-large-patch14` config + weights are fetched from Hugging Face
  via the `HF_ENDPOINT` mirror (already exported in the run scripts);
  overridable to a local path via `EFFORT_CLIP`.

## Out of scope

- The face-deepfake (FaceForensics++) checkpoint and dlib face alignment — we
  detect natural AIGI, so no face cropping.
- A threshold pass-rate metric (`effort-real-rate@(fake<0.5)`) — deferred; the
  pure-function design makes it a one-line add later if wanted.
- The simplified "RealGen" retrained checkpoint + its no-normalization / `[:, 0]`
  pipeline (selectable later via `EFFORT_CKPT` only if paired with its own code).
- Using Effort as a training reward.
- Best-of-N max-over-N aggregation curve.
