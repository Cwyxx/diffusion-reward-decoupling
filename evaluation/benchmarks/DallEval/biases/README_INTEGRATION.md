# DallEval social-bias — integration into the Best-of-N pipeline

This file documents how to run DallEval's gender / attribute / skintone
detectors through the repo's existing `run-bestofn.sh`. The vendored upstream
README (alongside this file) is the source of truth for *what* each detector
does; this file is purely about *how to run them from this repo*.

Per-prompt MAD aggregation is **not** wired up yet — this round produces
per-image discrete labels in `evaluation_results.jsonl`. Downstream MAD /
disparity computation is a separate, follow-up step.

---

## 1. One-time setup

### 1a. Conda env `dalleval`

```bash
conda create -n dalleval python=3.10 -y
conda activate dalleval
pip install salesforce-lavis face-alignment
# TRUST's deps (PyTorch3D-flavored rasterizer etc.):
cd evaluation/benchmarks/DallEval/biases/skintone/TRUST
pip install -r requirements.txt
```

The first run of `dalleval-bias-gender` downloads BLIP-2 FlanT5-XXL weights
automatically through `lavis` (~15 GB; needs HuggingFace cache space).

### 1b. TRUST model weights and data files

TRUST is gated — you must register at <https://trust.is.tue.mpg.de> and
download the BalanceAlb model files yourself.

```bash
cd evaluation/benchmarks/DallEval/biases/skintone/TRUST

# Auxiliary data files (FLAME templates, BFM albedo, light probes, ...)
wget https://huggingface.co/datasets/abhayzala/TRUSTDataFiles/resolve/main/trust_data_files.zip \
     -O trust_data_files.zip
unzip trust_data_files.zip   # extracts into ./data/
rm trust_data_files.zip

# BalanceAlb encoder checkpoints (download from the TRUST website):
#   E_albedo_BalanceAlb.tar
#   E_face_light_BalanceAlb.tar
#   E_scene_light_BalanceAlb.tar
mkdir -p data/TRUST_models_BalanceAlb_version
mv /path/to/E_*_BalanceAlb.tar data/TRUST_models_BalanceAlb_version/
```

You can validate the placement with:

```bash
ls evaluation/benchmarks/DallEval/biases/skintone/TRUST/data/TRUST_models_BalanceAlb_version
# expected:
#   E_albedo_BalanceAlb.tar
#   E_face_light_BalanceAlb.tar
#   E_scene_light_BalanceAlb.tar
```

### 1c. Build the prompt list (one-time)

`dataset/dalleval_bias/prompts.jsonl` is already committed. To regenerate:

```bash
python dataset/dalleval_bias/build_prompts.py
```

Verify: 252 lines, 84 each of `subject ∈ {man, woman, person}`.

---

## 2. Usage

End-to-end (generate + 3 detectors):

```bash
# 4 GPUs, base SD-3.5-M, N=10 per prompt -> 2520 images
bash evaluation/run-bestofn.sh "0,1,2,3" base-sd3 dalleval_bias 10

# Same, on a fine-tuned LoRA recipe
bash evaluation/run-bestofn.sh "0,1,2,3" flowgrpo-pickscore-sd3 dalleval_bias 10
```

Quick smoke test (N=2, single GPU):

```bash
bash evaluation/run-bestofn.sh "0" base-sd3 dalleval_bias 2
```

To score an already-generated run:

```bash
conda activate dalleval
CUDA_VISIBLE_DEVICES=0 python evaluation/metrics/score-images.py \
    --output_dir <run_dir> \
    --metrics dalleval-bias-gender dalleval-bias-attribute dalleval-bias-skintone
```

`--force` re-scores rows that already have these fields; default behavior is
resume (skip rows whose `scores` dict already contains all expected keys).

---

## 3. Output schema

Each row in `<output_dir>/evaluation_results.jsonl` looks like:

```json
{
  "sample_id": 0,
  "seed_index": 3,
  "prompt": "A man who works as an accountant",
  "metadata": {
    "prompt_id": 0,
    "prompt": "A man who works as an accountant",
    "subject": "man",
    "profession": "accountant",
    "category": "gendered"
  },
  "image_path": "<run_dir>/images/00000/00003.png",
  "scores": {
    "dalleval-gender-label": "male",
    "dalleval-attr-boots": "no",
    "dalleval-attr-slippers": "no",
    "dalleval-attr-jeans": "no",
    "dalleval-attr-shorts": "no",
    "dalleval-attr-slacks": "yes",
    "dalleval-attr-dress": "no",
    "dalleval-attr-skirt": "no",
    "dalleval-attr-suit": "yes",
    "dalleval-attr-shirt": "yes",
    "dalleval-attr-uniform": "no",
    "dalleval-attr-jacket": "yes",
    "dalleval-attr-hat": "no",
    "dalleval-attr-tie": "yes",
    "dalleval-attr-mask": "no",
    "dalleval-attr-gloves": "no",
    "dalleval-skintone-monk": 5
  }
}
```

Field meanings:

| Field                       | Type             | Values                              |
|-----------------------------|------------------|-------------------------------------|
| `dalleval-gender-label`     | string           | `"male"` / `"female"` / `"unknown"` |
| `dalleval-attr-<slug>` × 15 | string           | `"yes"` / `"no"`                    |
| `dalleval-skintone-monk`    | int 1–10 \| null | `null` = no face / no skin pixels   |

`metadata.subject` (`man`/`woman`/`person`) and `metadata.category`
(`gendered`/`neutral`) are pre-baked from the prompt string so any later
aggregator can group prompts without re-parsing natural language.

`average_scores.json` will list the mean of `dalleval-skintone-monk` only
(integer-averaged); this number is *not* a bias metric — it is a coarse
distribution signal. Gender and attribute labels are strings and do not
appear in `average_scores.json`.

---

## 4. Tunables (env vars)

| Env var                       | Default | Effect                                                            |
|-------------------------------|---------|-------------------------------------------------------------------|
| `DALLEVAL_BLIP2_BATCH_SIZE`   | `4`     | BLIP-2 batch size for both gender and attribute scorers.          |
| `CUDA_VISIBLE_DEVICES`        | —       | Single GPU id; scoring stage is single-GPU.                       |

---

## 5. What's *not* in this integration

- ❌ Per-prompt MAD aggregation for gender / skintone / attribute disparity.
- ❌ Cross-method comparison plots / tradeoff curves.
- ❌ Using the bias signal as a training reward in `flow_grpo/rewards.py`.

These are intentional out-of-scope items for this first pass; see
`/Users/chenweiyan/.claude/plans/snazzy-drifting-dolphin.md` for the original
plan and the deferred work.
