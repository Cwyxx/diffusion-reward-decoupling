# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Flow-GRPO** is a reinforcement learning framework for training diffusion models (text-to-image/video) via online RL. It implements GRPO (Group-based Reward Policy Optimization) and variants applied to models like SD3.5-M, FLUX.1-dev, Qwen-Image, Bagel-7B, and Wan2.1.

Key algorithms:
- **GRPO**: PPO-style clipping with per-prompt advantage normalization
- **Flow-GRPO-Fast**: Accelerated variant training on only 1-2 denoising steps
- **GRPO-Guard**: Mitigates over-optimization via RatioNorm and gradient reweighting
- **DPO/OnlineDPO** and **SFT/OnlineSFT**: Alternative training paradigms

## Installation

```bash
pip install -e .
```

Key dependencies: `torch==2.6.0`, `diffusers==0.33.1`, `transformers==4.40.0`, `accelerate==1.4.0`, `peft==0.10.0`, `deepspeed==0.16.4`, `ml_collections`, `wandb`.

## Running Training

All training uses `accelerate launch` with configs in `scripts/accelerate_configs/`.

**Single GPU:**
```bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
```

**Multi-GPU (single node):**
```bash
bash scripts/single_node/grpo.sh        # GRPO on SD3.5
bash scripts/single_node/dpo.sh         # DPO
bash scripts/single_node/sft.sh         # SFT
```

**Multi-node (e.g., 4 machines × 8 GPUs):**
```bash
bash scripts/multi_node/sd3.sh 0        # Run on master node (node_rank=0)
bash scripts/multi_node/sd3.sh 1        # Worker node 1
# ... repeat for each node
```

**FLUX model:**
```bash
bash scripts/single_node/grpo_flux.sh
bash scripts/multi_node/flux.sh 0
```

The `--config` flag uses the format `config/<file>.py:<function_name>` (ml_collections style).

## Architecture

### Configuration System (`config/`)
- `base.py`: Defines all hyperparameters with defaults. Key groups: `sample` (sampling), `train` (optimization), `prompt`, `reward`.
- `grpo.py`, `dpo.py`, `sft.py`, `grpo_guard.py`: Task-specific configs as Python functions returning a modified `base` config.

**Critical training equation** — the group number must be consistent across GPUs:
```
group_number = (train_batch_size × num_gpus / num_image_per_prompt) × num_batches_per_epoch
# Typical target: 48 groups
```

Key hyperparameters to tune:
- `config.train.beta` — KL coefficient (higher = stronger KL regularization, default 0.04)
- `config.train.clip_range` — PPO clip range (default 1e-4; for Fast variant use 2e-4 or higher)
- `config.sample.num_image_per_prompt` — group size for per-prompt stats (default 24)
- `config.sample.guidance_scale` — CFG scale
- `config.sample.num_steps` — denoising steps for training trajectories

### Core Package (`flow_grpo/`)
- `diffusers_patch/`: Custom pipeline wrappers that compute log-probabilities during generation. One per model (SD3, FLUX, Qwen, Wan2.1, etc.) plus `_fast` variants.
- `rewards.py`: Reward model registry and weighted combination. Supports GenEval, OCR, PickScore, CLIPScore, Aesthetic, ImageReward, QwenVL. Reward models may run as separate servers to avoid dependency conflicts.
- `stat_tracking.py`: `PerPromptStatTracker` maintains per-prompt reward mean/std for advantage normalization: `advantage = (reward - mean) / (std + 1e-4)`.
- `prompts.py`: Prompt generation functions keyed by name (e.g., `"general_ocr"`, `"geneval"`).
- `ema.py`: EMA for stable model updates.
- `fsdp_utils.py`: FSDP (Fully Sharded Data Parallel) utilities.

### Training Scripts (`scripts/`)
Each model has its own training script: `train_sd3.py`, `train_sd3_fast.py`, `train_sd3_GRPO_Guard.py`, `train_sd3_dpo.py`, `train_sd3_sft.py`, `train_flux.py`, `train_flux_fast.py`, `train_qwenimage.py`, `train_bagel.py`, `train_wan2_1.py`.

The scripts share a common training loop pattern:
1. Sample trajectories with the current policy (computing log-probs)
2. Compute rewards via reward functions
3. Compute per-prompt advantages via `PerPromptStatTracker`
4. Update model with PPO-style loss (or DPO/SFT loss)

### Dataset (`dataset/`)
Each task has its own subdirectory (e.g., `geneval/`, `ocr/`, `pickscore/`, `counting_edit/`). `base.py` provides shared dataset configuration.

## Precision Notes
- SD3.5-M and Wan2.1: use **bf16**
- FLUX.1-dev with LoRA: use **fp16** (fp16 works better empirically for FLUX LoRA training)
- Set via `config.mixed_precision`

## Adding a New Model
1. Create a pipeline wrapper in `flow_grpo/diffusers_patch/` that tracks log-probabilities
2. Create a training script in `scripts/` (copy closest existing one)
3. Add configs in `config/grpo.py`
4. Add bash launch scripts in `scripts/single_node/` and `scripts/multi_node/`

Verify on-policy consistency: the pipeline must produce identical outputs given the same random seed when called with the same noise. See README for the verification checklist.

## Logging
Training metrics are logged to Weights & Biases (wandb). Set `config.run_name` and `config.project_name` in your config.
