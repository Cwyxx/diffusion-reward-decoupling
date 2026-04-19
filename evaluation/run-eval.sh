#!/bin/bash
# End-to-end evaluation: generate images from a LoRA checkpoint, then score
# with a set of metrics. Each metric runs in its own conda env (dependency
# conflicts between reward libraries).
#
# Usage:
#   bash evaluation/run-eval.sh <cuda_device> <method> <ckpt> <dataset>
# Example:
#   bash evaluation/run-eval.sh 0 sd-3-5-medium 5000 drawbench-unique

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# ---- Positional args ----
cuda_device=$1
method=$2
ckpt=$3
dataset=$4

rl_framework=flow-grpo

if [[ -z "$cuda_device" || -z "$method" || -z "$ckpt" || -z "$dataset" ]]; then
    echo "Usage: $0 <cuda_device> <method> <ckpt> <dataset>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${cuda_device}

# ---- Config ----
base_root="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling"
base_ckpt_dir="${base_root}/${rl_framework}/sd-3-5-medium/model-ckpt"
ckpt_dir="${base_ckpt_dir}/${method}/checkpoints/checkpoint-${ckpt}"

seed_list=(42 123 456 789 1000)
metric_list=(pickscore imagereward aesthetic hpsv3 deqa visualquality_r1)

# Conda env per metric. Metrics not listed fall back to $DEFAULT_ENV.
DEFAULT_ENV=alignprop
declare -A metric_env=(
    [hpsv3]=hpsv3
    [deqa]=visualquality
    [visualquality_r1]=visualquality
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_PY="${SCRIPT_DIR}/metrics/generate-images.py"
SCORE_PY="${SCRIPT_DIR}/metrics/score-images.py"

# ---- Loop ----
for seed in "${seed_list[@]}"; do
    echo "============================================"
    echo "Seed: ${seed}"
    echo "============================================"

    image_dir="${base_root}/${rl_framework}/sd-3-5-medium/generate_images_seed_${seed}/${dataset}/${method}/ckpt-${ckpt}"

    # ---- Generate ----
    conda activate "${DEFAULT_ENV}"
    python "${GENERATE_PY}" \
        --seed ${seed} \
        --checkpoint_path "${ckpt_dir}" \
        --dataset "${dataset}" \
        --output_dir "${image_dir}" \
        --save_images

    # ---- Score with each metric ----
    for metric in "${metric_list[@]}"; do
        echo "--------------------------------------------"
        echo "Metric: ${metric} (seed=${seed})"
        echo "--------------------------------------------"
        env="${metric_env[$metric]:-$DEFAULT_ENV}"
        conda activate "${env}"
        python "${SCORE_PY}" \
            --output_dir "${image_dir}" \
            --metrics "${metric}"
    done
done
