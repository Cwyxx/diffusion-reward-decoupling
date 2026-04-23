#!/bin/bash
# Compute per-step reward trends for every prompt_* dir under input_dir,
# one reward at a time so each reward can run in its own conda env.
# The Python script merges per-prompt JSONs across invocations.
#
# Usage:
#   bash notebook/run-reward-evolution.sh <cuda_device> [input_dir] [dataset] [output_dir]

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

cuda_device=$1
input_dir=${2:-"/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/the-evoluation-of-generated-images"}
dataset=${3:-HPDv3}
output_dir=${4:-"/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/reward-evolution"}

if [[ -z "$cuda_device" ]]; then
    echo "Usage: $0 <cuda_device> [input_dir] [dataset] [output_dir]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${cuda_device}

# Rewards analyzed independently (same set as notebook default).
reward_list=(pickscore imagereward aesthetic omniaid_remote hpsv3 deqa visualquality_r1)

# Per-reward conda env (matches evaluation/run-eval.sh). Unlisted rewards
# fall back to $DEFAULT_ENV.
DEFAULT_ENV=alignprop
declare -A reward_env=(
    [hpsv3]=hpsv3
    [deqa]=internvl
    [visualquality_r1]=visualquality
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/reward-evolution-of-generated-images.py"

for reward in "${reward_list[@]}"; do
    echo "============================================"
    echo "Reward: ${reward}"
    echo "============================================"
    env="${reward_env[$reward]:-$DEFAULT_ENV}"
    conda activate "${env}"
    python "${PY}" \
        --input_dir "${input_dir}" \
        --dataset "${dataset}" \
        --rewards "${reward}" \
        --output_dir "${output_dir}"
done

echo "All rewards done. Merged results in ${output_dir}"
