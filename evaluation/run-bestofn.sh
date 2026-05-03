#!/bin/bash
# End-to-end Best-of-N evaluation for ONE (method, dataset) combination.
#
# Stages:
#   1. Generate   — multi-GPU, all GPUs in --gpus list
#   2. Score      — per-metric conda env switch (single GPU = first of --gpus)
#   3. Aggregate  — CPU only (numpy + matplotlib)
#
# Usage:
#   bash evaluation/run-bestofn.sh <gpus> <method> <dataset> <n_max>
# Example:
#   bash evaluation/run-bestofn.sh "0,1,2,3" dpo       drawbench-unique 32
#   bash evaluation/run-bestofn.sh "0"       base      ocr              32
#   bash evaluation/run-bestofn.sh "0,1,2,3" dpo-sdxl  geneval          32
#
# Method suffix "-sdxl" switches to SDXL: family="sdxl" subdir, 1024px.

set -eo pipefail

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# ---- Positional args ----
gpus=${1:?gpus (comma-separated, e.g. 0,1,2,3)}
method=${2:?method (SD15: base, dpo, kto, spo, smpo, dro, inpo; SDXL: base-sdxl, dpo-sdxl, spo-sdxl, inpo-sdxl, smpo-sdxl)}
dataset=${3:?dataset (one of: drawbench-unique, ocr, geneval)}
n_max=${4:?n_max (e.g. 32)}

# ---- Family-aware defaults (derived from method suffix) ----
if [[ "${method}" == *-sdxl ]]; then
    family="sdxl"
    resolution=1024
else
    family="sd-v1-5"
    resolution=512
fi

# ---- Config ----
base_root="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling"
output_dir="${base_root}/bestofn-eval/${family}/${method}/${dataset}"
mkdir -p "${output_dir}"

# Per-dataset metric set.
case "${dataset}" in
    drawbench-unique) metric_list=(pickscore hpsv3 deqa aesthetic) ;;
    ocr)              metric_list=(ocr) ;;
    geneval)          metric_list=(geneval) ;;
    *) echo "Unknown dataset: ${dataset}" >&2; exit 1 ;;
esac

# Conda env per metric. Metrics not listed fall back to DEFAULT_ENV.
DEFAULT_ENV=alignprop
declare -A metric_env=(
    [hpsv3]=hpsv3
    [deqa]=internvl
    [visualquality_r1]=visualquality
    [ocr]=visualquality
    [geneval]=internvl
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_PY="${SCRIPT_DIR}/metrics/generate-images-bestofn.py"
SCORE_PY="${SCRIPT_DIR}/metrics/score-images.py"
AGGREGATE_PY="${SCRIPT_DIR}/metrics/aggregate-bestofn.py"

# First GPU from the comma list, used for scoring stage (single GPU is enough).
score_gpu="${gpus%%,*}"

# ---- Stage 1: Generate (multi-GPU) ----
echo "============================================"
echo "Stage 1: Generate"
echo "  method=${method} dataset=${dataset} n_max=${n_max} gpus=${gpus}"
echo "============================================"
conda activate "${DEFAULT_ENV}"
python "${GENERATE_PY}" \
    --gpus "${gpus}" \
    --method "${method}" \
    --dataset "${dataset}" \
    --output_dir "${output_dir}" \
    --n_max "${n_max}" \
    --resolution "${resolution}"

# ---- Stage 2: Score (per-metric conda env) ----
for metric in "${metric_list[@]}"; do
    echo "--------------------------------------------"
    echo "Stage 2: Score ${metric} (gpu=${score_gpu})"
    echo "--------------------------------------------"
    env="${metric_env[$metric]:-$DEFAULT_ENV}"
    conda activate "${env}"
    CUDA_VISIBLE_DEVICES="${score_gpu}" python "${SCORE_PY}" \
        --output_dir "${output_dir}" \
        --metrics "${metric}"
done

# ---- Stage 3: Aggregate (CPU only) ----
echo "============================================"
echo "Stage 3: Aggregate"
echo "============================================"
conda activate "${DEFAULT_ENV}"
python "${AGGREGATE_PY}" --output_dir "${output_dir}"

echo ""
echo "Done. Output: ${output_dir}/bestofn/"
