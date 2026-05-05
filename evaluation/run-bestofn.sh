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
#   bash evaluation/run-bestofn.sh "0,1,2,3" dpo                    drawbench-unique 32
#   bash evaluation/run-bestofn.sh "0"       base                   ocr              32
#   bash evaluation/run-bestofn.sh "0,1,2,3" dpo-sdxl               geneval          32
#   bash evaluation/run-bestofn.sh "0,1,2,3" flowgrpo-pickscore-sd3 geneval          32
#   bash evaluation/run-bestofn.sh "0,1,2,3" base-sd3               wise             32
#
# WISE-specific: requires a vLLM OpenAI-compatible endpoint serving the
# judge model (default Qwen3.5-35B-A3B). Set VLLM_API_BASE / VLLM_API_KEY
# / JUDGE_MODEL to override defaults. See evaluation/benchmarks/WISE/README.md.
#
# Method suffix selects the model family:
#   *-sdxl -> SDXL (1024px, 50 steps, CFG 7.5, fp16)
#   *-sd3  -> SD-3.5-M (512px, 40 steps, CFG 4.5, fp16)
#   else   -> SD-v1.5 (512px, 50 steps, CFG 7.5, fp32)

set -eo pipefail

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

# ---- Positional args ----
gpus=${1:?gpus (comma-separated, e.g. 0,1,2,3)}
method=${2:?method (SD15: base, dpo, kto, spo, smpo, dro, inpo; SDXL: base-sdxl, dpo-sdxl, spo-sdxl, inpo-sdxl, smpo-sdxl; SD-3.5-M: base-sd3, flowgrpo-pickscore-sd3, grpo-guard-sd3, diffusion-dpo-sd3, realalign-sd3)}
dataset=${3:?dataset (one of: drawbench-unique, ocr, geneval, wise)}
n_max=${4:?n_max (e.g. 32)}

# ---- Family-aware defaults (derived from method suffix) ----
if [[ "${method}" == *-sdxl ]]; then
    family="sdxl"
    resolution=1024
    num_inference_steps=50
    guidance_scale=7.5
elif [[ "${method}" == *-sd3 ]]; then
    family="sd-3.5-m"
    resolution=512
    num_inference_steps=40
    guidance_scale=4.5
else
    family="sd-v1-5"
    resolution=512
    num_inference_steps=50
    guidance_scale=7.5
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
    wise)             metric_list=(wise) ;;
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
    --resolution "${resolution}" \
    --num_inference_steps "${num_inference_steps}" \
    --guidance_scale "${guidance_scale}"

# ---- Stage 2 prep (WISE only): verify vLLM judge endpoint is up ----
# WISE judging hits a remote vLLM OpenAI-compatible endpoint over HTTP, so
# fail fast here if it isn't reachable; otherwise score-images.py would
# burn time queuing 32K HTTP requests against a dead socket.
if [[ "${dataset}" == "wise" ]]; then
    : "${VLLM_API_BASE:=http://127.0.0.1:8000/v1}"
    : "${VLLM_API_KEY:=EMPTY}"
    : "${JUDGE_MODEL:=Qwen3.5-35B-A3B}"
    export VLLM_API_BASE VLLM_API_KEY JUDGE_MODEL
    echo "Probing vLLM judge endpoint at ${VLLM_API_BASE}..."
    if ! curl -sSf -m 10 -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_API_BASE}/models" >/dev/null; then
        echo "ERROR: vLLM endpoint ${VLLM_API_BASE}/models is not reachable." >&2
        echo "  Start vLLM first, e.g.:" >&2
        echo "    vllm serve /path/to/${JUDGE_MODEL} --served-model-name ${JUDGE_MODEL} --host 0.0.0.0 --port 8000" >&2
        exit 1
    fi
fi

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
