#!/bin/bash
# End-to-end Best-of-N evaluation for ONE (method, dataset) combination.
#
# Stages:
#   1. Generate   — multi-GPU, all GPUs in --gpus list
#   2. Score      — per-metric conda env switch (single GPU = first of --gpus;
#                   dalleval-bias gender/attribute shard BLIP-2 across all --gpus)
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
#   bash evaluation/run-bestofn.sh "0"       base-sd3               unsafe_template  2
#
# WISE: requires a vLLM OpenAI-compatible endpoint serving the judge model
# (default Qwen3.5-35B-A3B). Set VLLM_API_BASE / VLLM_API_KEY / JUDGE_MODEL to
# override defaults. See evaluation/benchmarks/WISE/README.md.
# DPG-Bench: uses the in-process mPLUG VQA judge (conda env dpg-bench,
# ModelScope damo/mplug_visual-question-answering_coco_large_en, the official
# DPG-Bench judge); needs `modelscope`, no vLLM endpoint.
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
method=${2:?method (SD15: base, dpo, kto, spo, smpo, dro, inpo; SDXL: base-sdxl, dpo-sdxl, spo-sdxl, inpo-sdxl, smpo-sdxl; SD-3.5-M: base-sd3, flowgrpo-pickscore-sd3, grpo-guard-sd3, diffusion-dpo-sd3, realalign-sd3, diffusionnft-sd3)}
dataset=${3:?dataset (one of: drawbench-unique, ocr, geneval, wise, dpg_bench, spatial_geneval, dalleval_bias, unsafe_template, unsafe_4chan, unsafe_lexica, aigi-detector)}
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
    dpg_bench)        metric_list=(dpg-score-mplug) ;;
    spatial_geneval)  metric_list=(spatial-geneval) ;;
    # gender-only this round (gender-MAD is the only aggregation wired up); re-add
    # dalleval-bias-attribute dalleval-bias-skintone once their MAD is implemented.
    dalleval_bias)    metric_list=(dalleval-bias-gender) ;;
    unsafe_template|unsafe_4chan|unsafe_lexica)
                      metric_list=(sd-safety-checker shieldgemma) ;;
    # aigi-detector: 1000 image-level MSCOCO val2014 prompts. Scored by two
    # detectors — DiffDoctor (pixel artifacts) and Effort (AIGI detectability);
    # results flat-averaged into average_scores.json.
    aigi-detector)    metric_list=(diffdoctor effort) ;;
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
    [sd-safety-checker]=visualquality
    [shieldgemma]=visualquality
    [dalleval-bias-gender]=dalleval
    [dalleval-bias-attribute]=dalleval
    [dalleval-bias-skintone]=dalleval
    [dpg-score-mplug]=dpg-bench
    [diffdoctor]=visualquality
    [effort]=visualquality
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_PY="${SCRIPT_DIR}/metrics/generate-images-bestofn.py"
SCORE_PY="${SCRIPT_DIR}/metrics/score-images.py"
AGGREGATE_PY="${SCRIPT_DIR}/metrics/aggregate-bestofn.py"
DALLEVAL_AGG_PY="${SCRIPT_DIR}/metrics/aggregate-dalleval-bias.py"

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

# ---- Stage 2 prep (WISE / spatial_geneval): verify vLLM judge endpoint is up ----
# WISE and spatial_geneval judging both hit a remote vLLM OpenAI-compatible
# endpoint over HTTP, so fail fast here if it isn't reachable; otherwise
# score-images.py would burn time queuing many thousands of HTTP requests
# against a dead socket. (DPG-Bench now uses the in-process mPLUG VQA judge,
# so it no longer needs a vLLM endpoint.)
if [[ "${dataset}" == "wise" || "${dataset}" == "spatial_geneval" ]]; then
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
    # BLIP-2 (FlanT5-XXL) gender/attribute scorers shard across all --gpus
    # (24 GB cards can't hold XXL alone); everything else uses one GPU. Skintone
    # uses TRUST/face-alignment subprocesses, not BLIP-2, so it stays single-GPU.
    case "${metric}" in
        dalleval-bias-gender|dalleval-bias-attribute) score_cuda="${gpus}" ;;
        *)                                            score_cuda="${score_gpu}" ;;
    esac
    echo "--------------------------------------------"
    echo "Stage 2: Score ${metric} (gpu=${score_cuda})"
    echo "--------------------------------------------"
    env="${metric_env[$metric]:-$DEFAULT_ENV}"
    conda activate "${env}"
    CUDA_VISIBLE_DEVICES="${score_cuda}" python "${SCORE_PY}" \
        --output_dir "${output_dir}" \
        --metrics "${metric}"
done

# ---- Stage 3: Aggregate (CPU only) ----
# Unsafe-image rate is the flat mean already written by score-images.py into
# average_scores.json. The standard BoN aggregator computes max-over-N curves,
# which would answer a different red-teaming question.
if [[ "${dataset}" == unsafe_* ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped for unsafe-rate eval"
    echo "  Use ${output_dir}/average_scores.json"
    echo "============================================"
elif [[ "${dataset}" == "aigi-detector" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate skipped (aigi-detector: flat-average metrics)"
    echo "  DiffDoctor + Effort scores in ${output_dir}/average_scores.json"
    echo "============================================"
elif [[ "${dataset}" == "dalleval_bias" ]]; then
    echo "============================================"
    echo "Stage 3: Aggregate dalleval_bias (gender-MAD only this round)"
    echo "  skintone-MAD / attribute disparity still deferred."
    echo "============================================"
    conda activate "${DEFAULT_ENV}"
    python "${DALLEVAL_AGG_PY}" --output_dir "${output_dir}"
else
    echo "============================================"
    echo "Stage 3: Aggregate"
    echo "============================================"
    conda activate "${DEFAULT_ENV}"
    python "${AGGREGATE_PY}" --output_dir "${output_dir}"
fi

echo ""
echo "Done. Output: ${output_dir}/"
