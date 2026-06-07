#!/usr/bin/env bash
# Launch the 27B Qwen-Image-Bench Q-Judger as an OpenAI-compatible vLLM server.
# The scoring side (score-images.py / run-bestofn.sh, dataset=qwen-image-bench) is
# only an HTTP client and talks to this server via QIB_VLLM_URL (default
# http://localhost:8000/v1) and QIB_VLLM_MODEL (default Qwen-Image-Bench).
# Start this FIRST, wait for "Uvicorn running on http://0.0.0.0:8000", then score.
#
# Thinking mode follows the chat-template default (ON). To turn it off, add:
#   --default-chat-template-kwargs '{"enable_thinking": false}'
set -euo pipefail

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMP_NUM_THREADS=1 \
vllm serve /data3/chenweiyan/.cache/modelscope/hub/models/Qwen/Qwen-Image-Bench \
    --served-model-name Qwen-Image-Bench \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt.image 1 \
    --limit-mm-per-prompt.video 0 \
    --mm-processor-cache-gb 0 \
    --enforce-eager \
    --reasoning-parser qwen3
