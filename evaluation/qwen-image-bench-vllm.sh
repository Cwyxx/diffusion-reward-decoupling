#!/usr/bin/env bash
# Launch the 27B Qwen-Image-Bench Q-Judger as an OpenAI-compatible vLLM server.
# The scoring side (score-images.py / run-bestofn.sh, dataset=qwen-image-bench) is
# only an HTTP client and talks to this server via QIB_VLLM_URL (default
# http://localhost:8000/v1) and QIB_VLLM_MODEL (default Qwen-Image-Bench).
# Start this FIRST, wait for "Uvicorn running on http://0.0.0.0:8000", then score.
#
# Thinking mode is ON server-side via --default-chat-template-kwargs below, to
# match the official ms-swift judge (--enable_thinking true) for leaderboard
# parity. The scoring client does not override this per-request. Thinking ON
# means longer decodes (lower throughput); flip enable_thinking to false for a
# faster JSON-only run when leaderboard parity is not required.
#
# Throughput: --max-num-seqs 64 + --gpu-memory-utilization 0.90 let the server
# batch many in-flight requests; set the client's QIB_VLLM_CONCURRENCY to match
# (e.g. 64) so it keeps those slots full. If you hit OOM at startup, lower
# --gpu-memory-utilization or --max-num-seqs. Removing --enforce-eager enables
# CUDA graphs for a further decode speedup but needs more memory (may OOM here).
# If you have 8 GPUs free, 2 replicas (--data-parallel-size 2, each TP=4) ~2x.
# conda's init scripts reference unset vars (e.g. PS1) which trip `set -u`, so
# source + activate first, then enable nounset.
set -eo pipefail

source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vllm
set -u

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
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --limit-mm-per-prompt.image 1 \
    --limit-mm-per-prompt.video 0 \
    --mm-processor-cache-gb 0 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}'
