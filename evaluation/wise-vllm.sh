source /data3/chenweiyan/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=False

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=1 \
vllm serve /data_center/data2/dataset/chenwy/21164-data/model-ckpt/Qwen3.5-35B-A3B \
    --served-model-name Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt.image 1 \
    --limit-mm-per-prompt.video 0 \
    --mm-processor-cache-gb 0 \
    --enforce-eager