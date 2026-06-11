# 6 GPU for training + 1 dedicated GPU for DINOv3 (config.dinov3_device = "cuda:6").
# GPU 6 must be visible but is NOT a training rank (num_processes=6 -> ranks use cuda:0-5).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export HF_ENDPOINT=https://hf-mirror.com

# pickscore_sd3_gardo (DINOv3 weights are gated on HF; make sure `huggingface-cli login` has been done)
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=6 --main_process_port 29501 scripts/train_sd3_gardo.py --config config/gardo.py:pickscore_sd3_gardo
