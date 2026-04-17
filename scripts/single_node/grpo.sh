# 1 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 6 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export HF_ENDPOINT=https://hf-mirror.com

# pickscore_sd3
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=6 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:pickscore_sd3

# pickscore_sd3_decoupled_reward
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=6 --main_process_port 29501 scripts/train_sd3_decoupled_reward.py --config config/grpo.py:sd3_decoupled_reward
