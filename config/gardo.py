import imp
import os

grpo = imp.load_source("grpo", os.path.join(os.path.dirname(__file__), "grpo.py"))


def pickscore_sd3_gardo():
    config = grpo.pickscore_sd3()

    config.train.algorithm = 'gardo'
    # The advantage is driven by main_reward only; aesthetic/imagereward serve as auxiliary
    # judges for rank-disagreement risk detection (their weights only affect logging).
    config.main_reward = 'pickscore_remote'
    # All rewards run as servers (pickscore 18091, imagereward 18093, aesthetic 18094)
    # so training ranks don't load CLIP-L/BLIP locally.
    config.reward_fn = {
        "pickscore_remote": 0.8,
        "aesthetic_remote": 0.05,
        "imagereward_remote": 0.15,
    }

    # Selective KL against the periodically re-synced "old" adapter.
    config.train.beta = 0.04
    # KL threshold that triggers re-syncing the "old" anchor; tune per task (GARDO used 3e-4 for ocr).
    config.kl_thres = 3e-4
    # Force a re-sync of the "old" anchor after this many epochs even if KL stays below kl_thres.
    config.reset_freq = 15

    # Dedicated GPU for DINOv3 (must be visible via CUDA_VISIBLE_DEVICES but outside the
    # training ranks, i.e. index >= num_processes). Empty string = colocate with training.
    config.dinov3_device = "cuda:6"

    config.run_name = "pickscore-gardo"
    config.save_dir = f'/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/flow-grpo/sd-3-5-medium/model-ckpt/{config.run_name}'
    return config


def get_config(name):
    return globals()[name]()
