"""Download UNet weights for the SDXL post-trained methods we BoN-evaluate.

Triggers HuggingFace's auto-download into the default HF cache (override
via HF_HOME). Run once per machine; BoN scripts then load from cache.
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffusers import UNet2DConditionModel  # noqa: E402  (import after env)


# (label, repo_id, subfolder)
MODELS = [
    ("dpo",  "mhdang/dpo-sdxl-text2image-v1",           "unet"),
    ("spo",  "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep", "unet"),
    ("inpo", "JaydenLu666/InPO-SDXL",                    "unet"),
]


def main():
    for label, repo_id, subfolder in MODELS:
        print(f"\n=== {label}: {repo_id} (subfolder={subfolder!r}) ===")
        unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder=subfolder)
        del unet  # release RAM; weights stay in HF cache
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
