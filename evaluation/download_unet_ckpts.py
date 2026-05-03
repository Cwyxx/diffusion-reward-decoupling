"""Download SDXL base pipeline + UNet weights for post-trained methods.

Triggers HuggingFace's auto-download into the default HF cache (override
via HF_HOME). Run once per machine; BoN scripts then load from cache.
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel  # noqa: E402


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# (label, repo_id, subfolder)
UNETS = [
    ("dpo",  "mhdang/dpo-sdxl-text2image-v1",           "unet"),
    ("spo",  "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep", "unet"),
    ("inpo", "JaydenLu666/InPO-SDXL",                    "unet"),
]


def main():
    print(f"\n=== base: {BASE_MODEL_ID} (full pipeline, fp16) ===")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    del pipe  # release RAM; weights stay in HF cache

    for label, repo_id, subfolder in UNETS:
        print(f"\n=== {label}: {repo_id} (subfolder={subfolder!r}) ===")
        unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder=subfolder)
        del unet
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
