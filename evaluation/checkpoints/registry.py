"""Registry mapping method name to its checkpoint loading recipe.

Each RL method's public release has a different format (LoRA adapter, full
UNet weights, full pipeline). This module isolates those differences behind
a uniform CheckpointRecipe; loaders.py dispatches to the right loader.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


SD15 = "runwayml/stable-diffusion-v1-5"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SD35M = "stabilityai/stable-diffusion-3.5-medium"

# Local root for SD-3.5-M post-trained LoRA adapters (PEFT format: each
# subdir contains adapter_config.json + adapter_model.safetensors).
SD35M_LORA_ROOT = "/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/ckpt/SD-3.5-M"

LoadKind = Literal["base", "lora", "peft_lora", "unet", "full"]


@dataclass
class CheckpointRecipe:
    method: str
    base_model_id: str
    load_kind: LoadKind
    repo_id: Optional[str] = None
    subfolder: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


REGISTRY: Dict[str, CheckpointRecipe] = {
    "base": CheckpointRecipe(method="base", base_model_id=SD15, load_kind="base"),
    "dpo": CheckpointRecipe(
        method="dpo", base_model_id=SD15, load_kind="unet",
        repo_id="mhdang/dpo-sd1.5-text2image-v1",
    ),
    "kto": CheckpointRecipe(
        method="kto", base_model_id=SD15, load_kind="unet",
        repo_id="jacklishufan/diffusion-kto",
    ),
    "inpo": CheckpointRecipe(
        method="inpo", base_model_id=SD15, load_kind="unet",
        repo_id="JaydenLu666/InPO-SD1.5",
    ),
    "smpo": CheckpointRecipe(
        method="smpo", base_model_id=SD15, load_kind="unet",
        repo_id="JaydenLu666/SmPO-SD1.5",
    ),
    "dro": CheckpointRecipe(
        method="dro", base_model_id=SD15, load_kind="unet",
        repo_id="ylwu/diffusion-dro-sd1.5",
    ),
    "spo": CheckpointRecipe(
        method="spo", base_model_id=SD15, load_kind="unet",
        repo_id="SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    ),
    "base-sdxl": CheckpointRecipe(
        method="base-sdxl", base_model_id=SDXL, load_kind="base",
    ),
    "dpo-sdxl": CheckpointRecipe(
        method="dpo-sdxl", base_model_id=SDXL, load_kind="unet",
        repo_id="mhdang/dpo-sdxl-text2image-v1",
    ),
    "spo-sdxl": CheckpointRecipe(
        method="spo-sdxl", base_model_id=SDXL, load_kind="unet",
        repo_id="SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
    ),
    "inpo-sdxl": CheckpointRecipe(
        method="inpo-sdxl", base_model_id=SDXL, load_kind="unet",
        repo_id="JaydenLu666/InPO-SDXL",
    ),
    "smpo-sdxl": CheckpointRecipe(
        method="smpo-sdxl", base_model_id=SDXL, load_kind="unet",
        repo_id="JaydenLu666/SmPO-SDXL",
    ),
    "base-sd3": CheckpointRecipe(
        method="base-sd3", base_model_id=SD35M, load_kind="base",
    ),
    "flowgrpo-pickscore-sd3": CheckpointRecipe(
        method="flowgrpo-pickscore-sd3", base_model_id=SD35M, load_kind="peft_lora",
        repo_id=f"{SD35M_LORA_ROOT}/FlowGRPO-PickScore",
    ),
    "grpo-guard-sd3": CheckpointRecipe(
        method="grpo-guard-sd3", base_model_id=SD35M, load_kind="peft_lora",
        repo_id=f"{SD35M_LORA_ROOT}/GRPO-Guard",
    ),
    "diffusion-dpo-sd3": CheckpointRecipe(
        method="diffusion-dpo-sd3", base_model_id=SD35M, load_kind="peft_lora",
        repo_id=f"{SD35M_LORA_ROOT}/Diffusion-DPO",
    ),
    "realalign-sd3": CheckpointRecipe(
        method="realalign-sd3", base_model_id=SD35M, load_kind="peft_lora",
        repo_id=f"{SD35M_LORA_ROOT}/RealAlign",
    ),
}


def get_recipe(method: str) -> CheckpointRecipe:
    if method not in REGISTRY:
        raise KeyError(f"Unknown method: {method!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[method]
