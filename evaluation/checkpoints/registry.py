"""Registry mapping method name to its checkpoint loading recipe.

Each RL method's public release has a different format (LoRA adapter, full
UNet weights, full pipeline). This module isolates those differences behind
a uniform CheckpointRecipe; loaders.py dispatches to the right loader.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


SD15 = "runwayml/stable-diffusion-v1-5"

LoadKind = Literal["base", "lora", "unet", "full"]


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
        method="spo", base_model_id=SD15, load_kind="lora",
        repo_id="SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
        extra_kwargs={"weight_name": "spo-sd-v1-5_4k-p_10ep_lora_diffusers.safetensors"},
    ),
}


def get_recipe(method: str) -> CheckpointRecipe:
    if method not in REGISTRY:
        raise KeyError(f"Unknown method: {method!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[method]
