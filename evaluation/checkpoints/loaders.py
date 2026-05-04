"""Generic checkpoint loaders. Each takes a CheckpointRecipe + device + dtype
and returns a ready-to-call StableDiffusionPipeline.
"""
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from evaluation.checkpoints.registry import CheckpointRecipe, get_recipe


def _pipeline_cls(method: str):
    if method.endswith("-sdxl"):
        return StableDiffusionXLPipeline
    if method.endswith("-sd3"):
        return StableDiffusion3Pipeline
    return StableDiffusionPipeline


def _finalize(pipeline: DiffusionPipeline, method: str, device, dtype) -> DiffusionPipeline:
    pipeline.to(device, dtype=dtype)
    # SD-3.5-M VAE is unstable in bf16/fp16; keep it in fp32.
    if method.endswith("-sd3"):
        pipeline.vae.to(device, dtype=torch.float32)
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def load_base(recipe: CheckpointRecipe, device, dtype) -> DiffusionPipeline:
    cls = _pipeline_cls(recipe.method)
    pipeline = cls.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    return _finalize(pipeline, recipe.method, device, dtype)


def load_unet(recipe: CheckpointRecipe, device, dtype) -> DiffusionPipeline:
    """Load base pipeline, then replace its UNet with weights from recipe.repo_id."""
    if recipe.repo_id is None:
        raise ValueError(f"load_unet requires repo_id (method={recipe.method})")

    cls = _pipeline_cls(recipe.method)
    pipeline = cls.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(
        recipe.repo_id,
        subfolder=recipe.subfolder or "unet",
        torch_dtype=dtype,
    )
    pipeline.unet = unet
    return _finalize(pipeline, recipe.method, device, dtype)


def load_lora(recipe: CheckpointRecipe, device, dtype) -> DiffusionPipeline:
    """Load base pipeline, then attach a LoRA adapter from recipe.repo_id."""
    if recipe.repo_id is None:
        raise ValueError(f"load_lora requires repo_id (method={recipe.method})")

    cls = _pipeline_cls(recipe.method)
    pipeline = cls.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    pipeline.load_lora_weights(
        recipe.repo_id,
        subfolder=recipe.subfolder,
        **recipe.extra_kwargs,
    )
    return _finalize(pipeline, recipe.method, device, dtype)


def load_peft_lora(recipe: CheckpointRecipe, device, dtype) -> DiffusionPipeline:
    """Attach a PEFT-format LoRA adapter to the pipeline's transformer.

    Used for SD-3.5-M post-trained methods: the directory at recipe.repo_id
    holds adapter_config.json + adapter_model.safetensors targeting the
    MMDiT transformer. PEFT reads the saved adapter_config.json directly,
    so we don't need to hardcode rank/alpha/target_modules here.
    """
    from peft import PeftModel

    if recipe.repo_id is None:
        raise ValueError(f"load_peft_lora requires repo_id (method={recipe.method})")

    cls = _pipeline_cls(recipe.method)
    pipeline = cls.from_pretrained(recipe.base_model_id, torch_dtype=dtype)
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer, recipe.repo_id, is_trainable=False
    )
    pipeline.transformer.eval()
    return _finalize(pipeline, recipe.method, device, dtype)


def load_full(recipe: CheckpointRecipe, device, dtype) -> DiffusionPipeline:
    """Load entire pipeline directly from recipe.repo_id (release is self-contained)."""
    if recipe.repo_id is None:
        raise ValueError(f"load_full requires repo_id (method={recipe.method})")

    cls = _pipeline_cls(recipe.method)
    pipeline = cls.from_pretrained(recipe.repo_id, torch_dtype=dtype)
    return _finalize(pipeline, recipe.method, device, dtype)


_LOADERS = {
    "base": load_base,
    "lora": load_lora,
    "peft_lora": load_peft_lora,
    "unet": load_unet,
    "full": load_full,
}


def load_pipeline(method: str, device="cuda", dtype=torch.float32) -> DiffusionPipeline:
    recipe = get_recipe(method)
    if recipe.load_kind not in _LOADERS:
        raise NotImplementedError(
            f"No loader registered for load_kind={recipe.load_kind!r} (method={method!r})."
        )
    return _LOADERS[recipe.load_kind](recipe, device, dtype)
