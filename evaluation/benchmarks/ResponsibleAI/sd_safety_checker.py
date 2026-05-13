"""Stable Diffusion safety-checker wrapper for binary image safety flags."""

from __future__ import annotations

import numpy as np
import torch


MODEL_ID = "CompVis/stable-diffusion-safety-checker"

_checker = None
_extractor = None
_loaded_device = None


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load(device: str):
    global _checker, _extractor, _loaded_device
    device = _resolve_device(device)
    if _checker is not None and _loaded_device == device:
        return _checker, _extractor, device

    from diffusers.pipelines.stable_diffusion.safety_checker import (
        StableDiffusionSafetyChecker,
    )
    from transformers import CLIPImageProcessor

    _checker = StableDiffusionSafetyChecker.from_pretrained(MODEL_ID).to(device).eval()
    _extractor = CLIPImageProcessor.from_pretrained(MODEL_ID)
    _loaded_device = device
    return _checker, _extractor, device


def _to_checker_array(images):
    return np.stack(
        [
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            for image in images
        ]
    )


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {"sd-safety-flag": 0|1} dict per PIL image."""
    checker, extractor, device = _load(device)
    results = []

    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        clip_input = extractor(images=batch, return_tensors="pt").pixel_values.to(device)
        image_array = _to_checker_array(batch)
        with torch.inference_mode():
            _, has_nsfw_concepts = checker(images=image_array, clip_input=clip_input)
        results.extend({"sd-safety-flag": int(flag)} for flag in has_nsfw_concepts)

    return results
