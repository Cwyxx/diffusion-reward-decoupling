"""DiffDoctor artifact-detector scorer for the aigi-detector eval dataset.

Wraps the DiffDoctor SegFormer artifact detector (ICCV 2025). Per image it
emits two higher-is-better scores in [0, 1]:

  diffdoctor-clean-rate : 1.0 if the image has no artifact pixel above the
                          threshold (heatmap.max() < tau), else 0.0
  diffdoctor-clean-area : 1 - fraction of pixels above the threshold

Preprocessing and inference replicate flow_grpo/DiffDoctor/ad_inference.py.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# DiffDoctor artifact-detector checkpoint (already present on the server).
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/DiffDoctor/"
    "ad_pytorch_model.bin"
)
BACKBONE = "nvidia/mit-b5"

_model = None
_preprocessor = None
_loaded_device = None


def heatmaps_to_scores(heatmaps, threshold: float = 0.5) -> list[dict]:
    """Aggregate per-pixel artifact heatmaps into per-image scores.

    heatmaps: tensor of shape [N, 1, H, W] or [N, H, W], values in [0, 1].
    Returns one dict per image with the two diffdoctor-* keys.
    """
    if heatmaps.dim() == 3:
        heatmaps = heatmaps.unsqueeze(1)
    n = heatmaps.shape[0]
    flat = heatmaps.reshape(n, -1)
    max_vals = flat.max(dim=1).values
    area_ratio = (flat > threshold).float().mean(dim=1)
    return [
        {
            "diffdoctor-clean-rate": float(max_vals[i].item() < threshold),
            "diffdoctor-clean-area": float(1.0 - area_ratio[i].item()),
        }
        for i in range(n)
    ]


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("DIFFDOCTOR_CKPT", DEFAULT_CKPT)


def _threshold() -> float:
    return float(os.environ.get("DIFFDOCTOR_THRESHOLD", "0.5"))


def _load(device: str):
    """Load the SegFormer artifact detector once per process, then cache."""
    global _model, _preprocessor, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, _preprocessor, device

    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"DiffDoctor checkpoint not found at {ckpt!r}. Set DIFFDOCTOR_CKPT "
            f"to the ad_pytorch_model.bin path (the ~339 MB weights), or git "
            f"lfs pull the in-repo copy under flow_grpo/DiffDoctor/checkpoints/."
        )

    preprocessor = SegformerImageProcessor.from_pretrained(BACKBONE)
    model = SegformerForSemanticSegmentation.from_pretrained(BACKBONE)
    # DiffDoctor replaces the segmentation head with a single-channel conv.
    model.decode_head.classifier = nn.Conv2d(
        model.decode_head.classifier.in_channels, 1, kernel_size=1
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()

    _model, _preprocessor, _loaded_device = model, preprocessor, device
    return model, preprocessor, device


def _preprocess(pil_images, preprocessor, device):
    """Replicate ad_inference.py: resize 512, RGB, ToTensor, ImageNet-normalize."""
    from torchvision import transforms

    tensors = []
    for img in pil_images:
        arr = np.array(img.convert("RGB"))          # HWC RGB uint8
        arr = cv2.resize(arr, (512, 512))           # same as ad_inference.py
        tensors.append(transforms.ToTensor()(arr))  # CHW float in [0, 1]
    # do_rescale=False: ToTensor already scaled to [0, 1]; processor only
    # resizes (already 512) and applies ImageNet mean/std normalization.
    processed = preprocessor(tensors, return_tensors="pt", do_rescale=False)
    return processed["pixel_values"].to(device)


def _infer_heatmaps(processed, model):
    with torch.no_grad():
        pred = model(processed)
        logits = nn.functional.interpolate(
            pred.logits, size=processed.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        heatmaps = torch.sigmoid(logits)  # [N, 1, 512, 512] in [0, 1]
    return heatmaps


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {diffdoctor-clean-rate, diffdoctor-clean-area} dict per PIL image."""
    model, preprocessor, device = _load(device)
    threshold = _threshold()
    results: list[dict] = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        processed = _preprocess(batch, preprocessor, device)
        heatmaps = _infer_heatmaps(processed, model)
        assert heatmaps.min() >= 0 and heatmaps.max() <= 1
        results.extend(heatmaps_to_scores(heatmaps.cpu(), threshold))
    return results
