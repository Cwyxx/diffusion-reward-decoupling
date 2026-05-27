"""PAL4VST perceptual-artifacts scorer for the aigi-detector eval dataset.

Wraps the PAL4VST artifact localizer (ICCV 2023): a Swin-Large UPerNet semantic
segmentation model that flags perceptual-artifact regions in synthesized images.
Per image it emits two higher-is-better scores in [0, 1]:

  pal4vst-clean-rate : 1.0 if the image has no artifact pixel (mask is all zeros),
                       else 0.0
  pal4vst-clean-area : 1 - par, where par is the fraction of pixels flagged as
                       artifacts (the "perceptual artifacts ratio" from
                       PAL4VST/curate_images.py)

Inference uses the deployed torchscript checkpoint (mmdeploy end2end.pt). It is
self-contained — torch.jit.load + a plain forward, no mmseg/mmcv runtime needed.
The deployed net is fixed at batch_size=1, 512x512, FP32 (deploy.json), so images
are scored one at a time. Preprocessing replicates PAL4VST/utils.prepare_input.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# Deployed PAL4VST torchscript checkpoint (already present on the server).
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/PAL4VST/"
    "swin-large_upernet_unified_512x512/end2end.pt"
)

# Normalization constants from PAL4VST/utils.get_mean_stdinv (0-255 RGB range).
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]

_model = None
_loaded_device = None


def masks_to_scores(masks, threshold: float = 0.5) -> list[dict]:
    """Aggregate per-pixel artifact masks into per-image scores.

    masks: tensor of shape [N, 1, H, W] or [N, H, W]. The deployed net argmaxes
    to a binary {0, 1} map (1 = artifact), but soft values are handled too.
    Returns one dict per image with the two pal4vst-* keys.
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    n = masks.shape[0]
    flat = masks.reshape(n, -1)
    max_vals = flat.max(dim=1).values
    area_ratio = (flat > threshold).float().mean(dim=1)  # par
    return [
        {
            "pal4vst-clean-rate": float(max_vals[i].item() < threshold),
            "pal4vst-clean-area": float(1.0 - area_ratio[i].item()),
        }
        for i in range(n)
    ]


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("PAL4VST_CKPT", DEFAULT_CKPT)


def _threshold() -> float:
    return float(os.environ.get("PAL4VST_THRESHOLD", "0.5"))


def _load(device: str):
    """torch.jit.load the deployed torchscript net once per process, then cache."""
    global _model, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, device

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"PAL4VST checkpoint not found at {ckpt!r}. Set PAL4VST_CKPT to the "
            f"deployed torchscript end2end.pt path, or download "
            f"swin-large_upernet_unified_512x512 from the PAL4VST project page "
            f"(see flow_grpo/PAL4VST/README.md) and place it under "
            f"deployment/pal4vst/swin-large_upernet_unified_512x512/."
        )

    model = torch.jit.load(ckpt, map_location=device)
    model.to(device).eval()
    _model, _loaded_device = model, device
    return model, device


def _preprocess(pil_image, device):
    """Replicate PAL4VST/utils.prepare_input: resize 512, mean/std-normalize 0-255.

    Returns a [1, 3, 512, 512] float tensor (the deployed net is batch_size=1).
    """
    arr = np.array(pil_image.convert("RGB").resize((512, 512)))  # HWC RGB uint8
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (img - mean) / std


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 1) -> list[dict]:
    """Return one {pal4vst-clean-rate, pal4vst-clean-area} dict per PIL image.

    batch_size is accepted for API parity with the other scorers but ignored:
    the deployed torchscript net is fixed at batch_size=1 (deploy.json).
    """
    model, device = _load(device)
    threshold = _threshold()
    results: list[dict] = []
    for img in pil_images:
        processed = _preprocess(img, device)
        with torch.no_grad():
            pal = model(processed)  # [1, 1, 512, 512] argmaxed {0, 1} mask
        results.extend(masks_to_scores(pal.float().cpu(), threshold))
    return results
