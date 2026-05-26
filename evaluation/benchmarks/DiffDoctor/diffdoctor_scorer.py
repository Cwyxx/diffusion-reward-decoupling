"""DiffDoctor artifact-detector scorer for the aigi-detector eval dataset.

Wraps the DiffDoctor SegFormer artifact detector (ICCV 2025). Per image it
emits two higher-is-better scores in [0, 1]:

  diffdoctor-clean-rate : 1.0 if the image has no artifact pixel above the
                          threshold (heatmap.max() < tau), else 0.0
  diffdoctor-clean-area : 1 - fraction of pixels above the threshold

Preprocessing and inference replicate flow_grpo/DiffDoctor/ad_inference.py.
"""
from __future__ import annotations

import torch


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
