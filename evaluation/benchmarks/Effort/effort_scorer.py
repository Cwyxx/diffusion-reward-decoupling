"""Effort AIGI-detector scorer for the aigi-detector eval dataset.

Wraps the official Effort detector (ICML 2025): a CLIP ViT-L/14 vision tower
with an orthogonal SVD-residual adapter on every self_attn linear layer plus a
2-class head. Per image it emits a single higher-is-better score in [0, 1]:

  effort-real-score : 1 - fake_prob, where
                      fake_prob = softmax(head(pooled_feature))[:, 1].

Preprocessing and the model definition replicate the official demo
(flow_grpo/Effort-AIGI-Detection/DeepfakeBench/training/demo.py and
detectors/effort_detector.py), including CLIP mean/std normalization, so the
official GenImage(sdv1.4) checkpoint loads and behaves identically.
"""
from __future__ import annotations

import torch


def probs_to_scores(fake_probs) -> list[dict]:
    """Aggregate per-image fake probabilities into per-image scores.

    fake_probs: 1-D tensor of values in [0, 1].
    Returns one dict per image with the effort-real-score key.
    """
    return [{"effort-real-score": float(1.0 - p.item())} for p in fake_probs]
