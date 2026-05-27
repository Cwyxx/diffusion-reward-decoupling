"""CPU unit tests for the PAL4VST mask->score aggregation.

Runnable directly (no pytest required):
    python evaluation/benchmarks/PAL4VST/test_pal4vst_scorer.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.benchmarks.PAL4VST.pal4vst_scorer import masks_to_scores


def test_all_clean():
    m = torch.zeros(1, 1, 8, 8)
    s = masks_to_scores(m, threshold=0.5)[0]
    assert s["pal4vst-clean-rate"] == 1.0
    assert abs(s["pal4vst-clean-area"] - 1.0) < 1e-6


def test_one_artifact_pixel():
    m = torch.zeros(1, 1, 8, 8)
    m[0, 0, 0, 0] = 1.0
    s = masks_to_scores(m, threshold=0.5)[0]
    assert s["pal4vst-clean-rate"] == 0.0  # max 1.0 >= 0.5
    assert abs(s["pal4vst-clean-area"] - (1.0 - 1.0 / 64)) < 1e-6


def test_half_artifact():
    m = torch.zeros(1, 1, 8, 8)
    m[0, 0, :4, :] = 1.0  # half the pixels flagged
    s = masks_to_scores(m, threshold=0.5)[0]
    assert s["pal4vst-clean-rate"] == 0.0
    assert abs(s["pal4vst-clean-area"] - 0.5) < 1e-6


def test_batch_and_3d_input():
    m = torch.zeros(2, 8, 8)  # 3D [N,H,W] should be accepted
    m[1] = 1.0
    out = masks_to_scores(m, threshold=0.5)
    assert len(out) == 2
    assert out[0]["pal4vst-clean-rate"] == 1.0
    assert out[1]["pal4vst-clean-rate"] == 0.0
    assert abs(out[1]["pal4vst-clean-area"] - 0.0) < 1e-6


if __name__ == "__main__":
    test_all_clean()
    test_one_artifact_pixel()
    test_half_artifact()
    test_batch_and_3d_input()
    print("OK: all masks_to_scores tests passed")
