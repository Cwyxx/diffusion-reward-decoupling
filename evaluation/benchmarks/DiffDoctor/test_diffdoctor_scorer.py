"""CPU unit tests for the DiffDoctor heatmap->score aggregation.

Runnable directly (no pytest required):
    python evaluation/benchmarks/DiffDoctor/test_diffdoctor_scorer.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.benchmarks.DiffDoctor.diffdoctor_scorer import heatmaps_to_scores


def test_all_clean():
    hm = torch.zeros(1, 1, 8, 8)
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 1.0
    assert abs(s["diffdoctor-clean-area"] - 1.0) < 1e-6


def test_one_artifact_pixel():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 0, 0] = 0.9
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 0.0  # max 0.9 >= 0.5
    assert abs(s["diffdoctor-clean-area"] - (1.0 - 1.0 / 64)) < 1e-6


def test_half_artifact():
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, :4, :] = 0.9  # half the pixels above threshold
    s = heatmaps_to_scores(hm, threshold=0.5)[0]
    assert s["diffdoctor-clean-rate"] == 0.0
    assert abs(s["diffdoctor-clean-area"] - 0.5) < 1e-6


def test_batch_and_3d_input():
    hm = torch.zeros(2, 8, 8)  # 3D [N,H,W] should be accepted
    hm[1] = 0.9
    out = heatmaps_to_scores(hm, threshold=0.5)
    assert len(out) == 2
    assert out[0]["diffdoctor-clean-rate"] == 1.0
    assert out[1]["diffdoctor-clean-rate"] == 0.0


if __name__ == "__main__":
    test_all_clean()
    test_one_artifact_pixel()
    test_half_artifact()
    test_batch_and_3d_input()
    print("OK: all heatmaps_to_scores tests passed")
