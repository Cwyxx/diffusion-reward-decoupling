"""CPU unit tests for the Effort fake-prob -> score aggregation.

Runnable directly (no pytest required):
    python evaluation/benchmarks/Effort/test_effort_scorer.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.benchmarks.Effort.effort_scorer import probs_to_scores


def test_real_image():
    out = probs_to_scores(torch.tensor([0.0]))[0]
    assert abs(out["effort-real-score"] - 1.0) < 1e-6


def test_fake_image():
    out = probs_to_scores(torch.tensor([1.0]))[0]
    assert abs(out["effort-real-score"] - 0.0) < 1e-6


def test_midrange():
    out = probs_to_scores(torch.tensor([0.3]))[0]
    assert abs(out["effort-real-score"] - 0.7) < 1e-6


def test_batch():
    out = probs_to_scores(torch.tensor([0.0, 0.25, 1.0]))
    assert len(out) == 3
    assert abs(out[0]["effort-real-score"] - 1.0) < 1e-6
    assert abs(out[1]["effort-real-score"] - 0.75) < 1e-6
    assert abs(out[2]["effort-real-score"] - 0.0) < 1e-6


if __name__ == "__main__":
    test_real_image()
    test_fake_image()
    test_midrange()
    test_batch()
    print("OK: all probs_to_scores tests passed")
