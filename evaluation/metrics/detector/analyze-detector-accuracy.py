# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real/fake judgement accuracy for the Effort and DRCT AIGI detectors.

Standalone analysis over an aigi-detector eval directory's
evaluation_results.jsonl (does NOT re-score or touch the scoring pipeline).

Every image in the aigi-detector dataset is generated, so its ground-truth
label is "fake". Effort and DRCT each emit a higher-is-better *-real-score in
[0, 1] (probability the image is a REAL photo). A detector judges an image
"real" when real-score >= threshold and "fake" otherwise, so on this all-fake
set the per-image judgement is correct exactly when real-score < threshold.

For each detector we report, over all scored images:
  fake-detection-accuracy : mean(real-score < threshold)  (correctly flagged fake)
  fooled-rate             : mean(real-score >= threshold) (mistaken for real)
  mean/median real-score  : the raw score distribution summary

Output: ${output_dir}/detector_accuracy.json plus a printed table.

This only measures accuracy on the fake class (no real images are scored).
It is a detection rate, not a balanced real-vs-fake accuracy.
"""
import argparse
import json
import os
import statistics

# Detector name -> the per-image score key it writes into row["scores"].
# Both are "real" probabilities (higher = looks more like a real photo).
DETECTOR_SCORE_KEYS = {
    "effort": "effort-real-score",
    "drct": "drct-real-score",
}


def load_results(output_dir):
    path = os.path.join(output_dir, "evaluation_results.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No evaluation_results.jsonl in {output_dir!r}")
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def analyze_detector(rows, score_key, threshold):
    """Return the accuracy summary for one detector over all rows that have its score."""
    scores = [
        r["scores"][score_key]
        for r in rows
        if isinstance((r.get("scores") or {}).get(score_key), (int, float))
    ]
    n = len(scores)
    if n == 0:
        return None
    # All images are fake -> a correct judgement is "fake" i.e. real-score < threshold.
    n_detected_fake = sum(1 for s in scores if s < threshold)
    return {
        "score_key": score_key,
        "threshold": threshold,
        "n_images": n,
        "fake_detection_accuracy": n_detected_fake / n,
        "fooled_rate": 1.0 - n_detected_fake / n,
        "mean_real_score": statistics.fmean(scores),
        "median_real_score": statistics.median(scores),
    }


def main(args):
    rows = load_results(args.output_dir)
    n_total = len(rows)

    summary = {"threshold": args.threshold, "n_rows_total": n_total, "detectors": {}}
    for name in args.detectors:
        score_key = DETECTOR_SCORE_KEYS[name]
        result = analyze_detector(rows, score_key, args.threshold)
        if result is None:
            print(f"[skip] {name}: no rows carry '{score_key}' "
                  f"(run score-images.py with this metric first).")
            continue
        summary["detectors"][name] = result

    out_path = os.path.join(args.output_dir, "detector_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n=== Real/fake judgement accuracy (threshold={args.threshold}; "
          f"all images are fake) ===")
    header = f"{'detector':<10}{'n':>8}{'fake-detect-acc':>18}{'fooled':>10}{'mean-real':>12}"
    print(header)
    print("-" * len(header))
    for name, r in summary["detectors"].items():
        print(f"{name:<10}{r['n_images']:>8}"
              f"{r['fake_detection_accuracy']:>18.4f}"
              f"{r['fooled_rate']:>10.4f}"
              f"{r['mean_real_score']:>12.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute real/fake judgement accuracy for Effort/DRCT over an "
                    "aigi-detector evaluation_results.jsonl (fake-class detection rate).")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="aigi-detector eval dir containing evaluation_results.jsonl.")
    parser.add_argument(
        "--detectors", type=str, nargs="+", default=["effort", "drct"],
        choices=sorted(DETECTOR_SCORE_KEYS), help="Detectors to analyze.")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="real-score >= threshold is judged 'real'; below is 'fake' (default 0.5).")
    main(parser.parse_args())
