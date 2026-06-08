# evaluation/metrics

Standalone scripts for the Best-of-N evaluation pipeline and its downstream
analysis/plots. Every script is run directly (`python evaluation/metrics/<group>/<script>.py ...`),
never imported — filenames use hyphens, so there is no Python-module coupling
between them. They are grouped by **evaluation topic**:

```
core/        shared pipeline: generate -> score -> aggregate
radar/       overview radar charts (capability + properties)
capability/  per-benchmark Best-of-N comparison plots
safety/      Responsible-AI / unsafe-content analysis
bias/        DallEval social-bias analysis
detector/    AIGI real/fake detector accuracy
```

## core/ — shared Best-of-N pipeline

Run in order: **generate → score → aggregate**.

| Script | Purpose |
|---|---|
| `generate-images-bestofn.py` | Generate N images/prompt for the Best-of-N ceiling eval (SD-v1.5 / SDXL / SD-3.5-M). Multi-GPU launcher (`--gpus 0,1,2,3`) that forks one worker per GPU and merges per-rank outputs. |
| `generate-images.py` | Single-image generation for SD-3.5-M (older, non-BoN path). |
| `score-images.py` | Score generated images with every reward/metric scorer; writes `evaluation_results.jsonl`. The pipeline entry point most other scripts read from. |
| `aggregate-bestofn.py` | Compute Best-of-N (max@N / pass@N) curves per `(method, dataset)` from `evaluation_results.jsonl`. Produces the `bestofn/csv/*_curve.csv` consumed by the `capability/` and `safety/` plotters. |
| `average-across-seeds.py` | Average a metric across seeds. |
| `average-anytext-seeds.py` | AnyText OCR Average@K (the paper's mean-over-K protocol, not max). Read-only companion to `aggregate-bestofn.py`. |

## radar/ — overview radars

Collectors read the per-run `evaluation_results.jsonl` and emit a small JSON that
the radar plotters render.

| Script | Purpose |
|---|---|
| `collect-mean-of-n-metrics.py` | Collect the mean@N property metrics (per-image rates / bias / safety) → `mean-of-n.json`. |
| `collect-max-of-n-metrics.py` | Collect the max@N capability metrics (Best-of-N ceiling) → `max-of-n.json`. |
| `collect-geneval-max-of-n.py` | Collect per-GenEval-subskill pass@N (6 tags + Overall macro-avg) → `geneval-max-of-n.json`. Focused sibling of `collect-max-of-n-metrics.py`. |
| `collect-wise-max-of-n.py` | Collect per-WISE-category pass@N (6 categories + Overall weighted-sum) → `wise-max-of-n.json`. Focused sibling of `collect-max-of-n-metrics.py`. |
| `plot-radar-overall.py` | Combined overview radar (capability + properties on one 9-axis chart) at one N. |
| `plot-radar-max-of-n.py` | Best-of-N capability radar grid (2×3, one per N in 1/2/4/8/16/32, 6 axes). |
| `plot-radar-mean-of-n.py` | Mean-of-N properties radar (Safety / Gender Balance / Clean Rate / Realism, 4 axes) at one N. |
| `plot-radar-geneval.py` | GenEval sub-skill radar (7 axes: 6 tags + Overall) from `geneval-max-of-n.json`; same style as `plot-radar-overall.py`. |
| `plot-radar-wise.py` | WISE sub-domain radar (7 axes: 6 categories + Overall) from `wise-max-of-n.json`; same style as `plot-radar-overall.py`. |

## capability/ — per-benchmark Best-of-N comparison plots

All read `aggregate-bestofn.py`'s `bestofn/csv/*_curve.csv` and overlay the methods.

| Script | Purpose |
|---|---|
| `plot-bestofn-comparison.py` | Generic BoN curves, base vs post-training methods, one figure per metric. |
| `plot-bestofn-wise-comparison.py` | WISE BoN curves, one figure per category (+ weighted Overall). |
| `plot-bestofn-dpg-comparison.py` | DPG-Bench BoN curve across the five SD-3.5-M methods. |
| `plot-bestofn-geneval-comparison.py` | GenEval BoN curves, one figure per tag (+ macro Overall). |
| `plot-bestofn-anytext-comparison.py` | AnyText OCR BoN curves, one figure per metric. |
| `plot-tradeoff-hps-vs-benchmark.py` | Trade-off scatter: aggregated human-preference vs GenEval / WISE BoN@32. |

## safety/ — Responsible-AI / unsafe-content analysis

| Script | Purpose |
|---|---|
| `analyze-unsafe-mode.py` | Decompose each method's unsafe-rate increase over base into activation (Cold prompts) vs amplification (Warm/Hot prompts). |
| `plot-unsafe-at-n-comparison.py` | Unsafe-rate-vs-N curves, one figure per `(classifier, dataset)`. |
| `prompt-level-unsafe-ratio.py` | Per-prompt unsafe ratio across methods (`sd-safety-flag`, `shieldgemma_fp32-unsafe`). |
| `base-cold-prompt-rate.py` | Per-prompt unsafe rate on base-Cold prompts (where base produced zero unsafe images across all seeds). |

## bias/ — DallEval social bias

| Script | Purpose |
|---|---|
| `aggregate-dalleval-bias.py` | DallEval social-bias (MAD) aggregation from `score-images.py`'s per-image discrete labels. A separate CPU step from `aggregate-bestofn.py`. |
| `dalleval-gender-category-comparison.py` | Cross-model gender-category comparison over per-profession labels (male/female/neutral counts). |

## detector/ — AIGI detector accuracy

| Script | Purpose |
|---|---|
| `analyze-detector-accuracy.py` | Real/fake judgement accuracy for the Effort and DRCT AIGI detectors over an `aigi-detector` eval directory. |
