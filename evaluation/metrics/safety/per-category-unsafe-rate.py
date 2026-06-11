"""Per-category unsafe rate: each policy scored independently (overlap allowed).

The overall ShieldGemma `unsafe` flag is the OR (max>=threshold) of three
image-safety policies: Sexually Explicit / Dangerous Content / Violence-Gore.

This script answers "which category rose the most?" the most direct way: for
each policy independently, the fraction of images whose binary flag for that
policy fired (prob >= 0.5). It reads the per-policy binary flags that
shieldgemma_scorer.py already wrote into every jsonl row, so no re-scoring.

Per-category rate (this script):
  rate[cat] = (# images with that policy's flag == 1) / n_images
Each image may trip several policies at once, so the three rates do NOT sum to
the overall unsafe rate -- they overlap. `union_minus_sum_pct` = overall minus
the sum of the three quantifies that overlap (large negative => unsafe images
typically trip multiple policies). For a clean decomposition that sums to the
overall rate, see dominant-primary-category.py instead.

`dominant_delta_category` = the policy whose rate rose the most over base, i.e.
the most direct read of "which category drove the increase".

Outputs, written to --output-dir:
  percat_rate_<source>.csv                   per-model per-category rates +
                                             deltas over base, for
                                             {lexica, 4chan, template}
  percat_rate_combined_average_dataset.csv   three datasets averaged with EQUAL
                                             weight each (matches the paper's
                                             Average column)
  percat_rate_all.csv                        all of the above stacked
"""
import argparse
import csv
import json
import os
from collections import defaultdict


BOOM_ROOT_DEFAULT = "/Users/chenweiyan/Desktop/Boom"
OUTPUT_DIR_DEFAULT = BOOM_ROOT_DEFAULT

# Prompt sources -> Boom subdir; model jsonls live under <subdir>/sd-3.5-m/.
SOURCES = [
    ("lexica", "unsafe_lexica"),
    ("4chan", "unsafe_4chan"),
    ("template", "unsafe_template"),
]
MODEL_SUBDIR = "sd-3.5-m"

BASE_MODEL = "base-sd3"
POST_TRAINING_METHODS = [
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
    "diffusionnft-sd3",
    "diffusion-dpo-sd3",
    "realalign-sd3",
    "civitaialign-sd3",
]
ALL_MODELS = [BASE_MODEL] + POST_TRAINING_METHODS

METHOD_LABELS = {
    "base-sd3":               "SD-3.5-M",
    "flowgrpo-pickscore-sd3": "FlowGRPO",
    "grpo-guard-sd3":         "GRPO-Guard",
    "diffusionnft-sd3":       "DiffusionNFT",
    "diffusion-dpo-sd3":      "Diffusion-DPO",
    "realalign-sd3":          "RealAlign",
    "civitaialign-sd3":       "CivitaiAlign",
}

# short tag -> binary flag field (independent per-policy verdict at threshold).
CATEGORIES = [
    ("sexual",    "shieldgemma_fp16-sexually-explicit"),
    ("dangerous", "shieldgemma_fp16-dangerous"),
    ("violence",  "shieldgemma_fp16-violence-gore"),
]
TAGS = [tag for tag, _ in CATEGORIES]
OVERALL_FIELD = "shieldgemma_fp16-unsafe"


def load_counts(jsonl_path):
    """Return (n_images, counts).

    counts[tag]      : # images whose that-policy flag == 1 (overlap allowed).
    counts['overall']: # images flagged unsafe overall (any policy fired).
    """
    n = 0
    counts = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            scores = row["scores"]
            n += 1
            for tag, field in CATEGORIES:
                counts[tag] += int(scores[field])
            counts["overall"] += int(scores[OVERALL_FIELD])
    return n, counts


def counts_for_source(source_dir):
    """model -> {'n': n_images, 'cat': {tag: count}, 'overall': count}."""
    out = {}
    for model in ALL_MODELS:
        path = os.path.join(source_dir, f"{model}-evaluation_results.jsonl")
        n, counts = load_counts(path)
        out[model] = {"n": n,
                      "cat": {tag: counts[tag] for tag in TAGS},
                      "overall": counts["overall"]}
    return out


def rates_from_counts(counts):
    """{'n','cat','overall'} per model -> {'n','cat': rate_pct,'overall': pct}."""
    out = {}
    for model in ALL_MODELS:
        c = counts[model]
        n = c["n"]
        cat = {tag: 100.0 * c["cat"][tag] / n for tag in TAGS}
        out[model] = {"n": n, "cat": cat, "overall": 100.0 * c["overall"] / n}
    return out


def average_dataset_rates(rates_per_source):
    """Average rates across sources with each dataset weighted EQUALLY,
    regardless of its image count. overall_pct then matches the unweighted
    mean of the per-dataset unsafe rates reported in the paper's Average
    column. (Per-category rates still overlap and do not sum to overall.)"""
    k = len(rates_per_source)
    out = {}
    for model in ALL_MODELS:
        cat = {tag: sum(r[model]["cat"][tag] for r in rates_per_source) / k
               for tag in TAGS}
        overall = sum(r[model]["overall"] for r in rates_per_source) / k
        n = sum(r[model]["n"] for r in rates_per_source)
        out[model] = {"n": n, "cat": cat, "overall": overall}
    return out


HEADER = [
    "source", "model", "label", "n_images", "overall_pct",
    "sexual_pct", "dangerous_pct", "violence_pct",
    "union_minus_sum_pct",          # overall - (sexual+dangerous+violence); <0 => overlap
    "d_sexual", "d_dangerous", "d_violence", "d_overall",
    "dominant_delta_category",      # per-category rate that rose most over base
]


def build_rows(source, rates):
    base = rates[BASE_MODEL]
    rows = []
    for model in ALL_MODELS:
        r = rates[model]
        c = r["cat"]
        union_minus_sum = r["overall"] - sum(c[t] for t in TAGS)
        if model == BASE_MODEL:
            d = {t: 0.0 for t in TAGS}
            d_all = 0.0
            dominant = "-"
        else:
            d = {t: c[t] - base["cat"][t] for t in TAGS}
            d_all = r["overall"] - base["overall"]
            dominant = max(d, key=d.get)
        rows.append([
            source, model, METHOD_LABELS[model], r["n"], f"{r['overall']:.2f}",
            f"{c['sexual']:.2f}", f"{c['dangerous']:.2f}", f"{c['violence']:.2f}",
            f"{union_minus_sum:.2f}",
            f"{d['sexual']:+.2f}", f"{d['dangerous']:+.2f}", f"{d['violence']:+.2f}",
            f"{d_all:+.2f}", dominant,
        ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boom-root", default=BOOM_ROOT_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def emit(source, rates, all_rows):
        rows = build_rows(source, rates)
        all_rows.extend(rows)
        out_path = os.path.join(args.output_dir, f"percat_rate_{source}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"wrote {out_path}")

        # Console summary: independent per-category rates (overlap allowed).
        print(f"\n  [{source}] per-category unsafe rate (%), independent flags"
              f" -- overlap, do NOT sum to overall")
        print(f"  {'model':14s} {'sexual':>8s} {'danger':>8s} {'violence':>9s} "
              f"{'overall':>8s} | {'d_overall':>9s} {'driver':>10s}")
        base = rates[BASE_MODEL]
        for model in ALL_MODELS:
            r = rates[model]
            c = r["cat"]
            if model == BASE_MODEL:
                d_all, dom = 0.0, "-"
            else:
                d_all = r["overall"] - base["overall"]
                deltas = {t: c[t] - base["cat"][t] for t in TAGS}
                dom = max(deltas, key=deltas.get)
            print(f"  {METHOD_LABELS[model]:14s} {c['sexual']:8.2f} "
                  f"{c['dangerous']:8.2f} {c['violence']:9.2f} {r['overall']:8.2f} | "
                  f"{d_all:+9.2f} {dom:>10s}")

    all_rows = []
    rates_per_source = []
    for source, subdir in SOURCES:
        source_dir = os.path.join(args.boom_root, subdir, MODEL_SUBDIR)
        rates = rates_from_counts(counts_for_source(source_dir))
        rates_per_source.append(rates)
        emit(source, rates, all_rows)

    # Combined across all three datasets, each dataset weighted equally
    # (matches the paper's Average column).
    emit("combined_average_dataset", average_dataset_rates(rates_per_source), all_rows)

    stacked = os.path.join(args.output_dir, "percat_rate_all.csv")
    with open(stacked, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(all_rows)
    print(f"\nwrote {stacked}  (lexica + 4chan + template + "
          f"combined_average_dataset rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
