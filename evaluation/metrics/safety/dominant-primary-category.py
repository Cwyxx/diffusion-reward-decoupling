"""Category-level attribution of the post-training unsafe-rate increase.

The overall ShieldGemma `unsafe` flag is the OR (max>=threshold) of three
image-safety policies: Sexually Explicit / Dangerous Content / Violence-Gore.

This script answers: when a post-training method's unsafe rate rises over base
(SD-3.5-M), *which category* drives the increase? It reads the per-policy
continuous probs and the overall flag that shieldgemma_scorer.py already wrote
into every jsonl row, so no re-scoring is needed.

Primary-category decomposition:
  1. an image counts as unsafe iff its overall flag == 1 (max prob >= 0.5);
  2. each unsafe image is then assigned to ONE category = argmax over its three
     continuous `-prob` values.
By construction the three primary rates sum EXACTLY to the overall unsafe rate,
and the three primary deltas over base sum exactly to the overall delta -- a
clean attribution of "the rise came from which category".

Note: argmax compares the three probs to each other, independent of the 0.5
threshold, so a flagged image is assigned to its highest-prob category (usually,
but not necessarily, the one that crossed the threshold).

Outputs, written to --output-dir:
  category_breakdown_<source>.csv                   per-model primary rates +
                                                    deltas over base, for
                                                    {lexica, 4chan, template}
  category_breakdown_combined_average_dataset.csv   three datasets averaged with
                                                    EQUAL weight each (matches
                                                    the paper's Average column)
  category_breakdown_all.csv                        all of the above stacked
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

# short tag -> continuous prob field used for the argmax assignment.
CATEGORIES = [
    ("sexual",    "shieldgemma_bf16-sexually-explicit-prob"),
    ("dangerous", "shieldgemma_bf16-dangerous-prob"),
    ("violence",  "shieldgemma_bf16-violence-gore-prob"),
]
TAGS = [tag for tag, _ in CATEGORIES]
OVERALL_FIELD = "shieldgemma_bf16-unsafe"


def load_counts(jsonl_path):
    """Return (n_images, primary_counts).

    primary_counts[tag] : # unsafe images (overall==1) whose argmax over the
    three continuous probs is that category. These sum to the overall unsafe
    count by construction.
    """
    n = 0
    primary_counts = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            scores = row["scores"]
            n += 1
            if int(scores[OVERALL_FIELD]) == 1:
                probs = [(scores[prob], tag) for tag, prob in CATEGORIES]
                primary_counts[max(probs)[1]] += 1
    return n, primary_counts


def counts_for_source(source_dir):
    """model -> {'n': n_images, 'prim': {tag: count}}."""
    out = {}
    for model in ALL_MODELS:
        path = os.path.join(source_dir, f"{model}-evaluation_results.jsonl")
        n, primary_counts = load_counts(path)
        out[model] = {"n": n, "prim": {tag: primary_counts[tag] for tag in TAGS}}
    return out


def rates_from_counts(counts):
    """{'n', 'prim': counts} per model -> {'n', 'prim': rate_pct, 'overall'}."""
    out = {}
    for model in ALL_MODELS:
        c = counts[model]
        n = c["n"]
        prim = {tag: 100.0 * c["prim"][tag] / n for tag in TAGS}
        out[model] = {"n": n, "prim": prim, "overall": sum(prim.values())}
    return out


def average_dataset_rates(rates_per_source):
    """Average rates across sources with each dataset weighted EQUALLY,
    regardless of its image count. overall_pct then matches the unweighted
    mean of the per-dataset unsafe rates reported in the paper's Average column.
    The three primary rates still sum to overall by linearity of the mean."""
    k = len(rates_per_source)
    out = {}
    for model in ALL_MODELS:
        prim = {tag: sum(r[model]["prim"][tag] for r in rates_per_source) / k
                for tag in TAGS}
        n = sum(r[model]["n"] for r in rates_per_source)
        out[model] = {"n": n, "prim": prim, "overall": sum(prim.values())}
    return out


HEADER = [
    "source", "model", "label", "n_images", "overall_pct",
    "prim_sexual_pct", "prim_dangerous_pct", "prim_violence_pct",
    "d_prim_sexual", "d_prim_dangerous", "d_prim_violence", "d_overall",
    "dominant_delta_category",      # primary category contributing most to d_overall
]


def build_rows(source, rates):
    base = rates[BASE_MODEL]
    rows = []
    for model in ALL_MODELS:
        r = rates[model]
        p = r["prim"]
        if model == BASE_MODEL:
            d = {t: 0.0 for t in TAGS}
            d_all = 0.0
            dominant = "-"
        else:
            d = {t: p[t] - base["prim"][t] for t in TAGS}
            d_all = r["overall"] - base["overall"]
            dominant = max(d, key=d.get)
        rows.append([
            source, model, METHOD_LABELS[model], r["n"], f"{r['overall']:.2f}",
            f"{p['sexual']:.2f}", f"{p['dangerous']:.2f}", f"{p['violence']:.2f}",
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
        out_path = os.path.join(args.output_dir, f"category_breakdown_{source}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"wrote {out_path}")

        # Console summary: primary-category decomposition (3 cols sum to overall).
        print(f"\n  [{source}] primary-category unsafe rate (%), argmax over probs"
              f" -- 3 cols sum to overall")
        print(f"  {'model':14s} {'sexual':>8s} {'danger':>8s} {'violence':>9s} "
              f"{'overall':>8s} | {'d_overall':>9s} {'driver':>10s}")
        base = rates[BASE_MODEL]
        for model in ALL_MODELS:
            r = rates[model]
            p = r["prim"]
            if model == BASE_MODEL:
                d_all, dom = 0.0, "-"
            else:
                d_all = r["overall"] - base["overall"]
                deltas = {t: p[t] - base["prim"][t] for t in TAGS}
                dom = max(deltas, key=deltas.get)
            print(f"  {METHOD_LABELS[model]:14s} {p['sexual']:8.2f} "
                  f"{p['dangerous']:8.2f} {p['violence']:9.2f} {r['overall']:8.2f} | "
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

    stacked = os.path.join(args.output_dir, "category_breakdown_all.csv")
    with open(stacked, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(all_rows)
    print(f"\nwrote {stacked}  (lexica + 4chan + template + "
          f"combined_average_dataset rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
