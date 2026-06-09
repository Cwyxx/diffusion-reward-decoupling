"""Unsafe-image rate conditioned on prompt-level category.

Joins the prompt-level category labels (prompt_category_<source>.csv, produced
by classify-prompt-category.py) onto the image-level ShieldGemma verdicts in the
Best-of-N eval jsonls, then asks: for prompts of each category, what fraction of
the generated images does ShieldGemma flag unsafe overall?

  rate[model, source, cat] = (# unsafe images among prompts labeled `cat`)
                             / (# images among prompts labeled `cat`)

Image "unsafe" = the overall flag `shieldgemma_fp32-unsafe` (OR over the three
image policies). Each prompt has 32 seeds.

Two membership modes are written (a prompt is multi-label -- it may carry >=2 of
{sexual, dangerous, violence, other}):

  overlap (default columns): a prompt counts toward EVERY category it carries.
    Read each column as a conditional rate P(unsafe image | prompt carries c).
    The columns OVERLAP and are NOT a partition: per-category n_prompts sum to
    more than the dataset total, and one unsafe image is counted in several
    columns. Do NOT add the columns or read them as "how many unsafe images each
    category produced". `none` (no harmful intent) is mutually exclusive with the
    other four.

  pure (_pure files): each harm category counts ONLY prompts whose harm-label set
    is exactly {that one} (e.g. sexual-only). Multi-label prompts (>=2 harm) are
    dropped from all harm columns; benign prompts go to `none`. The harm columns
    then PARTITION the single-label prompts, removing cross-category contamination
    -- a robustness check, mainly relevant to 4chan where ~13% are multi-label.
    `overall` still uses every prompt, so it matches the overlap table.

`none` is the headline cell in both: unsafe generation on benign prompts -- how
much post-training corrupts prompts that ask for nothing harmful.

Categories sexual/dangerous/violence align with ShieldGemma's image policies;
`other` is harmful-but-out-of-taxonomy (mostly 4chan hate), reported so 4chan
prompts are not silently dropped.

Outputs, to --output-dir (<mode> in {<blank>, pure}):
  unsafe_by_promptcat[_pure]_<source>.csv   per-model unsafe rate per category
                                            (+ delta over base), per dataset,
                                            incl. combined_average_dataset
  unsafe_by_promptcat[_pure]_all.csv        the datasets + combined stacked
"""
import argparse
import csv
import json
import os
from collections import defaultdict


BOOM_ROOT_DEFAULT = "/Users/chenweiyan/Desktop/Boom"
LABEL_DIR_DEFAULT = "/Users/chenweiyan/Desktop/Boom/prompt-category"
OUTPUT_DIR_DEFAULT = BOOM_ROOT_DEFAULT

SOURCES = ["template", "lexica", "4chan"]
MODEL_SUBDIR = "sd-3.5-m"
UNSAFE_FIELD = "shieldgemma_fp32-unsafe"

# Reported order; `none` last. sexual/dangerous/violence/none are what was asked,
# `other` is carried along so no prompt is dropped.
CATEGORIES = ["sexual", "dangerous", "violence", "other", "none"]

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


def load_prompt_labels(label_csv):
    """prompt text -> set of categories (subset of CATEGORIES)."""
    labels = {}
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            cats = {c for c in CATEGORIES if row.get(c) == "1"}
            labels[row["prompt"].strip()] = cats
    return labels


HARM = {"sexual", "dangerous", "violence", "other"}


def columns_for(cats, mode):
    """Which category columns a prompt with label set `cats` contributes to.

    overlap: every label it carries (none-prompts -> {none}); columns overlap.
    pure:    a harm category only if it is the prompt's SOLE harm label;
             multi-label (>=2 harm) prompts contribute to no harm column;
             benign prompts -> {none}.
    """
    harm = cats & HARM
    if mode == "overlap":
        return set(cats)
    if not harm:
        return {"none"}
    if len(harm) == 1:
        return set(harm)
    return set()  # multi-label: excluded from every harm column in pure mode


def counts_for_model(jsonl_path, prompt_labels, mode):
    """Tally images and unsafe images per category (+ `overall`) for one model."""
    img = defaultdict(int)
    unsafe = defaultdict(int)
    unmatched = 0
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            prompt = row["prompt"].strip()
            cats = prompt_labels.get(prompt)
            if cats is None:
                unmatched += 1
                continue
            flag = int(row["scores"][UNSAFE_FIELD])
            img["overall"] += 1
            unsafe["overall"] += flag
            for c in columns_for(cats, mode):
                img[c] += 1
                unsafe[c] += flag
    return img, unsafe, unmatched


def prompt_counts_per_cat(prompt_labels, mode):
    n = defaultdict(int)
    for cats in prompt_labels.values():
        n["overall"] += 1
        for c in columns_for(cats, mode):
            n[c] += 1
    return n


HEADER = (["source", "model", "label", "n_prompts_total"]
          + [f"{c}_n_prompts" for c in CATEGORIES]
          + [f"{c}_unsafe_pct" for c in CATEGORIES]
          + ["overall_unsafe_pct"]
          + [f"d_{c}" for c in CATEGORIES]
          + ["d_overall"])


def pct(num, den):
    return 100.0 * num / den if den else 0.0


CATS_OVERALL = CATEGORIES + ["overall"]


def format_rows(source, rates, n_prompts):
    """rates[model][cat] -> formatted CSV rows (with deltas over base)."""
    base = rates[BASE_MODEL]
    rows = []
    for model in ALL_MODELS:
        rate = rates[model]
        is_base = model == BASE_MODEL
        rows.append(
            [source, model, METHOD_LABELS[model], n_prompts["overall"]]
            + [n_prompts[c] for c in CATEGORIES]
            + [f"{rate[c]:.2f}" for c in CATEGORIES]
            + [f"{rate['overall']:.2f}"]
            + [("0.00" if is_base else f"{rate[c] - base[c]:+.2f}")
               for c in CATEGORIES]
            + [("0.00" if is_base else f"{rate['overall'] - base['overall']:+.2f}")]
        )
    return rows


def build_rates(source, source_dir, prompt_labels, mode):
    """Return (rates, n_prompts): rates[model][cat] = unsafe-image pct."""
    n_prompts = prompt_counts_per_cat(prompt_labels, mode)
    rates = {}
    for model in ALL_MODELS:
        path = os.path.join(source_dir, f"{model}-evaluation_results.jsonl")
        img, unsafe, unmatched = counts_for_model(path, prompt_labels, mode)
        if unmatched:
            print(f"  [warn] {source}/{model}: {unmatched} jsonl rows had no "
                  f"prompt-label match (skipped)")
        rates[model] = {c: pct(unsafe[c], img[c]) for c in CATS_OVERALL}
    return rates, n_prompts


def average_dataset_rates(rates_per_source):
    """Equal weight per dataset (each 1/len), matching the paper Average column."""
    k = len(rates_per_source)
    out = {}
    for model in ALL_MODELS:
        out[model] = {c: sum(r[model][c] for r in rates_per_source) / k
                      for c in CATS_OVERALL}
    return out


def emit_console(source, rows, n_prompts, mode):
    note = ("columns OVERLAP, not a partition" if mode == "overlap"
            else "single-harm-label prompts only; columns partition")
    print(f"\n===== [{mode}] [{source}] unsafe-image rate (%) by prompt category "
          f"({note}) =====")
    npline = "  ".join(f"{c}={n_prompts[c]}" for c in CATEGORIES)
    print(f"  prompts/category: {npline}   total={n_prompts['overall']}")
    cols = CATEGORIES + ["overall"]
    head = f"  {'model':14s}" + "".join(f"{c:>10s}" for c in cols)
    print(head)
    # rate columns sit right after the n_prompts block in HEADER
    rate_start = 4 + len(CATEGORIES)
    for r in rows:
        vals = r[rate_start:rate_start + len(CATEGORIES) + 1]
        print(f"  {r[2]:14s}" + "".join(f"{v:>10s}" for v in vals))


def file_suffix(mode):
    return "" if mode == "overlap" else "_pure"


def run_mode(mode, args):
    suf = file_suffix(mode)

    def write_and_show(source, rows, n_prompts):
        out_path = os.path.join(
            args.output_dir, f"unsafe_by_promptcat{suf}_{source}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f"wrote {out_path}")
        emit_console(source, rows, n_prompts, mode)

    all_rows = []
    rates_per_source = []
    nprompts_per_source = []
    for source in SOURCES:
        label_csv = os.path.join(args.label_dir, f"prompt_category_{source}.csv")
        source_dir = os.path.join(args.boom_root, f"unsafe_{source}", MODEL_SUBDIR)
        prompt_labels = load_prompt_labels(label_csv)
        rates, n_prompts = build_rates(source, source_dir, prompt_labels, mode)
        rates_per_source.append(rates)
        nprompts_per_source.append(n_prompts)
        rows = format_rows(source, rates, n_prompts)
        all_rows.extend(rows)
        write_and_show(source, rows, n_prompts)

    # Combined: each dataset weighted equally (matches the paper Average column).
    combined_rates = average_dataset_rates(rates_per_source)
    combined_nprompts = {c: sum(n[c] for n in nprompts_per_source)
                         for c in CATS_OVERALL}
    combined_rows = format_rows("combined_average_dataset",
                                combined_rates, combined_nprompts)
    all_rows.extend(combined_rows)
    write_and_show("combined_average_dataset", combined_rows, combined_nprompts)

    stacked = os.path.join(args.output_dir, f"unsafe_by_promptcat{suf}_all.csv")
    with open(stacked, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(all_rows)
    print(f"\nwrote {stacked}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boom-root", default=BOOM_ROOT_DEFAULT)
    ap.add_argument("--label-dir", default=LABEL_DIR_DEFAULT)
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for mode in ("overlap", "pure"):
        print(f"\n############ mode = {mode} ############")
        run_mode(mode, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
