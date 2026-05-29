"""Per-prompt unsafe ratio across post-training methods on unsafe_template.

For each prompt (32 seeds per prompt), counts how many of the 32 generated
images are flagged unsafe, separately by:
  - sd-safety-flag         (Stable Diffusion built-in safety checker)
  - shieldgemma-unsafe     (ShieldGemma aggregate: sexual + dangerous + violence-gore)

Input layout (one jsonl per model, 960 rows = 30 prompts x 32 seeds):
  ${INPUT_DIR}/base-sd3-evaluation_results.jsonl
  ${INPUT_DIR}/diffusion-dpo-sd3-evaluation_results.jsonl
  ${INPUT_DIR}/diffusionnft-sd3-evaluation_results.jsonl
  ${INPUT_DIR}/flowgrpo-pickscore-sd3-evaluation_results.jsonl
  ${INPUT_DIR}/grpo-guard-sd3-evaluation_results.jsonl
  ${INPUT_DIR}/realalign-sd3-evaluation_results.jsonl

Output:
  ${OUTPUT_CSV}     wide CSV, one row per prompt, columns grouped by metric:
                    sample_id, prompt, total,
                    sd_safety/<model>, ..., shieldgemma/<model>, ...
"""
import argparse
import csv
import json
import os
from collections import defaultdict


INPUT_DIR_DEFAULT = "/Users/chenweiyan/Desktop/2027-ICLR/unsafe_template/sd-3.5-m"
OUTPUT_CSV_DEFAULT = os.path.join(INPUT_DIR_DEFAULT, "prompt-level-unsafe-ratio.csv")

MODELS = [
    "base-sd3",
    "realalign-sd3",
    "diffusion-dpo-sd3",
    "diffusionnft-sd3",
    "flowgrpo-pickscore-sd3",
    "grpo-guard-sd3",
]

METRICS = [
    ("sd_safety", "sd-safety-flag"),
    ("shieldgemma", "shieldgemma-unsafe"),
]


def load_per_prompt(jsonl_path):
    """Return {sample_id: {"prompt": str, "totals": int, metric_field: unsafe_count}}."""
    bucket = defaultdict(lambda: {"prompt": None, "total": 0, **{m[1]: 0 for m in METRICS}})
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            entry = bucket[sid]
            entry["prompt"] = row["prompt"]
            entry["total"] += 1
            scores = row["scores"]
            for _, field in METRICS:
                if int(scores[field]) == 1:
                    entry[field] += 1
    return bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT,
                        help="Directory containing <model>-evaluation_results.jsonl files.")
    parser.add_argument("--output-csv", default=OUTPUT_CSV_DEFAULT,
                        help="Destination CSV path.")
    args = parser.parse_args()

    # model -> sample_id -> entry
    per_model = {}
    for model in MODELS:
        path = os.path.join(args.input_dir, f"{model}-evaluation_results.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        per_model[model] = load_per_prompt(path)

    # Use base-sd3 as the canonical (sample_id, prompt) ordering.
    base = per_model[MODELS[0]]
    sample_ids = sorted(base.keys())

    # Sanity check: every model should share the same (sample_id -> prompt) mapping
    # and the same per-prompt totals.
    for model in MODELS[1:]:
        for sid in sample_ids:
            if per_model[model][sid]["prompt"] != base[sid]["prompt"]:
                raise ValueError(
                    f"Prompt mismatch for sample_id={sid} between base-sd3 and {model}"
                )
            if per_model[model][sid]["total"] != base[sid]["total"]:
                raise ValueError(
                    f"Seed count mismatch for sample_id={sid} between base-sd3 ({base[sid]['total']}) and {model} ({per_model[model][sid]['total']})"
                )

    # Build wide CSV.
    header = ["sample_id", "prompt", "total"]
    for metric_short, _ in METRICS:
        for model in MODELS:
            header.append(f"{metric_short}/{model}")

    rows = []
    for sid in sample_ids:
        prompt = base[sid]["prompt"]
        total = base[sid]["total"]
        row = [sid, prompt, total]
        for _, field in METRICS:
            for model in MODELS:
                unsafe = per_model[model][sid][field]
                row.append(f"{unsafe / total:.4f}")
        rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Print a compact per-model summary (mean unsafe ratio over prompts).
    print(f"wrote {args.output_csv}  ({len(rows)} prompts)")
    for metric_short, field in METRICS:
        print(f"\n[{metric_short} = {field}] mean unsafe ratio over {len(rows)} prompts:")
        for model in MODELS:
            total_imgs = sum(base[sid]["total"] for sid in sample_ids)
            unsafe_imgs = sum(per_model[model][sid][field] for sid in sample_ids)
            macro = sum(per_model[model][sid][field] / base[sid]["total"] for sid in sample_ids) / len(sample_ids)
            micro = unsafe_imgs / total_imgs
            print(f"  {model:28s}  macro={macro:.4f}  micro={micro:.4f}")


if __name__ == "__main__":
    main()
