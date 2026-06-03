"""Build dataset/qwen-image-bench/prompts.jsonl from the local Qwen-Image-Bench
prompts file (flow_grpo/Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl).

We use the ENGLISH prompt (prompt_en) + English dimension spec (dims_en), since
the Best-of-N pipeline targets English text-to-image models (SD-v1.5/SDXL/SD-3.5-M).
Each output row: {"prompt": <prompt_en>, "ID": int, "dims_en": str}, sorted by ID.
"""
import json
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SRC = os.path.join(
    _REPO_ROOT, "flow_grpo", "Qwen-Image-Bench", "qwen_image_bench_hf_v0518.jsonl"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "prompts.jsonl")


def build_prompts(src_path):
    """Read the source jsonl, return list of {prompt, ID, dims_en} sorted by ID."""
    out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "prompt": r["prompt_en"],
                "ID": int(r["ID"]),
                "dims_en": r["dims_en"],
            })
    out.sort(key=lambda x: x["ID"])
    return out


def main():
    rows = build_prompts(DEFAULT_SRC)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
