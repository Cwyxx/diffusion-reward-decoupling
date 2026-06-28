"""Build dataset/qwen-image-bench-creative/prompts.jsonl from the full
Qwen-Image-Bench prompts file (dataset/qwen-image-bench/prompts.jsonl).

Creative-Generation-only variant: keep only the prompts that cover the
"Creative Generation" L1 dimension, and trim each kept row's dims_en down to
its "Creative Generation / ..." segments. This restricts BOTH the generation
prompt set AND the 27B judge (which builds one inference task per L1 dim in
dims_en) to Creative Generation, with no change to the judging code.

Each output row: {"prompt": <prompt_en>, "ID": int, "dims_en": <creative-only>},
sorted by ID.
"""
import json
import os

SRC_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "qwen-image-bench", "prompts.jsonl"
))
OUT_PATH = os.path.join(os.path.dirname(__file__), "prompts.jsonl")

CREATIVE_L1 = "Creative Generation"


def creative_only_dims(dims_en_str):
    """Return the '; '-joined dims_en segments whose L1 dim is Creative Generation.

    A segment looks like 'Creative Generation / Text Rendering / Text Accuracy';
    the L1 dim is the first '/'-separated part. Returns '' if none match.
    """
    kept = []
    for seg in dims_en_str.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        l1 = seg.split('/')[0].strip()
        if l1 == CREATIVE_L1:
            kept.append(seg)
    return "; ".join(kept)


def build_prompts(src_path):
    """Read the full prompts.jsonl, keep Creative rows with trimmed dims_en."""
    out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            creative_dims = creative_only_dims(r["dims_en"])
            if not creative_dims:
                continue
            out.append({
                "prompt": r["prompt"],
                "ID": int(r["ID"]),
                "dims_en": creative_dims,
            })
    out.sort(key=lambda x: x["ID"])
    return out


def main():
    rows = build_prompts(SRC_PATH)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
