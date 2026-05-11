"""One-time: extract unique (item_id, prompt) pairs from dpg_bench.csv
into dataset/dpg_bench/prompts.jsonl for use by generate-images-bestofn.py.

prompt_id = position in first-appearance order from the CSV (deterministic).
item_id   = original DPG string id, used by score-images.py to look up questions.
"""
import json
import os
import os.path as osp

import pandas as pd

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", "..", ".."))
CSV_PATH = osp.join(_THIS_DIR, "dpg_bench.csv")
OUT_DIR = osp.join(_REPO_ROOT, "dataset", "dpg_bench")
OUT_PATH = osp.join(OUT_DIR, "prompts.jsonl")


def main():
    df = pd.read_csv(CSV_PATH)
    seen = {}
    order = []
    for _, row in df.iterrows():
        iid = row.item_id
        if iid not in seen:
            seen[iid] = row.text
            order.append(iid)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for pid, iid in enumerate(order):
            f.write(json.dumps({
                "prompt_id": pid,
                "item_id":   iid,
                "prompt":    seen[iid],
            }) + "\n")

    print(f"Wrote {len(order)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
