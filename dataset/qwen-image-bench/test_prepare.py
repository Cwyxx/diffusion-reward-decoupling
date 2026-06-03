import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prepare import build_prompts


def test_build_prompts_extracts_en_and_sorts(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [
        {"ID": 2, "prompt_en": "a red cube", "prompt_cn": "红色立方体",
         "dims_en": "Quality / Realism / Physical Logic", "dims_cn": "x", "junk": 1},
        {"ID": 1, "prompt_en": "a blue sphere", "prompt_cn": "蓝色球",
         "dims_en": "Aesthetics / Composition / Composition", "dims_cn": "y"},
    ]
    with open(src, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = build_prompts(str(src))

    assert [r["ID"] for r in out] == [1, 2]           # sorted by ID
    assert out[0]["prompt"] == "a blue sphere"        # prompt == prompt_en
    assert out[0]["dims_en"] == "Aesthetics / Composition / Composition"
    assert set(out[0].keys()) == {"prompt", "ID", "dims_en"}  # only the 3 fields
