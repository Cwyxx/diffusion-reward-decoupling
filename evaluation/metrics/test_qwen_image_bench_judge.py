import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from qwen_image_bench_judge import (
    DIM_KEY,
    build_tasks,
    load_and_resize_image,
    scores_from_raw,
)


def test_load_and_resize_keeps_small_and_shrinks_large(tmp_path):
    small = tmp_path / "s.png"
    Image.new("RGB", (512, 512), "red").save(small)
    assert load_and_resize_image(str(small)).size == (512, 512)

    big = tmp_path / "b.png"
    Image.new("RGB", (2048, 2048), "blue").save(big)
    assert load_and_resize_image(str(big)).size == (1024, 1024)


def test_build_tasks_one_per_applicable_l1():
    dims_en = ("Quality / Realism / Physical Logic; "
               "Aesthetics / Composition / Composition")
    img = Image.new("RGB", (64, 64))
    tasks = build_tasks("a cat", dims_en, img)
    l1s = [l1 for l1, _ in tasks]
    assert l1s == ["Quality", "Aesthetics"]
    # checklist text for the dim must be embedded in the user prompt
    quality_task = tasks[0][1]
    assert "Physical Logic" in quality_task["user_text"]
    assert quality_task["image"] is img


def test_scores_from_raw_maps_and_aggregates():
    # Quality: Realism has 2 facets (1->60, 2->100 => L2=80);
    #          Resolution has 1 facet (0->0 => L2=0). L1 = mean(80,0)=40.
    quality_raw = (
        '{"Realism": {"Physical Logic": {"score": 1}, '
        '"Material Texture": {"score": 2}}, '
        '"Resolution": {"Resolution": {"score": 0}}}'
    )
    # Aesthetics: single facet 2 -> 100 => L1 = 100.
    aesth_raw = '{"Composition": {"Composition": {"score": 2}}}'

    overall, dim_scores, parsed = scores_from_raw(
        {"Quality": quality_raw, "Aesthetics": aesth_raw}
    )
    assert dim_scores[DIM_KEY["Quality"]] == 40.0
    assert dim_scores[DIM_KEY["Aesthetics"]] == 100.0
    assert overall == 70.0          # mean(40, 100)
    assert "Quality" in parsed and "Aesthetics" in parsed


def test_scores_from_raw_unparseable_dim_skipped():
    overall, dim_scores, parsed = scores_from_raw(
        {"Quality": "not json at all"}
    )
    assert overall is None
    assert dim_scores == {}
    assert parsed == {}
