import importlib.util
import os

import numpy as np

_AGG = os.path.join(os.path.dirname(__file__), "aggregate-bestofn.py")
_spec = importlib.util.spec_from_file_location("agg_bestofn", _AGG)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_bon_select_picks_overall_argmax_and_reads_companion_dim():
    # 1 prompt, 3 seeds. overall maxes at seed1 from n>=2.
    overall = np.array([[10.0, 50.0, 30.0]])
    dim = np.array([[100.0, 0.0, 40.0]])
    # n=1 -> only seed0 -> dim 100
    assert agg.bon_select(overall, dim, 1) == 100.0
    # n=2 -> argmax overall in {10,50}=seed1 -> dim 0  (non-monotonic drop)
    assert agg.bon_select(overall, dim, 2) == 0.0
    # n=3 -> argmax overall still seed1 -> dim 0
    assert agg.bon_select(overall, dim, 3) == 0.0


def test_total_curve_equals_bon_continuous_of_overall():
    overall = np.array([[10.0, 50.0, 30.0], [70.0, 20.0, 60.0]])
    for n in (1, 2, 3):
        # selecting overall's companion == overall itself equals max-over-n
        assert agg.bon_select(overall, overall, n) == agg.bon_continuous(overall, n)


def _row(sid, seed, scores):
    return {"sample_id": sid, "seed_index": seed, "prompt": f"p{sid}",
            "image_path": f"/img/{sid}_{seed}.png", "metadata": {"ID": sid},
            "scores": scores}


def test_aggregate_only_averages_dim_over_covering_prompts(tmp_path):
    # prompt 0 covers quality+alignment; prompt 1 covers only quality.
    rows = [
        _row(0, 0, {"qwen-image-bench": 40.0, "qwen-image-bench-quality": 60.0,
                    "qwen-image-bench-alignment": 0.0}),
        _row(0, 1, {"qwen-image-bench": 80.0, "qwen-image-bench-quality": 100.0,
                    "qwen-image-bench-alignment": 60.0}),
        _row(1, 0, {"qwen-image-bench": 60.0, "qwen-image-bench-quality": 60.0}),
        _row(1, 1, {"qwen-image-bench": 20.0, "qwen-image-bench-quality": 20.0}),
    ]
    bestofn = tmp_path / "bestofn"
    plots = bestofn / "plots"
    csvd = bestofn / "csv"
    for d in (plots, csvd):
        os.makedirs(d, exist_ok=True)

    out = agg._aggregate_qwen_image_bench(rows, str(bestofn), str(plots), str(csvd))

    # Total at n=2: prompt0 max(40,80)=80; prompt1 max(60,20)=60 -> mean 70.
    assert out["qwen-image-bench"]["curve"][2] == 70.0
    # quality at n=2: prompt0 winner=seed1(overall80)->100; prompt1 winner=seed0(60)->60 -> mean 80.
    assert out["qwen-image-bench-quality"]["curve"][2] == 80.0
    # alignment only prompt0 covers it: winner seed1 -> 60. mean over 1 prompt = 60.
    assert out["qwen-image-bench-alignment"]["curve"][2] == 60.0
    assert out["qwen-image-bench-alignment"]["num_prompts"] == 1
