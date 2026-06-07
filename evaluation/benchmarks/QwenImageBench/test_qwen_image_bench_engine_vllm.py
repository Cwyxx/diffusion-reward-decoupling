import os
import sys

from PIL import Image

# repo root on sys.path so the benchmark package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from evaluation.benchmarks.QwenImageBench.qwen_image_bench_engine_vllm import (
    QwenImageBenchJudge,
    _build_messages,
    _image_to_data_url,
)


def _item():
    return {
        "system_prompt": "you are a judge",
        "user_text": "score this",
        "image": Image.new("RGB", (8, 8), "red"),
    }


def test_image_to_data_url_is_png_base64():
    url = _image_to_data_url(Image.new("RGB", (8, 8)))
    assert url.startswith("data:image/png;base64,")
    assert len(url) > len("data:image/png;base64,")


def test_build_messages_image_first_then_text():
    msgs = _build_messages(_item())
    assert msgs[0]["role"] == "system"
    user = msgs[1]
    assert user["role"] == "user"
    # image precedes the instruction text in the user content
    assert user["content"][0]["type"] == "image_url"
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][1] == {"type": "text", "text": "score this"}


def test_build_body_pins_bench_decoding():
    judge = QwenImageBenchJudge(model_path="Qwen-Image-Bench", max_new_tokens=1234)
    body = judge._build_body(_item())
    assert body["model"] == "Qwen-Image-Bench"
    assert body["max_tokens"] == 1234
    assert body["temperature"] == 0
    assert body["top_p"] == 1.0
    assert body["top_k"] == 1
    assert body["repetition_penalty"] == 1.05
    assert body["seed"] == 42
    # thinking is left to the server default -> no per-request override sent
    assert "chat_template_kwargs" not in body


def test_generate_batch_preserves_order_and_uses_post(monkeypatch):
    judge = QwenImageBenchJudge(max_batch_size=4)
    # stub the network call: echo each item's user_text so we can check ordering
    monkeypatch.setattr(judge, "_post_one", lambda item: "ANS:" + item["user_text"])
    items = [
        {"system_prompt": "s", "user_text": f"q{i}", "image": Image.new("RGB", (4, 4))}
        for i in range(5)
    ]
    out = judge.generate_batch(items)
    assert out == [f"ANS:q{i}" for i in range(5)]


def test_generate_batch_empty():
    assert QwenImageBenchJudge().generate_batch([]) == []


def test_url_built_from_base_url():
    judge = QwenImageBenchJudge(base_url="http://host:8000/v1/")
    assert judge.url == "http://host:8000/v1/chat/completions"
