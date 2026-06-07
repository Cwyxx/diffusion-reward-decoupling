"""vLLM-server client for the 27B Qwen-Image-Bench Q-Judger.

QwenImageBenchJudge.generate_batch(items) talks to an OpenAI-compatible vLLM
server (`vllm serve Qwen/Qwen-Image-Bench`) instead of loading the 27B in-process.
vLLM gives real tensor-parallel + paged attention + continuous batching, so all
GPUs run at once. This module is just an HTTP client and uses no GPU itself.

Uses only the stdlib (urllib + json + base64) so it adds no dependency to the
scoring conda env. Decoding is pinned to match the bench: temperature 0, top_k 1,
top_p 1.0, repetition_penalty 1.05, seed 42. Thinking mode is NOT overridden
per-request — it follows the server's chat-template default (set it on the server
with --default-chat-template-kwargs '{"enable_thinking": false}' if desired).
"""
import base64
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _image_to_data_url(image):
    """PIL.Image -> 'data:image/png;base64,...' (PNG = lossless, no JPEG artifacts
    that would skew the Quality dimension)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_messages(item):
    """Item {"system_prompt","user_text","image"} -> OpenAI chat messages.

    Image goes first in the user content (conventional VL ordering), followed by
    the checklist instruction text."""
    return [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": _image_to_data_url(item["image"])}},
            {"type": "text", "text": item["user_text"]},
        ]},
    ]


class QwenImageBenchJudge:
    def __init__(self, model_path="Qwen-Image-Bench",
                 base_url="http://localhost:8000/v1", api_key="EMPTY",
                 max_batch_size=8, max_new_tokens=4096,
                 timeout=600, max_retries=3):
        # model_path here is the vLLM --served-model-name (NOT a checkpoint dir).
        self.model = model_path
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        # client-side concurrency; the real batching happens server-side via
        # --max-num-seqs, this just keeps that many requests in flight.
        self.max_batch_size = max(1, max_batch_size)
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def _build_body(self, item):
        return {
            "model": self.model,
            "messages": _build_messages(item),
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,                      # vLLM extension
            "repetition_penalty": 1.05,      # vLLM extension
            "seed": 42,
        }

    def _post_one(self, item):
        body = json.dumps(self._build_body(item)).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self.url, data=body, headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError,
                    KeyError, json.JSONDecodeError) as e:
                last_err = e
        raise RuntimeError(
            f"vLLM judge request failed after {self.max_retries} attempts: "
            f"{last_err}"
        )

    def generate_batch(self, items):
        """Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}.
        Returns list of generated text strings (one per item), order preserved."""
        if not items:
            return []
        workers = min(self.max_batch_size, len(items))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._post_one, items))
