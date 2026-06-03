"""ms-swift TransformersEngine wrapper for the 27B Qwen-Image-Bench Q-Judger.

Vendored into the tracked tree (instead of editing the untracked upstream
flow_grpo/Qwen-Image-Bench/backends/ms_swift_backend.py) so all integration code
lives in this repo. Mirrors upstream's MsSwiftJudge but pins max_batch_size=1
(24 OOMs on 3090s) and device_map="auto" to shard the 27B across the 4x3090
cards. Imports `swift` (ms-swift>=4.0), only present in the qwen-image-bench
conda env, so this module is not imported at unit-test time.

Fixed decoding to match Qwen-Image-Bench: seed 42, temperature 0, top_k 1,
top_p 1.0, repetition_penalty 1.05, enable_thinking True.
"""
from swift import TransformersEngine, RequestConfig, InferRequest


class QwenImageBenchJudge:
    def __init__(self, model_path, max_batch_size=1, max_new_tokens=4096,
                 device_map="auto"):
        # device_map="auto" shards the 27B across all CUDA_VISIBLE_DEVICES.
        # Older ms-swift TransformersEngine may not accept device_map -> fall
        # back to its default placement.
        try:
            self.engine = TransformersEngine(
                model_path, max_batch_size=max_batch_size, device_map=device_map,
            )
        except TypeError:
            self.engine = TransformersEngine(model_path, max_batch_size=max_batch_size)
        self.request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )
        # Enable Qwen3 thinking mode on the engine's default template.
        try:
            self.engine.default_template.template_meta.template_kwargs = {
                "enable_thinking": True
            }
        except AttributeError:
            pass

    def generate_batch(self, items):
        """Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}.
        Returns list of generated text strings (one per item)."""
        infer_requests = []
        for item in items:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_text"]},
            ]
            infer_requests.append(
                InferRequest(messages=messages, images=[item["image"]])
            )
        resp_list = self.engine.infer(infer_requests, self.request_config)
        return [r.choices[0].message.content for r in resp_list]
