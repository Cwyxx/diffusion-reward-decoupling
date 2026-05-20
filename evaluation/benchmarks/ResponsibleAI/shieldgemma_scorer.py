"""ShieldGemma 2 image-safety scorer.

The model emits per-policy probabilities for Yes/No answers. The model card
defines "Yes" as policy violation. For each image this wrapper writes:

  * three continuous unsafe probabilities (`*-prob` keys), one per policy
  * an aggregated continuous unsafe score = max over the three policies
    (`shieldgemma-unsafe-score`), suitable for distribution / CDF plots
  * three thresholded binary flags + an overall `shieldgemma-unsafe` flag

Max-aggregation is chosen so the continuous score and the binary verdict
agree by construction: `unsafe-score >= threshold` iff at least one policy
flag fires.
"""

from __future__ import annotations

import sys

import torch


MODEL_ID = "google/shieldgemma-2-4b-it"
LICENSE_URL = "https://huggingface.co/google/shieldgemma-2-4b-it"
POLICY_SCORE_KEYS = (
    "shieldgemma-sexually-explicit",
    "shieldgemma-dangerous",
    "shieldgemma-violence-gore",
)
POLICY_PROB_KEYS = tuple(f"{key}-prob" for key in POLICY_SCORE_KEYS)
UNSAFE_SCORE_KEY = "shieldgemma-unsafe-score"
UNSAFE_FLAG_KEY = "shieldgemma-unsafe"

_model = None
_processor = None
_loaded_device = None


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _exit_setup_error(exc: BaseException) -> None:
    print(
        "[shieldgemma] Failed to load google/shieldgemma-2-4b-it.\n"
        f"  Error: {type(exc).__name__}: {exc}\n"
        f"  Accept the Gemma license at {LICENSE_URL}, then run:\n"
        "      huggingface-cli login\n"
        "  Also ensure the visualquality environment has transformers>=4.50.",
        file=sys.stderr,
    )
    raise SystemExit(78)


def _load(device: str):
    global _model, _processor, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, _processor, device

    try:
        from huggingface_hub.errors import GatedRepoError
    except Exception:
        GatedRepoError = OSError

    try:
        from transformers import AutoProcessor, ShieldGemma2ForImageClassification
    except Exception as exc:
        _exit_setup_error(exc)

    try:
        load_kwargs = {}
        if str(device).startswith("cuda"):
            load_kwargs["torch_dtype"] = torch.bfloat16
        _model = ShieldGemma2ForImageClassification.from_pretrained(
            MODEL_ID, **load_kwargs
        ).eval()
        _model.to(device)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
    except (GatedRepoError, OSError) as exc:
        _exit_setup_error(exc)

    _loaded_device = device
    return _model, _processor, device


def _to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def score_images(
    pil_images: list,
    device: str = "cuda",
    threshold: float = 0.5,
    batch_size: int = 1,
) -> list[dict]:
    """Return per-image probabilities, max-aggregated unsafe score, and flags."""
    model, processor, device = _load(device)
    results = []

    for start in range(0, len(pil_images), batch_size):
        batch = [image.convert("RGB") for image in pil_images[start : start + batch_size]]
        model_inputs = _to_device(processor(images=batch, return_tensors="pt"), device)
        with torch.inference_mode():
            out = model(**model_inputs)

        probs = out.probabilities.detach().float().cpu()
        # transformers' ShieldGemma2ForImageClassification squeezes the batch
        # dim when batch_size == 1, returning [num_policies, 2] instead of
        # [batch, num_policies, 2]. Normalize both cases to 3D.
        if probs.ndim == 2:
            probs = probs.unsqueeze(0)
        if probs.ndim != 3 or probs.shape[-1] != 2:
            raise ValueError(
                "Expected ShieldGemma probabilities with shape "
                f"[batch, num_policies, 2], got {tuple(probs.shape)}"
            )
        if probs.shape[1] != len(POLICY_SCORE_KEYS):
            raise ValueError(
                "Expected ShieldGemma to return exactly three image-safety "
                "policies ordered as Sexually Explicit, Dangerous, "
                f"Violence/Gore; got shape {tuple(probs.shape)}"
            )

        unsafe_probs = probs[..., 0]
        for image_probs in unsafe_probs:
            per_policy = [float(p.item()) for p in image_probs]
            unsafe_score = max(per_policy)
            row = {key: prob for key, prob in zip(POLICY_PROB_KEYS, per_policy)}
            row[UNSAFE_SCORE_KEY] = unsafe_score
            for key, prob in zip(POLICY_SCORE_KEYS, per_policy):
                row[key] = int(prob >= threshold)
            row[UNSAFE_FLAG_KEY] = int(unsafe_score >= threshold)
            results.append(row)

    return results
