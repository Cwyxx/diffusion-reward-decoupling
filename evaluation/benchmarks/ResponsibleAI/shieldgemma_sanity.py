"""Sanity-check that the ShieldGemma-2 checkpoint actually loads and judges.

Background: transformers v4.52-v4.53 refactored VLM module layout but forgot
the ShieldGemma2 checkpoint key remapping (fixed upstream in PR #39348,
released in v4.54.0). On the broken versions from_pretrained silently leaves
most of the model (vision tower, projector, language model, lm_head) randomly
initialized -- producing exactly the pathology observed in our jsonls:
yes-probabilities smeared uniformly over [0,1], never confident, and nearly
identical across the three policies (per-image spread ~0.02). Random weights
also explain the bf16 all-NaN overflow that prompted the fp32 default.

Run on the scoring machine, in the same env used by run-eval.sh:
    conda activate visualquality
    python evaluation/benchmarks/ResponsibleAI/shieldgemma_sanity.py [--image PATH]

PASS requires BOTH:
  1. from_pretrained reports no missing/mismatched keys (checkpoint loaded);
  2. on a known-benign image every policy's yes-prob < 0.2 and the model is
     confidently low (a working ShieldGemma answers ~0.01-0.05 on the bee).
"""

import argparse
import io
import os
import sys

import requests
import torch

MODEL_ID = "google/shieldgemma-2-4b-it"
POLICIES = ("sexually-explicit", "dangerous", "violence-gore")
BENIGN_URL = (
    os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    + "/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
)


def load_model():
    from transformers import ShieldGemma2ForImageClassification

    # torch_dtype was renamed to dtype in transformers v5.
    for dtype_kw in ("torch_dtype", "dtype"):
        try:
            return ShieldGemma2ForImageClassification.from_pretrained(
                MODEL_ID, output_loading_info=True, **{dtype_kw: torch.float32}
            )
        except TypeError:
            continue
    raise RuntimeError("from_pretrained rejected both torch_dtype= and dtype=")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", default=None,
        help="Optional local benign image; default downloads the HF docs bee photo.",
    )
    args = parser.parse_args()

    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    print(f"transformers {transformers.__version__}  torch {torch.__version__}")
    print("NOTE: ShieldGemma2 checkpoint loading is broken in v4.52-v4.53 "
          "(fixed in v4.54.0 by PR #39348).\n")

    model, loading_info = load_model()
    model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    keys_ok = True
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        entries = loading_info.get(kind) or []
        print(f"{kind}: {len(entries)}")
        for e in entries[:10]:
            print(f"    {e}")
        if entries and kind != "unexpected_keys":
            keys_ok = False

    # The checkpoint stores no lm_head (tied to embed_tokens), and the wrapper's
    # tie_weights() override routes through language_model (no lm_head there
    # post-4.52 refactor), so lm_head stays randomly initialized -> confident
    # garbage (all yes-probs ~0.99). Re-tie explicitly; this makes a sole
    # 'model.lm_head.weight' missing key acceptable.
    inner = model.model  # Gemma3ForConditionalGeneration
    emb = inner.get_input_embeddings()
    if inner.lm_head.weight.data_ptr() != emb.weight.data_ptr():
        print("\nlm_head was NOT tied to embed_tokens (wrapper tie_weights bug); "
              "re-tying manually.")
        inner.lm_head.weight = emb.weight
    if (loading_info.get("missing_keys") or []) == ["model.lm_head.weight"]:
        keys_ok = True

    if not keys_ok:
        print("\n=> CHECKPOINT DID NOT LOAD: weights above stayed randomly "
              "initialized; every score produced by this env is noise.")

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"\nScoring local image: {args.image}")
    else:
        image = Image.open(io.BytesIO(requests.get(BENIGN_URL, timeout=60).content)).convert("RGB")
        print(f"\nScoring known-benign image: {BENIGN_URL}")

    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    probs = out.probabilities.detach().float().cpu()
    if probs.ndim == 3:
        probs = probs[0]
    yes = probs[:, 0].tolist()  # index 0 = P("Yes") = violation
    for name, p in zip(POLICIES, yes):
        print(f"  {name:18s} yes-prob = {p:.4f}")
    spread = max(yes) - min(yes)
    print(f"  policy spread = {spread:.4f}")

    probs_ok = max(yes) < 0.2
    print(f"\nkeys_ok={keys_ok}  benign_probs_ok={probs_ok}")
    if keys_ok and probs_ok:
        print("PASS: checkpoint loaded and the judge is confidently safe on a "
              "benign image.")
        return 0
    print("FAIL: fix the env (pip install -U 'transformers>=4.54,<5' in "
          "visualquality), rerun this script, then re-score every "
          "shieldgemma_fp32 jsonl with score-images.py --force.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
