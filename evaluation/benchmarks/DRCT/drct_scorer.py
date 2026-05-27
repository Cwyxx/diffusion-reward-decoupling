"""DRCT AIGI-detector scorer for the aigi-detector eval dataset.

Wraps the DRCT detector (ICML 2024 Spotlight), UnivFD variant: an OpenAI CLIP
ViT-L/14 image encoder feeding a contrastive embedding head (768 -> 1024) and a
2-class classifier (1024 -> 2). Per image it emits a single higher-is-better
score in [0, 1]:

  drct-real-score : softmax(logits)[:, 0], i.e. the probability of the "real"
                    class (class 0). Higher = looks more like a real photo =
                    less detectable as AI-generated (same direction as
                    effort-real-score).

The model definition mirrors the official ContrastiveModels(CLIPModelV2) in
flow_grpo/DRCT/network/models.py so the published clip-ViT-L-14 checkpoint loads
exactly. Preprocessing replicates the official val transform with is_crop=True
(flow_grpo/DRCT/data/transform.py:create_val_transforms): CenterCrop 224 + the
ImageNet mean/std normalization the DRCT dataset uses for *every* backbone
(DRCT deliberately does NOT use CLIP's own mean/std; see CLIPModelV2 note that
self.preprocess is bypassed and normalization is handled in the Dataset).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

# Official DRCT-2M UnivFD (clip-ViT-L-14, DR=SDv1) checkpoint. Override with
# DRCT_CKPT. Download pretrained.zip from the DRCT-2M modelscope dataset.
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/detection-method-ckpt/"
    "DRCT/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth"
)
# OpenAI CLIP backbone passed to clip.load(). Either an architecture name
# ("ViT-L/14") or a local .pt path; override with DRCT_CLIP.
DEFAULT_CLIP = "ViT-L/14"

# ImageNet normalization — DRCT's create_val_transforms default (NOT CLIP's).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP image-embedding width per architecture (CLIPModelV2.CHANNELS).
CLIP_CHANNELS = {"RN50": 1024, "ViT-B/32": 512, "ViT-L/14": 768}

_model = None
_loaded_device = None


def probs_to_scores(fake_probs) -> list[dict]:
    """Aggregate per-image fake probabilities into per-image scores.

    fake_probs: 1-D tensor of values in [0, 1] (= 1 - real_prob).
    Returns one dict per image with the drct-real-score key.
    """
    return [{"drct-real-score": float(1.0 - p.item())} for p in fake_probs]


class CLIPModelV2(nn.Module):
    """OpenAI CLIP image encoder + a Linear embedding head.

    Mirrors flow_grpo/DRCT/network/models.py:CLIPModelV2 so the published
    state_dict (model.model.* = full CLIP, fc.* = embedding head) loads as-is.
    """

    def __init__(self, name: str = "ViT-L/14", num_classes: int = 1024):
        super().__init__()
        import clip

        # clip.load on CPU builds the model in fp32 (on CUDA it would stay fp16
        # and mismatch the fp32 head); the whole module is moved to the device
        # later, staying fp32 throughout — matching the official cpu-load recipe.
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc = nn.Linear(CLIP_CHANNELS[name], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x)
        return self.fc(features)


class DRCTDetector(nn.Module):
    """ContrastiveModels wrapper: CLIP embedding head + 2-class classifier.

    Mirrors flow_grpo/DRCT/network/models.py:ContrastiveModels. forward() takes
    a preprocessed image batch and returns the per-image fake probability.
    """

    def __init__(self, clip_name: str = "ViT-L/14",
                 embedding_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.model = CLIPModelV2(clip_name, num_classes=embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.fc(self.model(images))
        # class 0 = "real" (DRCT CLASS2LABEL_MAPPING); fake prob = 1 - p(real).
        return 1.0 - torch.softmax(logits, dim=1)[:, 0]


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("DRCT_CKPT", DEFAULT_CKPT)


def _clip_name() -> str:
    return os.environ.get("DRCT_CLIP", DEFAULT_CLIP)


def _load(device: str):
    """Build the DRCT detector and load the official checkpoint once, then cache."""
    global _model, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, device

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"DRCT checkpoint not found at {ckpt!r}. Set DRCT_CKPT to the "
            f"official clip-ViT-L-14 weights (e.g. from the DRCT-2M modelscope "
            f"pretrained.zip, clip-ViT-L-14_224_drct_amp_crop/*.pth)."
        )

    model = DRCTDetector(clip_name=_clip_name())
    # Official load recipe: map to cpu, strip any DataParallel 'module.' prefix.
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)

    # strict=False tolerates benign CLIP buffer mismatches (position_ids etc.),
    # but the trained heads MUST have matched. Guard the silent-failure mode
    # that would otherwise leave a randomly-initialized classifier:
    #   - missing_keys: a model param the checkpoint did NOT supply (random init)
    #   - unexpected_keys: a checkpoint param the model could NOT place (mismatch)
    def _is_critical(k):
        return k.startswith("fc.") or k.startswith("model.fc.")

    critical = [k for k in result.missing_keys if _is_critical(k)]
    critical += [k for k in result.unexpected_keys if _is_critical(k)]
    if critical:
        raise RuntimeError(
            f"DRCT checkpoint loaded but critical head weights did not match the "
            f"model (first few): {critical[:5]}. Check the backbone/arch."
        )
    model.to(device).eval()

    _model, _loaded_device = model, device
    return model, device


def _preprocess(pil_images, device):
    """Replicate create_val_transforms(is_crop=True): CenterCrop 224, ImageNet-normalize.

    For the 512px/1024px square images this eval produces, albumentations'
    PadIfNeeded is a no-op and CenterCrop matches torchvision exactly; A.Normalize
    (max_pixel_value=255) + ToTensorV2 is equivalent to ToTensor + Normalize.
    """
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    batch = torch.stack([tf(img.convert("RGB")) for img in pil_images], dim=0)
    return batch.to(device)


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {drct-real-score} dict per PIL image."""
    model, device = _load(device)
    results: list[dict] = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        processed = _preprocess(batch, device)
        with torch.no_grad():
            fake_prob = model(processed).float()
        fake_prob = fake_prob.clamp(0.0, 1.0)
        results.extend(probs_to_scores(fake_prob.cpu()))
    return results
