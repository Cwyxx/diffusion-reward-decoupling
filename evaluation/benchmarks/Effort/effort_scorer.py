"""Effort AIGI-detector scorer for the aigi-detector eval dataset.

Wraps the official Effort detector (ICML 2025): a CLIP ViT-L/14 vision tower
with an orthogonal SVD-residual adapter on every self_attn linear layer plus a
2-class head. Per image it emits a single higher-is-better score in [0, 1]:

  effort-real-score : 1 - fake_prob, where
                      fake_prob = softmax(head(pooled_feature))[:, 1].

Preprocessing and the model definition replicate the official demo
(flow_grpo/Effort-AIGI-Detection/DeepfakeBench/training/demo.py and
detectors/effort_detector.py), including CLIP mean/std normalization, so the
official GenImage(sdv1.4) checkpoint loads and behaves identically.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Official Effort GenImage(sdv1.4) checkpoint (already present on the server).
DEFAULT_CKPT = (
    "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/Effort/"
    "effort_clip_L14_trainOn_sdv14.pth"
)
# CLIP ViT-L/14 backbone (architecture + pretrained encoder). Fetched via the
# HF_ENDPOINT mirror already exported by the run scripts; override with EFFORT_CLIP.
DEFAULT_BACKBONE = "openai/clip-vit-large-patch14"

# CLIP image normalization (must match training; see effort.yaml mean/std).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_model = None
_loaded_device = None


def probs_to_scores(fake_probs) -> list[dict]:
    """Aggregate per-image fake probabilities into per-image scores.

    fake_probs: 1-D tensor of values in [0, 1].
    Returns one dict per image with the effort-real-score key.
    """
    return [{"effort-real-score": float(1.0 - p.item())} for p in fake_probs]


class SVDResidualLinear(nn.Module):
    """nn.Linear whose weight = frozen top-r SVD reconstruction + trainable residual."""

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to keep frozen

        # Original (frozen) main weight
        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def apply_svd_residual_to_self_attn(model, r):
    """Replace every nn.Linear inside any self_attn submodule with SVDResidualLinear."""
    for name, module in model.named_children():
        if 'self_attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        replace_with_svd_residual(sub_module, r),
                    )
        else:
            apply_svd_residual_to_self_attn(module, r)
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def replace_with_svd_residual(module, r):
    """Build an SVDResidualLinear from an nn.Linear via SVD of its weight."""
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        new_module = SVDResidualLinear(
            in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone()
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
        r = min(r, len(S))  # do not exceed the number of singular values

        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        new_module.weight_main.data.copy_(weight_main)

        U_residual = U[:, r:]
        S_residual = S[r:]
        Vh_residual = Vh[r:, :]

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())

            new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
            new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
            new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None

        return new_module
    else:
        return module


class EffortDetector(nn.Module):
    """CLIP ViT-L/14 vision tower + SVD-residual adapter + 2-class head.

    Parameter names mirror the official EffortDetector so the published
    checkpoint loads exactly. forward() takes a preprocessed image batch and
    returns the per-image fake probability.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = self._build_backbone(backbone_name)
        self.head = nn.Linear(1024, 2)

    def _build_backbone(self, backbone_name: str):
        from transformers import CLIPModel

        clip_model = CLIPModel.from_pretrained(backbone_name)
        # SVD-residual on self_attn only; ViT-L/14 keeps the top 1024-1 components.
        clip_model.vision_model = apply_svd_residual_to_self_attn(
            clip_model.vision_model, r=1024 - 1
        )
        return clip_model.vision_model

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)["pooler_output"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(images))
        return torch.softmax(logits, dim=1)[:, 1]  # fake probability


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _ckpt_path() -> str:
    return os.environ.get("EFFORT_CKPT", DEFAULT_CKPT)


def _backbone_name() -> str:
    return os.environ.get("EFFORT_CLIP", DEFAULT_BACKBONE)


def _load(device: str):
    """Build the Effort detector and load the official checkpoint once, then cache."""
    global _model, _loaded_device
    device = _resolve_device(device)
    if _model is not None and _loaded_device == device:
        return _model, device

    ckpt = _ckpt_path()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Effort checkpoint not found at {ckpt!r}. Set EFFORT_CKPT to the "
            f"official GenImage(sdv1.4) weights (effort_clip_L14_trainOn_sdv14.pth)."
        )

    model = EffortDetector(_backbone_name())
    # Official load recipe: unwrap state_dict, strip DataParallel 'module.' prefix.
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    # strict=False tolerates benign mismatches (e.g. position_ids), but the
    # trained head + SVD residuals MUST have matched the model. Guard the exact
    # silent-failure mode that would otherwise yield a randomly-initialized head.
    critical_unloaded = [
        k for k in state
        if (k.startswith("head.") or k.endswith("_residual"))
        and k in result.unexpected_keys
    ]
    if critical_unloaded:
        raise RuntimeError(
            f"Effort checkpoint loaded but critical weights did not match the "
            f"model (first few): {critical_unloaded[:5]}. Check the backbone/arch."
        )
    model.to(device).eval()

    _model, _loaded_device = model, device
    return model, device


def _preprocess(pil_images, device):
    """Replicate the official demo: resize 224, RGB, ToTensor, CLIP-normalize."""
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    batch = torch.stack([tf(img.convert("RGB")) for img in pil_images], dim=0)
    return batch.to(device)


def score_images(pil_images: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {effort-real-score} dict per PIL image."""
    model, device = _load(device)
    results: list[dict] = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        processed = _preprocess(batch, device)
        with torch.no_grad():
            fake_prob = model(processed)
        assert fake_prob.min() >= 0 and fake_prob.max() <= 1
        results.extend(probs_to_scores(fake_prob.cpu()))
    return results
