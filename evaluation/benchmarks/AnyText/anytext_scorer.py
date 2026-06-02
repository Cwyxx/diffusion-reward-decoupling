"""Position-free OCR scorer for the AnyText-Benchmark datasets (anytext-en / anytext-zh).

AnyText's own eval (``flow_grpo/AnyText/eval/eval_dgocr.py``) crops each text region at the
ground-truth polygon and runs recognition there. That assumes a *position-controlled* generator
(AnyText / TextDiffuser / GlyphControl, which are fed a glyph/position map). A plain text-to-image
model (SD-3.5-M, FLUX, ...) is only told *what* text to render, never *where*, so cropping at the
GT polygon would score the wrong region and collapse every model to ~0 for position reasons rather
than legibility.

So here we score **position-free**:

  generated image
    -> DuGuang text DETECTION  (ModelScope Tasks.ocr_detection)   -- find every text box
    -> per box: crop + rectify (reused AnyText geometry)          -- straighten the line
    -> DuGuang RECOGNITION      (cv_convnextTiny ...general)       -- read the line
    -> set of detected strings
    -> order-independent match against metadata['texts'] (the GT lines)

Per image (so it plugs into best-of-N's max-over-N curves):

  anytext-senacc : (# GT lines whose normalized form exactly equals some detected string)
                   / (# GT lines)                                  -- hard exact-match recall
  anytext-ned    : mean over GT lines of  max over detected strings of (1 - lev/maxlen)
                   -- soft, edit-distance recall; 0 if nothing was detected

Both are higher-is-better in [0, 1]. The recognition step + the crop/rectify/pre_process geometry
are reused verbatim from AnyText (see attribution on each helper); only the *source* of the boxes
changes (full-image detection instead of GT polygons), which is the whole point.

Engine = DuGuang (ModelScope), chosen over PaddleOCR for recognition quality / consistency with the
vendored AnyText code. Runs in the dedicated ``anytext`` conda env (modelscope / easydict / torch
2.0.x). Model ids are overridable via env vars so the exact ModelScope snapshot can be pinned on the
server during smoke testing:

  ANYTEXT_DET_MODEL  (default damo/cv_resnet18_ocr-detection-line-level_damo)
  ANYTEXT_REC_MODEL  (default damo/cv_convnextTiny_ocr-recognition-general_damo)
  ANYTEXT_OCR_DEVICE (default gpu; set cpu to force CPU)
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import Levenshtein
from skimage.transform._geometric import _umeyama as get_sym_mat

# ----------------------------------------------------------------------------------------------
# Geometry helpers — copied verbatim from flow_grpo/AnyText/cldm/recognizer.py (min_bounding_rect,
# adjust_image, crop_image). Copied rather than imported because recognizer.py does
# `from ocr_recog.RecModel import RecModel` at module load, which drags in the whole recognition
# backbone + sys.path hacks we don't need here (we use the ModelScope recognition pipeline instead).
# ----------------------------------------------------------------------------------------------

def min_bounding_rect(img):
    """flow_grpo/AnyText/cldm/recognizer.py:19 — min-area rotated rect of the largest contour."""
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('Bad contours, using fake bbox...')
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # sort tl, tr, br, bl
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])


def adjust_image(box, img):
    """flow_grpo/AnyText/cldm/recognizer.py:44 — perspective-warp the rotated box to upright."""
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
    height = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    result = result[:, :int(height), :int(width)]
    return result


def crop_image(src_img, mask):
    """flow_grpo/AnyText/cldm/recognizer.py:67 — mask (HWC uint8) -> rectified CHW line tensor."""
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result] * 3, axis=-1)
    return result


def pre_process(img_list, shape):
    """flow_grpo/AnyText/eval/eval_dgocr.py:55 — rotate-if-tall, resize to (imgH, w<=imgW), pad."""
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = F.interpolate(
            img.unsqueeze(0), size=(imgH, resized_w), mode='bilinear', align_corners=True,
        )
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC numpy
    return numpy_list


# ----------------------------------------------------------------------------------------------
# DuGuang detection + recognition pipelines (lazy, cached per process).
# ----------------------------------------------------------------------------------------------

REC_IMAGE_SHAPE = "3, 48, 320"
DEFAULT_DET_MODEL = "damo/cv_resnet18_ocr-detection-line-level_damo"
DEFAULT_REC_MODEL = "damo/cv_convnextTiny_ocr-recognition-general_damo"

_detector = None
_recognizer = None


def _ocr_device(device: str) -> str:
    """ModelScope wants 'gpu'/'cpu'. Honor ANYTEXT_OCR_DEVICE, then the passed device, then CUDA."""
    forced = os.environ.get("ANYTEXT_OCR_DEVICE")
    if forced:
        return forced
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return "gpu"
    return "cpu"


def _load(device: str):
    global _detector, _recognizer
    if _detector is not None and _recognizer is not None:
        return _detector, _recognizer
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    ms_device = _ocr_device(device)
    det_model = os.environ.get("ANYTEXT_DET_MODEL", DEFAULT_DET_MODEL)
    rec_model = os.environ.get("ANYTEXT_REC_MODEL", DEFAULT_REC_MODEL)
    _detector = pipeline(Tasks.ocr_detection, model=det_model, device=ms_device)
    _recognizer = pipeline(Tasks.ocr_recognition, model=rec_model, device=ms_device)
    return _detector, _recognizer


# ----------------------------------------------------------------------------------------------
# Detection -> crops -> recognition -> set of strings.
# ----------------------------------------------------------------------------------------------

def _detect_polygons(detector, rgb_img: np.ndarray) -> list[np.ndarray]:
    """Run DuGuang detection on an RGB image; return a list of (k, 2) polygon vertex arrays."""
    out = detector(rgb_img)
    polys = out.get("polygons", out.get("det_polygons", []))
    polys = np.asarray(polys)
    if polys.size == 0:
        return []
    result = []
    for row in polys:
        row = np.asarray(row, dtype=np.float32).reshape(-1, 2)  # (8,)->(4,2), or (k,2) as-is
        if len(row) >= 3:  # need >=3 points to fill a polygon
            result.append(row)
    return result


def _recognize_strings(image_bgr_chw: torch.Tensor, polygons: list[np.ndarray], recognizer) -> list[str]:
    """Crop+rectify each detected polygon from the BGR CHW source, recognize, return strings."""
    if not polygons:
        return []
    _, H, W = image_bgr_chw.shape
    crops = []
    for poly in polygons:
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=255)
        crops.append(crop_image(image_bgr_chw, mask))
    crops = pre_process(crops, REC_IMAGE_SHAPE)
    texts = []
    for pt in crops:
        rst = recognizer(pt)
        texts.append(rst["text"][0] if rst.get("text") else "")
    return texts


# ----------------------------------------------------------------------------------------------
# Matching GT lines against detected strings.
# ----------------------------------------------------------------------------------------------

def _normalize(s: str, lang: str) -> str:
    s = "".join(s.split())  # drop all whitespace
    if lang == "en":
        s = s.lower()       # English OCR is case-insensitive here (mirrors flow_grpo/ocr.py)
    return s


def _ned(a: str, b: str) -> float:
    """1 - levenshtein/max_len, clamped to [0, 1]; eval_dgocr.py:50 form."""
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / (max(len(a), len(b)) + 1e-5)


def _score_one(gt_texts: list[str], detected: list[str], lang: str) -> dict:
    """Order-independent per-image Sen.ACC + NED against the detected string set."""
    if len(gt_texts) == 0:
        # No GT text lines: nothing to read. AnyText excludes these from the metric entirely;
        # the anytext benchmark has none, but stay safe and emit a neutral 1.0.
        return {"anytext-senacc": 1.0, "anytext-ned": 1.0}
    det_norm = [_normalize(d, lang) for d in detected]
    det_set = set(det_norm)
    hits = 0
    ned_sum = 0.0
    for gt in gt_texts:
        g = _normalize(gt, lang)
        if g in det_set:
            hits += 1
        ned_sum += max((_ned(g, d) for d in det_norm), default=0.0)
    n = len(gt_texts)
    return {"anytext-senacc": hits / n, "anytext-ned": ned_sum / n}


# ----------------------------------------------------------------------------------------------
# Public entry point (matches the framework's score_images contract; +metadatas for GT texts).
# ----------------------------------------------------------------------------------------------

def score_images(pil_images: list, metadatas: list, device: str = "cuda", batch_size: int = 8) -> list[dict]:
    """Return one {anytext-senacc, anytext-ned} dict per PIL image.

    metadatas[i] carries the GT for image i: keys 'texts' (list[str]) and 'lang' ('en'|'zh').
    Handles the _load_jsonl nesting convention where the whole row became metadata, so the real
    fields may sit under metadata['metadata'].
    """
    assert len(pil_images) == len(metadatas), "images/metadatas length mismatch"
    detector, recognizer = _load(device)
    results: list[dict] = []
    for img, meta in zip(pil_images, metadatas):
        meta = meta or {}
        if isinstance(meta.get("metadata"), dict):  # nested-metadata unwrap (see _load_jsonl)
            meta = meta["metadata"]
        gt_texts = meta.get("texts", []) or []
        lang = meta.get("lang", "en")

        rgb = np.array(img.convert("RGB"))                  # RGB HWC uint8 -> detection input
        bgr = rgb[:, :, ::-1].copy()                        # BGR to match cv2.imread (recognition)
        bgr_chw = torch.from_numpy(bgr).permute(2, 0, 1).float()  # CHW float, 0-255

        polygons = _detect_polygons(detector, rgb)
        detected = _recognize_strings(bgr_chw, polygons, recognizer)
        results.append(_score_one(gt_texts, detected, lang))
    return results
