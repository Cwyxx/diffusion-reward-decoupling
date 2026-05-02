"""Local GenEval scorer — drops the HTTP reward-server dependency.

Algorithm copied verbatim from the cloned official repo at
``flow_grpo/geneval-official/evaluation/evaluate_images.py`` so the
binary ``correct`` flag this returns matches the headline number reported
in the GenEval paper. Used by ``evaluation/metrics/score-images.py``.

Models are lazy-loaded as a process-wide singleton on the first call to
``score(...)``; subsequent calls reuse them.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


# ---- Paths ----
# Both the config .py and the .pth checkpoint live in the mmdet
# mask2former config dir: <mmdet>/../configs/mask2former/. Set to None to
# derive that path from ``mmdet.__file__`` at load time, matching the
# official evaluate_images.py convention.
MASK2FORMER_CKPT_DIR = None
MMDET_CONFIG_PATH = None

OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
CLIP_ARCH = "ViT-L-14"

# Hyperparameters — match evaluate_images.py:286-290.
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1

COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODELS = None  # (object_detector, clip_model, transform, tokenizer, classnames)
_COLOR_CLASSIFIERS = {}


def _load_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    import mmdet
    from mmdet.apis import init_detector
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc
    zsc.tqdm = lambda it, *args, **kwargs: it  # silence inner progress bar

    mask2former_dir = os.path.join(os.path.dirname(mmdet.__file__), "../configs/mask2former")
    config_path = MMDET_CONFIG_PATH or os.path.join(mask2former_dir, f"{OBJECT_DETECTOR}.py")
    ckpt_dir = MASK2FORMER_CKPT_DIR or mask2former_dir
    ckpt_path = os.path.join(ckpt_dir, f"{OBJECT_DETECTOR}.pth")
    object_detector = init_detector(config_path, ckpt_path, device=DEVICE)

    clip_model, _, transform = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained="openai", device=DEVICE,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_ARCH)

    object_names_file = os.path.join(
        os.path.dirname(__file__),
        "geneval-official", "evaluation", "object_names.txt",
    )
    with open(object_names_file) as f:
        classnames = [line.strip() for line in f]

    _MODELS = (object_detector, clip_model, transform, tokenizer, classnames)
    return _MODELS


# ---- Begin verbatim port from evaluate_images.py:81-220 ----

class _ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image, objects, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999")
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index, self._image.size[::-1], mask.shape,
            )
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self._transform(image), 0)


def _color_classification(image, bboxes, classname):
    from clip_benchmark.metrics import zeroshot_classification as zsc
    _, clip_model, transform, tokenizer, _ = _load_models()
    if classname not in _COLOR_CLASSIFIERS:
        _COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ],
            DEVICE,
        )
    clf = _COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        _ImageCrops(image, bboxes, transform),
        batch_size=16, num_workers=4,
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[i.item()] for i in pred.argmax(1)]


def _compute_iou(box_a, box_b):
    area_fn = lambda b: max(b[2] - b[0] + 1, 0) * max(b[3] - b[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def _relative_position(obj_a, obj_b):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def _evaluate(image, objects, metadata):
    """Official loose evaluator. Returns (correct, reason)."""
    correct = True
    reason = []
    matched_groups = []
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                colors = _color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = _relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


# ---- End verbatim port ----


def _detect(image_pil, metadata):
    """Run Mask2Former + threshold/NMS, return {classname: [(box, mask)]}."""
    from mmdet.apis import inference_detector
    object_detector, _, _, _, classnames = _load_models()
    result = inference_detector(object_detector, np.array(image_pil))
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    detected = {}
    confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
        ordering = ordering[:MAX_OBJECTS].tolist()
        per_class = []
        while ordering:
            max_obj = ordering.pop(0)
            per_class.append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1
                or _compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if per_class:
            detected[classname] = per_class
    return detected


def score(image_paths, metadatas):
    """Score a batch of (image, metadata) pairs against the official rules.

    Args:
        image_paths: list of image file paths.
        metadatas: list of dicts; each must have 'tag', 'include', and may
            have 'exclude'. These are exact rows from
            ``dataset/geneval/test_metadata.jsonl``.

    Returns:
        List[int] of 0/1, one per image — the official loose ``correct``.
    """
    assert len(image_paths) == len(metadatas), (len(image_paths), len(metadatas))
    _load_models()  # eager so the tqdm bar is honest
    out = []
    for path, meta in tqdm(list(zip(image_paths, metadatas)), desc="geneval"):
        if not meta or "tag" not in meta:
            raise ValueError(f"GenEval needs metadata with 'tag'/'include'; got {meta!r} for {path}")
        image_pil = ImageOps.exif_transpose(Image.open(path))
        detected = _detect(image_pil, meta)
        is_correct, _ = _evaluate(image_pil, detected, meta)
        out.append(int(is_correct))
    return out
