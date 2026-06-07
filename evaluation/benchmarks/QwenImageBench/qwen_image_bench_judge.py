"""Reusable Qwen-Image-Bench judging logic, decoupled from the ms-swift engine.

Builds the per-L1-dimension checklist inference tasks and turns raw judge text
into a per-image overall score + 5 L1 dimension scores. Imports only the
checklist/score helpers vendored alongside this module (checklists.py,
score_utils.py — copied from upstream Qwen-Image-Bench; no swift dependency),
so it is unit-testable without the 27B model. The judge engine is injected by
the caller (score-images.py).
"""
from PIL import Image

from .checklists import (
    DIM_TO_CHECKLIST,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_dims_by_level1,
)
from .score_utils import (
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

# L1 dimension name -> score-key suffix used in evaluation_results.jsonl.
DIM_KEY = {
    "Quality": "quality",
    "Aesthetics": "aesthetics",
    "Alignment": "alignment",
    "Real-world Fidelity": "fidelity",
    "Creative Generation": "creative",
}


def load_and_resize_image(path):
    """Load RGB image; resize to 1024x1024 if any side > 1024 (matches judge.py).

    Note: the resize is a fixed 1024x1024 and does NOT preserve aspect ratio
    (a 2048x512 image becomes 1024x1024). This is intentional parity with the
    upstream judge engine's preprocessing — do not "fix" it to a thumbnail.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
    img.load()
    return img


def build_tasks(prompt, dims_en, image):
    """Return [(level1_dim, task_dict), ...] for the L1 dimensions this prompt covers.

    task_dict matches the judge engine's item schema:
    {"system_prompt": str, "user_text": str, "image": PIL.Image}.
    """
    dims_by_l1 = parse_dims_by_level1(dims_en)
    tasks = []
    for level1_dim in dims_by_l1:
        if level1_dim not in DIM_TO_CHECKLIST:
            continue
        user_text = USER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            level1_dim=level1_dim,
            format_checklist=DIM_TO_CHECKLIST[level1_dim],
        )
        tasks.append((level1_dim, {
            "system_prompt": SYSTEM_PROMPT,
            "user_text": user_text,
            "image": image,
        }))
    return tasks


def scores_from_raw(raw_by_dim):
    """Turn {L1_dim: raw_judge_text} into (overall, dim_scores, parsed_by_dim).

    overall = mean of non-None L1 scores; dim_scores keyed by DIM_KEY suffix
    (only dims that parsed and yielded a non-None L1 score). Unparseable dims
    are skipped. parsed_by_dim holds the fixed L3 score JSON per dim (for the
    raw-judgments file).
    """
    parsed_by_dim = {}
    l1_by_dim = {}
    for level1_dim, raw in raw_by_dim.items():
        score_json = extract_json_from_response(raw)
        if score_json is None:
            continue
        fixed = fix_score_json(score_json, level1_dim)
        parsed_by_dim[level1_dim] = fixed
        l1_by_dim[level1_dim] = compute_dimension_score(fixed)["level1_score"]

    valid = [v for v in l1_by_dim.values() if v is not None]
    overall = sum(valid) / len(valid) if valid else None
    dim_scores = {
        DIM_KEY[d]: v for d, v in l1_by_dim.items()
        if d in DIM_KEY and v is not None
    }
    return overall, dim_scores, parsed_by_dim
