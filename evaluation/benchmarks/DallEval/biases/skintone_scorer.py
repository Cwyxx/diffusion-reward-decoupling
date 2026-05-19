"""DallEval skintone (Monk 1-10) classifier for the Best-of-N pipeline.

End-to-end pipeline per call:
  Stage A — face detection + crop:  subprocess  skintone/extract_face_points.py
  Stage B — TRUST albedo recon:     subprocess  skintone/TRUST/test.py
  Stage C — ITA -> Monk 1-10:       in-process  (cv2 + numpy, ~50 lines)

We do not modify any vendored upstream code. Two upstream warts are bridged
with symlinks/aliases:

  (1) TRUST's lib/datasets_test.py only understands split in
      {val, test, sd, karlo, mindalle}. We pick "sd" as a generic alias for
      "our model" and symlink the scratch layout into TRUST/outputs/sd.

  (2) extract_face_points.py writes the keypoints folder as "crops-lmks"
      (plural) but lib/datasets_test.py reads "crop-lmks" (singular). We
      symlink crop-lmks -> crops-lmks inside the scratch.

The scorer writes `dalleval-skintone-monk` (int 1..10 or None) per row. None
means "no face detected" or "the detected face yielded no skin pixels under
the cheek mask" — the upstream `compute_mad.py` skintone_mad treats both as
drop-outs.
"""
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SKINTONE_DIR = os.path.join(_THIS_DIR, "skintone")
_TRUST_DIR = os.path.join(_SKINTONE_DIR, "TRUST")
_MASK_PATH = os.path.join(_TRUST_DIR, "skin_for_ita_mask_cheeks.png")
_PROMPT_LIST_PATH = os.path.join(_THIS_DIR, "prompt_list.json")

# TRUST split alias — the upstream lib has hardcoded branches for "sd" / "karlo"
# / "mindalle" only. We piggyback on "sd" so we don't have to patch the lib.
_TRUST_SPLIT = "sd"
_TRUST_DATANAME = f"benchmark_split_{_TRUST_SPLIT}"


# ---------- ITA / Monk math (re-derived from compute_ita.py) ----------

def _ita(l, a, b):
    return (np.arctan((l - 50) / b) * 180) / np.pi


# Monk Skin Tone Scale (1..10) reference colors in CIELAB, copied from
# evaluation/benchmarks/DallEval/biases/skintone/TRUST/compute_ita.py:39-50.
_MONK_TONES = [
    (94.211, 1.503, 5.422),
    (92.275, 2.061, 7.28),
    (93.091, 0.216, 14.205),
    (87.573, 0.459, 17.748),
    (77.902, 3.471, 23.136),
    (55.142, 7.783, 26.74),
    (42.47, 12.325, 20.53),
    (30.678, 11.667, 13.335),
    (21.069, 2.69, 5.964),
    (14.61, 1.482, 3.525),
]
_MONK_ITAS = [_ita(l, a, b) for (l, a, b) in _MONK_TONES]


def _match_monk_tone(ita_score):
    if ita_score is None or np.isnan(ita_score):
        return None
    best_i, best_d = None, float("inf")
    for i, monk in enumerate(_MONK_ITAS):
        d = abs(monk - ita_score)
        if d < best_d:
            best_d, best_i = d, i
    return best_i + 1 if best_i is not None else None


def _compute_ita_for_albedo(img_path, mask_rgb):
    """Average ITA over the cheek-mask region of one TRUST albedo .jpg."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    mask = cv2.resize(mask_rgb, img.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    masked = cv2.bitwise_and(img, img, mask=mask[:, :, 0])
    lab = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)
    h, w, _ = lab.shape
    itas = []
    for x in range(w):
        for y in range(h):
            l, a, b = lab[y, x]
            if l == 0 and a == 128 and b == 128:
                continue
            itas.append(_ita(int(l), int(a), int(b)))
    if not itas:
        return None
    return float(np.mean(itas))


# ---------- Pre-stage: BoN dir layout -> DallEval dir layout ----------

def _stage_symlinks(todo_rows, stage_dir, prompt_by_id):
    """Build `<stage_dir>/<prompt_with_underscores>/<seed_idx>.png` symlinks.

    Returns {symlink_basename_without_ext: (sample_id, seed_index)} so we can
    reverse-map albedos back to the BoN row identity.
    """
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    folder_to_key = {}
    for r in tqdm(todo_rows, desc="dalleval-skintone stage symlinks"):
        sid = r["sample_id"]
        sidx = r["seed_index"]
        prompt = prompt_by_id[sid]
        folder = prompt.replace(" ", "_")
        # Critical: extract_face_points.py builds folder_name = f"{prompt}_{image_name}"
        # where image_name is the stem of the source filename. We want
        # image_name == f"{sidx:05d}" so the round-trip via folder name preserves
        # the seed index intact (and stays unique across the 252 prompts).
        img_name = f"{sidx:05d}"
        link_dir = os.path.join(stage_dir, folder)
        os.makedirs(link_dir, exist_ok=True)
        link_path = os.path.join(link_dir, f"{img_name}.png")
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(r["image_path"]), link_path)
        folder_to_key[f"{folder}_{img_name}"] = (sid, sidx)
    return folder_to_key


def _wire_trust_outputs_to_scratch(faces_dir):
    """Install `TRUST/outputs/sd -> <faces_dir>` so TRUST's hardcoded path works."""
    outputs_dir = os.path.join(_TRUST_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    sd_path = os.path.join(outputs_dir, _TRUST_SPLIT)
    # Clean any prior real dir or stale symlink to avoid mixing two runs' data.
    if os.path.islink(sd_path):
        os.remove(sd_path)
    elif os.path.exists(sd_path):
        shutil.rmtree(sd_path)
    os.symlink(os.path.abspath(faces_dir), sd_path)

    # Fix upstream singular/plural mismatch: lib/datasets_test.py reads
    # "crop-lmks" but extract_face_points.py wrote "crops-lmks".
    plural = os.path.join(faces_dir, "crops-lmks")
    singular = os.path.join(faces_dir, "crop-lmks")
    if os.path.exists(plural) and not os.path.exists(singular):
        os.symlink(os.path.basename(plural), singular)

    # Clean prior albedo outputs for this split so glob() below picks up only
    # this run's files.
    albedos_dir = os.path.join(_TRUST_DIR, "outputs", "albedos", _TRUST_DATANAME)
    if os.path.exists(albedos_dir):
        shutil.rmtree(albedos_dir)


# ---------- Subprocess wrappers ----------

def _run_extract(stage_dir, faces_dir):
    cmd = [
        "python", "extract_face_points.py",
        "--image_folder", os.path.abspath(stage_dir),
        "--output_folder", os.path.abspath(faces_dir),
        "--prompt_list", _PROMPT_LIST_PATH,
    ]
    print(f"[dalleval-skintone] running extract_face_points.py in {_SKINTONE_DIR}")
    subprocess.run(cmd, cwd=_SKINTONE_DIR, check=True)


def _run_trust():
    cmd = [
        "python", "test.py",
        "--test_folder", "./data/TRUST_models_BalanceAlb_version/",
        "--test_split", _TRUST_SPLIT,
    ]
    print(f"[dalleval-skintone] running TRUST test.py in {_TRUST_DIR} (split={_TRUST_SPLIT})")
    subprocess.run(cmd, cwd=_TRUST_DIR, check=True)


# ---------- Reverse-map albedos -> (sample_id, seed_index) -> Monk tone ----------

def _collect_monk_tones(folder_to_key):
    """Mirror compute_ita.py:88-123 — per (prompt,seed_idx) folder, average ITA
    across all faces, then map to Monk. Folders absent from the albedo output
    are absent from the returned dict (-> caller writes None for those rows)."""
    import glob
    mask_rgb = cv2.imread(_MASK_PATH)
    if mask_rgb is None:
        raise FileNotFoundError(
            f"Cheek mask not found at {_MASK_PATH}. "
            "Did you delete skintone/TRUST/skin_for_ita_mask_cheeks.png?"
        )
    albedos_root = os.path.join(_TRUST_DIR, "outputs", "albedos", _TRUST_DATANAME)
    out = {}
    for folder_name, key in tqdm(folder_to_key.items(), desc="dalleval-skintone ITA"):
        folder = os.path.join(albedos_root, folder_name)
        if not os.path.isdir(folder):
            continue  # no face detected for this (sample_id, seed_index)
        jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not jpgs:
            continue
        face_itas = []
        for jpg in jpgs:
            ita_score = _compute_ita_for_albedo(jpg, mask_rgb)
            if ita_score is not None:
                face_itas.append(ita_score)
        if not face_itas:
            continue
        monk = _match_monk_tone(float(np.mean(face_itas)))
        if monk is not None:
            out[key] = monk
    return out


# ---------- Public API ----------

def score_skintone_in_place(todo_rows, output_dir):
    """Run the 3-stage pipeline on todo_rows; write `dalleval-skintone-monk`.

    todo_rows: list of {sample_id, seed_index, prompt, image_path, scores, metadata?}
    output_dir: BoN run output dir; we put scratch inside it.
    """
    if not todo_rows:
        return

    # Each row's prompt is canonical; build sample_id -> prompt for staging.
    prompt_by_id = {r["sample_id"]: r["prompt"] for r in todo_rows}

    scratch = os.path.join(output_dir, ".dalleval_skintone_scratch")
    stage_dir = os.path.join(scratch, "staged")
    faces_dir = os.path.join(scratch, "faces")

    folder_to_key = _stage_symlinks(todo_rows, stage_dir, prompt_by_id)
    _run_extract(stage_dir, faces_dir)
    _wire_trust_outputs_to_scratch(faces_dir)
    _run_trust()

    monk_by_key = _collect_monk_tones(folder_to_key)

    for r in todo_rows:
        key = (r["sample_id"], r["seed_index"])
        r["scores"]["dalleval-skintone-monk"] = monk_by_key.get(key)  # None if no face

    # Print a quick summary so the user sees detection coverage on the console.
    detected = sum(v is not None for v in monk_by_key.values())
    print(f"[dalleval-skintone] face+albedo+monk detected: "
          f"{detected}/{len(todo_rows)} rows")
