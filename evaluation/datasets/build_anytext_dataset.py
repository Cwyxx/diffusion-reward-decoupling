#!/usr/bin/env python3
"""Build AnyText-Benchmark prompt datasets for the best-of-N framework.

Converts AnyText-Benchmark ``test1k.json`` (laion_word = English text, wukong_word
= Chinese text) into ``dataset/anytext-en/test.jsonl`` and
``dataset/anytext-zh/test.jsonl`` so they plug into ``evaluation/run-bestofn.sh``.

Prompt construction is a faithful reimplementation of AnyText's own pipeline:

  * ``load_data``  -> ``flow_grpo/AnyText/eval/anytext_singleGPU.py:57``
  * ``get_item``   -> ``flow_grpo/AnyText/eval/anytext_singleGPU.py:98`` (caption part only;
                      glyph / position-mask rendering is irrelevant for base T2I models)
  * ``get_caption_pos`` + ``phrase_list`` -> ``flow_grpo/AnyText/t3_dataset.py:12`` and ``:125``

Difference from AnyText's own eval: AnyText keeps ``fill_caption=False`` because it feeds
the ground-truth glyphs through an ``embedding_manager`` that encodes them into the ``*``
token. Plain text-to-image models (SD3.5, FLUX, ...) have no such mechanism, so we set
``fill_caption=True``: each ``*`` placeholder is replaced **in order** with the quoted
ground-truth string (e.g. ``"Piano"``). This matches the "T2I baseline" input distribution
reported in the AnyText paper.

``random.seed`` is fixed (default 0) so the per-sample guide phrase chosen from
``phrase_list`` is deterministic -> the generated prompt files are fully reproducible, and
base vs. post-trained models are evaluated on identical prompt text.

Each output line:

  {"prompt": "<filled caption>",
   "metadata": {"item_id": <int>, "img_name": <str>,
                "polygons": [[[x, y], ...], ...], "texts": [...], "pos": [...],
                "lang": "en" | "zh"}}

The ``polygons`` / ``texts`` are required downstream by ``anytext_scorer.py`` to crop the
rendered text regions and run OCR (Sen.ACC + NED).

Usage:
  python evaluation/datasets/build_anytext_dataset.py \
      --src ~/Downloads/benchmark \
      --dst dataset/
"""
import argparse
import json
import os
import random

PLACE_HOLDER = '*'
MAX_CHARS = 20
MAX_LINES = 20

# --- verbatim from flow_grpo/AnyText/t3_dataset.py:12-23 ---
phrase_list = [
    ', content and position of the texts are ',
    ', textual material depicted in the image are ',
    ', texts that says ',
    ', captions shown in the snapshot are ',
    ', with the words of ',
    ', that reads ',
    ', the written materials on the picture: ',
    ', these texts are written on it: ',
    ', captions are ',
    ', content of the text in the graphic is '
]


def get_caption_pos(ori_caption, pos_idxs, prob=1.0, place_holder='*'):
    """Verbatim from flow_grpo/AnyText/t3_dataset.py:125-146.

    With prob=0.0 (used by AnyText eval): appends a randomly chosen guide phrase, then one
    ``*`` placeholder per text line joined by ' , ', terminated with '.'. The ``idx2pos``
    dict and the per-line ``random.random()`` draw are kept intact so the random-number
    stream (and therefore the chosen phrase) is identical every run for a fixed seed.
    """
    idx2pos = {
        0: " top left",
        1: " top",
        2: " top right",
        3: " left",
        4: random.choice([" middle", " center"]),
        5: " right",
        6: " bottom left",
        7: " bottom",
        8: " bottom right"
    }
    new_caption = ori_caption + random.choice(phrase_list)
    pos = ''
    for i in range(len(pos_idxs)):
        if random.random() < prob and pos_idxs[i] > 0:
            pos += place_holder + random.choice([' located', ' placed', ' positioned', '']) + random.choice([' at', ' in', ' on']) + idx2pos[pos_idxs[i]] + ', '
        else:
            pos += place_holder + ' , '
    pos = pos[:-2] + '.'
    new_caption += pos
    return new_caption


def load_samples(json_path):
    """Mirror anytext_singleGPU.load_data (lines 57-87): keep only valid, non-empty-polygon
    annotations; if a caption already contains '*', AnyText replaces it with a space."""
    with open(json_path, "r") as f:
        content = json.load(f)
    samples = []
    for gt in content['data_list']:
        caption = gt['caption']
        if PLACE_HOLDER in caption:
            caption = caption.replace(PLACE_HOLDER, " ")
        polygons, texts, pos = [], [], []
        for ann in gt.get('annotations', []):
            if len(ann['polygon']) == 0:
                continue
            if ann['valid'] is False:
                continue
            polygons.append(ann['polygon'])
            texts.append(ann['text'])
            pos.append(ann['pos'])
        samples.append({
            'img_name': gt['img_name'],
            'caption': caption,
            'polygons': polygons,
            'texts': texts,
            'pos': pos,
        })
    return samples


def build_item(cur, item_id, lang):
    """Mirror anytext_singleGPU.get_item caption logic (lines 98-138) with fill_caption=True."""
    caption = cur['caption']
    texts = cur['texts']
    polygons = cur['polygons']
    pos = cur['pos']
    if len(texts) > 0:
        sel = list(range(len(texts)))
        if len(texts) > MAX_LINES:
            sel = sel[:MAX_LINES]
        pos_idxs = [pos[i] for i in sel]
        caption = get_caption_pos(caption, pos_idxs, 0.0, PLACE_HOLDER)
        polygons = [polygons[i] for i in sel]
        texts = [texts[i][:MAX_CHARS] for i in sel]
        pos = [pos[i] for i in sel]
    # fill_caption=True: base T2I models have no embedding_manager, so the '*' placeholders
    # must be replaced (in order) with the quoted ground-truth text.
    #
    # AnyText's get_item (lines 127-129) does `caption.replace('*', f'"{txt}"', 1)` in a loop.
    # That re-scans from the start each time, so when a ground-truth text itself contains a
    # literal '*' (e.g. "300*300厨"), the inserted '*' gets consumed by a later iteration and
    # the order scrambles. AnyText never hits this because its own eval keeps fill_caption=False
    # (embedding_manager path), so there is no official number to match. We use a split-based
    # fill instead: get_caption_pos added exactly len(texts) placeholders and the raw caption
    # was '*'-stripped by load_samples, so every '*' here is a placeholder. Splitting on '*' and
    # interleaving fills each placeholder in order without re-scanning inserted text -> robust to
    # '*' inside a ground-truth string.
    parts = caption.split(PLACE_HOLDER)
    assert len(parts) - 1 == len(texts), \
        f"placeholder/text count mismatch: {len(parts) - 1} vs {len(texts)} ({cur['img_name']})"
    filled = parts[0]
    for r_txt, tail in zip(texts, parts[1:]):
        filled += f'"{r_txt}"' + tail
    caption = filled
    return {
        'prompt': caption,
        'metadata': {
            'item_id': item_id,
            'img_name': cur['img_name'],
            'polygons': polygons,
            'texts': texts,
            'pos': pos,
            'lang': lang,
        },
    }


# laion_word -> English text; wukong_word -> Chinese text.
SUBSETS = {
    'laion_word':  ('anytext-en', 'en'),
    'wukong_word': ('anytext-zh', 'zh'),
}


def build_subset(src_root, dst_root, subset, out_name, lang, seed):
    json_path = os.path.join(src_root, subset, 'test1k.json')
    samples = load_samples(json_path)
    random.seed(seed)  # reset per-subset so each file is independently reproducible
    items = [build_item(cur, i, lang) for i, cur in enumerate(samples)]

    out_dir = os.path.join(dst_root, out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'test.jsonl')
    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_empty = sum(1 for it in items if len(it['metadata']['texts']) == 0)
    print(f"[{subset} -> {out_name}] {len(items)} prompts written to {out_path}"
          f"  (lang={lang}, {n_empty} with no text lines)")
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='benchmark root containing laion_word/ and wukong_word/')
    ap.add_argument('--dst', default='dataset',
                    help='output dataset root (default: dataset/)')
    ap.add_argument('--seed', type=int, default=0,
                    help='random seed for guide-phrase selection (default: 0)')
    args = ap.parse_args()

    src_root = os.path.expanduser(args.src)
    dst_root = os.path.expanduser(args.dst)
    for subset, (out_name, lang) in SUBSETS.items():
        if not os.path.isfile(os.path.join(src_root, subset, 'test1k.json')):
            print(f"[skip] {subset}: no test1k.json under {src_root}")
            continue
        build_subset(src_root, dst_root, subset, out_name, lang, args.seed)


if __name__ == '__main__':
    main()
