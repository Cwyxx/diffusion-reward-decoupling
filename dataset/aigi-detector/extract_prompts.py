"""从 MSCOCO val2014 caption 文件中按 image-level 抽取 1000 条 prompt，用于后续图像生成。

输入: mscoco_val_2014.json —— 扁平 list，每条形如 {"image": "203564", "text": "A clock ..."}。
      一张图片可能对应多条 caption；本脚本先按 image id 去重（每张图保留一条 caption），
      再从所有唯一图片中随机抽取 N 张。

输出: mscoco_val_2014_1000prompts.json —— list[{"image": <id>, "text": <prompt>}]，
      同时输出一份纯文本 mscoco_val_2014_1000prompts.txt（每行一条 prompt）便于直接喂给生成脚本。

用法:
    python extract_prompts.py                 # 默认抽 1000 条，seed=42
    python extract_prompts.py --num 2000 --seed 0
"""
import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "mscoco_val_2014.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="caption json 路径")
    parser.add_argument("--num", type=int, default=1000, help="抽取的图片(prompt)数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--out-prefix", default=None,
                        help="输出文件前缀，默认 mscoco_val_2014_<num>prompts")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    # 按 image id 聚合所有 caption（一图多 caption 时全部收集，便于稳定挑选）
    img2caps = defaultdict(list)
    for r in records:
        img2caps[str(r["image"])].append(r["text"].strip())

    image_ids = sorted(img2caps.keys())
    print(f"读入 {len(records)} 条 caption，覆盖 {len(image_ids)} 张唯一图片")

    if args.num > len(image_ids):
        raise SystemExit(f"请求 {args.num} 张，但只有 {len(image_ids)} 张唯一图片")

    rng = random.Random(args.seed)
    sampled_ids = rng.sample(image_ids, args.num)

    # 每张图选一条 caption：若有多条，用同一 rng 确定性地选一条
    out = []
    for img_id in sampled_ids:
        caps = img2caps[img_id]
        text = caps[0] if len(caps) == 1 else rng.choice(caps)
        out.append({"image": img_id, "text": text})

    prefix = args.out_prefix or os.path.join(HERE, f"mscoco_val_2014_{args.num}prompts")
    json_path = prefix + ".json"
    txt_path = prefix + ".txt"

    with open(json_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w") as f:
        f.write("\n".join(item["text"] for item in out) + "\n")

    print(f"已写出 {len(out)} 条 prompt:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print("示例:")
    for item in out[:3]:
        print(f"  [{item['image']}] {item['text']}")


if __name__ == "__main__":
    main()
