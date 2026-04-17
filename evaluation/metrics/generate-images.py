# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        file_path = os.path.join(dataset_path, f"{split}.txt")
        with open(file_path, "r") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [e["prompt"] for e in examples]
    indices = [e["original_index"] for e in examples]
    return prompts, indices


def load_pipeline(checkpoint_path, dtype, device):
    pipeline = StableDiffusion3Pipeline.from_pretrained(MODEL_ID)

    if checkpoint_path:
        lora_path = os.path.join(checkpoint_path, "lora", "learner")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA directory not found at {lora_path}")
        print(f"Loading LoRA weights from: {lora_path}")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
        pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
        pipeline.transformer.load_adapter(
            lora_path, adapter_name="learner", is_trainable=False
        )
        pipeline.transformer.set_adapter("learner")

    pipeline.transformer.eval()
    pipeline.transformer.to(device, dtype=dtype)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main(args):
    device = torch.device("cuda")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}
    dtype = dtype_map[args.mixed_precision]
    enable_amp = args.mixed_precision != "no"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    pipeline = load_pipeline(args.checkpoint_path, dtype, device)

    dataset_dir = os.path.join(DATASET_ROOT, args.dataset)
    dataset = TextPromptDataset(dataset_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    results = []
    for prompts, indices in tqdm(dataloader, desc="Generating"):
        generator = torch.Generator(device).manual_seed(args.seed)
        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=dtype), torch.no_grad():
            images = pipeline(
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type="pt",
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            )[0]

        for i, sample_idx in enumerate(indices):
            result = {"sample_id": sample_idx, "prompt": prompts[i], "scores": {}}
            if args.save_images:
                image_path = os.path.join(
                    args.output_dir, "images", f"{sample_idx:05d}.png"
                )
                arr = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(arr).save(image_path)
                result["image_path"] = image_path
            results.append(result)

    results.sort(key=lambda x: x["sample_id"])
    results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images with SD3.5-M from a prompt list under dataset/<name>/<split>.txt."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="drawbench-unique",
        help="Subfolder name under dataset/ containing <split>.txt (e.g. drawbench-unique, partiprompts).",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="LoRA checkpoint dir containing lora/learner. Omit to use base model.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    main(parser.parse_args())
