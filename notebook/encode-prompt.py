"""
Pre-compute and cache SD3.5 prompt embeddings (safetensors format).

Usage:
    python notebook/encode-prompt.py \
        --model_path stabilityai/stable-diffusion-3.5-medium \
        --prompt_file dataset/ocr/train.txt \
        --output_file dataset/ocr/train_embeds.safetensors \
        --max_sequence_length 128 \
        --batch_size 16

Loading (lazy, only reads requested tensors into memory):
    from safetensors import safe_open
    with safe_open("train_embeds.safetensors", framework="pt") as f:
        embed_0 = f.get_slice("prompt_embeds")[0]          # single row
        neg_prompt_embeds = f.get_tensor("neg_prompt_embeds")
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SD3.5 prompt embeddings")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--prompt_file", type=str, default="../dataset/HPDv3/train.txt")
    parser.add_argument("--output_file", type=str, default="/data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/prompt-embedding/HPDv3/train.safetensors")
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    dtype = torch.float16

    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    print(f"Loading SD3 pipeline from {args.model_path} ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for enc in text_encoders:
        enc.to(device, dtype=dtype)
        enc.requires_grad_(False)
        enc.eval()

    all_prompt_embeds = []
    all_pooled_prompt_embeds = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Encoding prompts"):
        batch = prompts[i : i + args.batch_size]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, batch, args.max_sequence_length
            )
        all_prompt_embeds.append(prompt_embeds.cpu())
        all_pooled_prompt_embeds.append(pooled_prompt_embeds.cpu())

    # Encode negative prompt (empty string)
    with torch.no_grad():
        neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, [""], args.max_sequence_length
        )

    tensors = {
        "prompt_embeds": torch.cat(all_prompt_embeds, dim=0),
        "pooled_prompt_embeds": torch.cat(all_pooled_prompt_embeds, dim=0),
        "neg_prompt_embeds": neg_prompt_embeds.cpu(),
        "neg_pooled_prompt_embeds": neg_pooled_prompt_embeds.cpu(),
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    save_file(tensors, args.output_file)
    print(f"Saved embeddings to {args.output_file}")
    for k, v in tensors.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
