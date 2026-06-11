"""Precompute SD3.5-M text embeddings for flow-grpo training.

Reads dataset/<name>/{train,test}.txt (one prompt per line) and writes one
<split>.safetensors per split, with rows aligned to the txt line order:

    prompt_embeds            [N, 205, 4096]  (CLIP-L/G padded + T5, seq 77+128)
    pooled_prompt_embeds     [N, 2048]
    neg_prompt_embeds        [1, 205, 4096]  (empty prompt "")
    neg_pooled_prompt_embeds [1, 2048]

This is the format `train_sd3_gardo.py` / `train_sd3.py` etc. read via safe_open.
Stored in fp16, matching the existing HPDv3 embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/precompute_prompt_embeddings.py \
        --dataset dataset/HPDv3 \
        --output_dir /data_center/data2/dataset/chenwy/21164-data/diffusion-reward-decoupling/prompt-embedding/HPDv3 \
        --splits train test
"""

import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import save_file
from tqdm import tqdm

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


@torch.no_grad()
def encode_batch(prompts, text_encoders, tokenizers, max_sequence_length, device):
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders, tokenizers, prompts, max_sequence_length, device=device
    )
    return prompt_embeds, pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset dir containing <split>.txt files, e.g. dataset/HPDv3")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write <split>.safetensors")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sequence_length", type=int, default=128, help="T5 max length; final seq len is 77 + this")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Text encoders only; transformer/VAE are not needed.
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, transformer=None, vae=None)
    for enc in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        enc.requires_grad_(False)
        enc.to(device=device, dtype=dtype)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_batch(
        [""], text_encoders, tokenizers, args.max_sequence_length, device
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for split in args.splits:
        txt_path = os.path.join(args.dataset, f"{split}.txt")
        if not os.path.exists(txt_path):
            print(f"[skip] {txt_path} does not exist")
            continue
        # Must mirror TextPromptDataset exactly so row i == line i of the txt.
        with open(txt_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]

        n = len(prompts)
        seq_len, embed_dim = neg_prompt_embeds.shape[1], neg_prompt_embeds.shape[2]
        pooled_dim = neg_pooled_prompt_embeds.shape[1]
        prompt_embeds = torch.empty((n, seq_len, embed_dim), dtype=dtype)
        pooled_prompt_embeds = torch.empty((n, pooled_dim), dtype=dtype)

        for start in tqdm(range(0, n, args.batch_size), desc=f"{split} ({n} prompts)", dynamic_ncols=True):
            batch = prompts[start:start + args.batch_size]
            embeds, pooled = encode_batch(batch, text_encoders, tokenizers, args.max_sequence_length, device)
            prompt_embeds[start:start + len(batch)] = embeds.to(dtype).cpu()
            pooled_prompt_embeds[start:start + len(batch)] = pooled.to(dtype).cpu()

        out_path = os.path.join(args.output_dir, f"{split}.safetensors")
        save_file(
            {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "neg_prompt_embeds": neg_prompt_embeds.to(dtype).cpu(),
                "neg_pooled_prompt_embeds": neg_pooled_prompt_embeds.to(dtype).cpu(),
            },
            out_path,
        )
        print(f"[done] {out_path}: prompt_embeds {tuple(prompt_embeds.shape)}, pooled {tuple(pooled_prompt_embeds.shape)}")


if __name__ == "__main__":
    main()
