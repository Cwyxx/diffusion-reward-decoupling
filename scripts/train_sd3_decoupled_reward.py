import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
import contextlib
import datetime
from concurrent import futures
import time
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
import torch
import swanlab
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from safetensors import safe_open
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        if split == 'val':
            self.prompts = self.prompts[:120]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "idx": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        indices = [example["idx"] for example in examples]
        return prompts, metadatas, indices

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def calculate_zero_std_ratio(prompts, gathered_rewards, reward_key='ori_avg'):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.

    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards.
        reward_key: Which reward entry to diagnose (default 'ori_avg'; for decoupled
            mode pass 'early_avg' or 'late_avg' to diagnose each stage separately).

    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)

    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )

    # Group rewards for each prompt
    grouped_rewards = gathered_rewards[reward_key][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, test_embed_file, neg_prompt_embed, neg_pooled_prompt_embed, inference_dtype, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # Fix seed for eval so that the same noise is used across different checkpoints
    eval_generator = torch.Generator(device=accelerator.device).manual_seed(42)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata, indices = test_batch
        prompt_embeds = test_embed_file.get_slice("prompt_embeds")[indices].to(inference_dtype).to(accelerator.device)
        pooled_prompt_embeds = test_embed_file.get_slice("pooled_prompt_embeds")[indices].to(inference_dtype).to(accelerator.device)
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                    generator=eval_generator,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            swanlab.log(
                {
                    "eval_images": [
                        swanlab.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    if not config.run_name:
        config.run_name = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # Validate decoupled-reward configuration early so mis-configurations fail before
    # expensive model loading.
    if config.reward_decoupled:
        early_keys = set(config.reward_fn_early.keys())
        late_keys = set(config.reward_fn_late.keys())
        reward_keys = set(config.reward_fn.keys())
        assert early_keys, (
            "reward_decoupled=True but reward_fn_early is empty; "
            "specify at least one reward model for early denoising steps."
        )
        assert late_keys, (
            "reward_decoupled=True but reward_fn_late is empty; "
            "specify at least one reward model for late denoising steps."
        )
        overlap = early_keys & late_keys
        assert not overlap, (
            f"reward_fn_early and reward_fn_late share reward models {sorted(overlap)}; "
            "overlapping keys would be double-counted when splitting advantages."
        )
        missing = (early_keys | late_keys) - reward_keys
        assert not missing, (
            f"reward_fn is missing keys {sorted(missing)} that appear in reward_fn_early/late; "
            "ensure config.reward_fn includes the union of both (e.g. "
            "`config.reward_fn = config.reward_fn_early | config.reward_fn_late`)."
        )
        assert 0.0 <= config.reward_split_ratio <= 1.0, (
            f"reward_split_ratio must be in [0, 1], got {config.reward_split_ratio}."
        )
        # Single source of truth for the early/late split boundary. Reused wherever
        # we need to construct per-timestep advantages or log the split.
        split_step = int(num_train_timesteps * config.reward_split_ratio)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        swanlab.init(
            project="flow_grpo",
            experiment_name=config.run_name,
            config=config.to_dict(),
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    # Unload text encoders to save VRAM (using pre-computed embeddings)
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.text_encoder_3 = None

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae to device
    pipeline.vae.to(accelerator.device, dtype=torch.float32)

    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    train_dataset = TextPromptDataset(config.dataset, 'train')
    test_dataset = TextPromptDataset(config.dataset, 'val')

    # Open pre-computed prompt embeddings (lazy, read on demand)
    train_embed_file = safe_open(os.path.join(config.prompt_embed_dir, "train.safetensors"), framework="pt")
    test_embed_file = safe_open(os.path.join(config.prompt_embed_dir, "val.safetensors"), framework="pt")

    # Load negative prompt embeddings (small, just one row)
    neg_prompt_embed_raw = train_embed_file.get_tensor("neg_prompt_embeds").to(inference_dtype)
    neg_pooled_prompt_embed_raw = train_embed_file.get_tensor("neg_pooled_prompt_embeds").to(inference_dtype)

    micro_bs = config.sample.micro_batch_size
    assert config.sample.mini_num_images_per_prompt % micro_bs == 0, f"mini_num_images_per_prompt ({config.sample.mini_num_images_per_prompt}) must be divisible by micro_batch_size ({micro_bs})"

    # Create an infinite-loop DataLoader
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.mini_num_images_per_prompt,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )

    # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=TextPromptDataset.collate_fn,
    )

    # Create a regular DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    neg_prompt_embed = neg_prompt_embed_raw.to(accelerator.device)
    neg_pooled_prompt_embed = neg_pooled_prompt_embed_raw.to(accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(micro_bs, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(micro_bs, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.mini_num_images_per_prompt
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample mini_num_images_per_prompt per device = {config.sample.mini_num_images_per_prompt} (micro_batch_size = {micro_bs})")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    if config.reward_decoupled:
        early_steps = list(range(0, split_step))
        late_steps = list(range(split_step, num_train_timesteps))
        logger.info("")
        logger.info("  ===== Decoupled Reward =====")
        logger.info(f"  num_train_timesteps = {num_train_timesteps} (num_steps={config.sample.num_steps} * timestep_fraction={config.train.timestep_fraction})")
        logger.info(f"  reward_split_ratio  = {config.reward_split_ratio} -> split_step = {split_step}")
        logger.info(f"  reward_fn_early ({dict(config.reward_fn_early)}) -> early_steps (len={len(early_steps)}): {early_steps}")
        logger.info(f"  reward_fn_late  ({dict(config.reward_fn_late)}) -> late_steps  (len={len(late_steps)}): {late_steps}")
    # assert config.sample.mini_num_images_per_prompt >= config.train.batch_size
    # assert config.sample.mini_num_images_per_prompt % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, test_embed_file, neg_prompt_embed, neg_pooled_prompt_embed, inference_dtype, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata, indices = next(train_iter)

            prompt_embeds = train_embed_file.get_slice("prompt_embeds")[indices].to(inference_dtype).to(accelerator.device)
            pooled_prompt_embeds = train_embed_file.get_slice("pooled_prompt_embeds")[indices].to(inference_dtype).to(accelerator.device)
            prompt_ids = pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample with serial micro-batch accumulation
            if config.sample.same_latent:
                all_generators = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                all_generators = None

            acc_images, acc_latents, acc_log_probs = [], [], []
            for micro_idx in range(0, config.sample.mini_num_images_per_prompt, micro_bs):
                micro_slice = slice(micro_idx, micro_idx + micro_bs)
                micro_gen = all_generators[micro_slice] if all_generators else None
                with autocast():
                    with torch.no_grad():
                        imgs_m, lats_m, lps_m = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds[micro_slice],
                            pooled_prompt_embeds=pooled_prompt_embeds[micro_slice],
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=micro_gen,
                        )
                acc_images.append(imgs_m)
                acc_latents.append(lats_m)
                acc_log_probs.append(lps_m)

            # concat micro-batch results
            images = torch.cat(acc_images, dim=0)
            num_lat_steps = len(acc_latents[0])
            latents = torch.stack(
                [torch.cat([acc_latents[r][s] for r in range(len(acc_latents))], dim=0) for s in range(num_lat_steps)],
                dim=1,
            )  # (batch_size, num_steps + 1, ...)
            num_lp_steps = len(acc_log_probs[0])
            log_probs = torch.stack(
                [torch.cat([acc_log_probs[r][s] for r in range(len(acc_log_probs))], dim=0) for s in range(num_lp_steps)],
                dim=1,
            )  # (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.mini_num_images_per_prompt, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        # Before: samples is a list of num_batches_per_epoch dicts, each containing tensors of shape (mini_batch, ...):
        #   samples = [
        #       {"latents": (mini_batch, ...), "log_probs": (mini_batch, ...), "rewards": {"pickscore": (mini_batch,)}, ...},
        #       {"latents": (mini_batch, ...), "log_probs": (mini_batch, ...), "rewards": {"pickscore": (mini_batch,)}, ...},
        #       ...  # num_batches_per_epoch entries
        #   ]
        # After: samples becomes a single dict with all tensors concatenated along dim=0:
        #   samples = {
        #       "latents": (num_batches_per_epoch * mini_batch, ...),
        #       "log_probs": (num_batches_per_epoch * mini_batch, ...),
        #       "rewards": {"pickscore": (num_batches_per_epoch * mini_batch,)},
        #       ...
        #   }
        # For plain tensor fields, torch.cat is applied directly.
        # For nested dict fields (e.g. "rewards"), torch.cat is applied to each sub_key separately.
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                swanlab.log(
                    {
                        "images": [
                            swanlab.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        # Backup the original per-image scalar reward before reshaping, so it remains accessible for logging.
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        if config.reward_decoupled:
            # Split the combined reward back into two stage-specific scalar rewards.
            # early_avg drives advantages for the first split_ratio fraction of denoising steps;
            # late_avg drives advantages for the remaining steps. Each is the weighted sum of
            # its constituent reward models (same convention as 'avg').
            early_avg = torch.zeros_like(samples["rewards"]["avg"])
            for name, weight in config.reward_fn_early.items():
                early_avg = early_avg + weight * samples["rewards"][name]
            late_avg = torch.zeros_like(samples["rewards"]["avg"])
            for name, weight in config.reward_fn_late.items():
                late_avg = late_avg + weight * samples["rewards"][name]
            samples["rewards"]["early_avg"] = early_avg
            samples["rewards"]["late_avg"] = late_avg
        else:
            # Expand reward from (batch_size,) to (batch_size, num_train_timesteps) by repeating along the
            # timestep dimension. This makes it easier to introduce timestep-dependent advantages later
            # (e.g., adding a KL reward that varies across denoising steps).
            samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # Gather rewards from all GPUs onto every process (and move to CPU numpy), so that the
        # subsequent per-prompt advantage computation can see all samples for the same prompt.
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            swanlab.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            if config.reward_decoupled:
                # Normalize each stage's reward independently, then assign advantages per-timestep:
                # first split steps get adv_early, remaining steps get adv_late. This is
                # mathematically equivalent to building an (N, T) reward tensor with the two
                # scalars tiled across columns and calling stat_tracker.update once, but avoids
                # redundant per-timestep normalization of identical values.
                adv_early = stat_tracker.update(prompts, gathered_rewards['early_avg'])  # (N,)
                # Clear between the two updates; otherwise early stats would contaminate late stats.
                stat_tracker.clear()
                adv_late = stat_tracker.update(prompts, gathered_rewards['late_avg'])    # (N,)
                advantages = np.concatenate(
                    [
                        np.broadcast_to(adv_early[:, None], (adv_early.shape[0], split_step)),
                        np.broadcast_to(adv_late[:, None], (adv_late.shape[0], num_train_timesteps - split_step)),
                    ],
                    axis=1,
                )  # (N, num_train_timesteps)
            else:
                advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
            # In decoupled mode, diagnose each stage independently so we can detect when
            # one reward model saturates while the other still has signal.
            if config.reward_decoupled:
                zero_std_ratio_early, reward_std_mean_early = calculate_zero_std_ratio(
                    prompts, gathered_rewards, reward_key='early_avg'
                )
                zero_std_ratio_late, reward_std_mean_late = calculate_zero_std_ratio(
                    prompts, gathered_rewards, reward_key='late_avg'
                )

            if accelerator.is_main_process:
                log_dict = {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                }
                if config.reward_decoupled:
                    log_dict.update({
                        "zero_std_ratio_early": zero_std_ratio_early,
                        "reward_std_mean_early": reward_std_mean_early,
                        "zero_std_ratio_late": zero_std_ratio_late,
                        "reward_std_mean_late": reward_std_mean_late,
                    })
                swanlab.log(log_dict, step=global_step)
            stat_tracker.clear()
        else:
            if config.reward_decoupled:
                re = gathered_rewards['early_avg']
                rl = gathered_rewards['late_avg']
                adv_early = (re - re.mean()) / (re.std() + 1e-4)
                adv_late = (rl - rl.mean()) / (rl.std() + 1e-4)
                advantages = np.concatenate(
                    [
                        np.broadcast_to(adv_early[:, None], (adv_early.shape[0], split_step)),
                        np.broadcast_to(adv_late[:, None], (adv_late.shape[0], num_train_timesteps - split_step)),
                    ],
                    axis=1,
                )
            else:
                advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            swanlab.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.mini_num_images_per_prompt * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                train_sample_size = len(sample["latents"])
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        num_micro_batches = (train_sample_size + micro_bs - 1) // micro_bs
                        for micro_idx in range(0, train_sample_size, micro_bs):
                            micro_end = micro_idx + micro_bs
                            micro_sample = {k: v[micro_idx:micro_end] for k, v in sample.items()}
                            if config.train.cfg:
                                # embeds layout is [neg_0..neg_n, pos_0..pos_n], slice both halves in sync
                                micro_embeds = torch.cat([
                                    embeds[micro_idx:micro_end],
                                    embeds[train_sample_size + micro_idx:train_sample_size + micro_end]
                                ])
                                micro_pooled = torch.cat([
                                    pooled_embeds[micro_idx:micro_end],
                                    pooled_embeds[train_sample_size + micro_idx:train_sample_size + micro_end]
                                ])
                            else:
                                micro_embeds = embeds[micro_idx:micro_end]
                                micro_pooled = pooled_embeds[micro_idx:micro_end]

                            with autocast():
                                ps, lp, psm, sd = compute_log_prob(transformer, pipeline, micro_sample, j, micro_embeds, micro_pooled, config)
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            _, _, psm_ref, _ = compute_log_prob(transformer, pipeline, micro_sample, j, micro_embeds, micro_pooled, config)

                            # Compute loss per micro-batch and backward immediately to free computation graph
                            micro_advantages = torch.clamp(
                                micro_sample["advantages"][:, j],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )
                            ratio = torch.exp(lp - micro_sample["log_probs"][:, j])
                            unclipped_loss = -micro_advantages * ratio
                            clipped_loss = -micro_advantages * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            if config.train.beta > 0:
                                kl_loss = ((psm - psm_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * sd ** 2)
                                kl_loss = torch.mean(kl_loss)
                                loss = policy_loss + config.train.beta * kl_loss
                            else:
                                loss = policy_loss

                            # Scale loss by num_micro_batches so accumulated gradients are equivalent
                            accelerator.backward(loss / num_micro_batches)

                            # Log metrics — detach to release computation graph
                            info["approx_kl"].append(
                                (0.5 * torch.mean((lp - micro_sample["log_probs"][:, j]) ** 2)).detach()
                            )
                            info["clipfrac"].append(
                                torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()).detach()
                            )
                            info["clipfrac_gt_one"].append(
                                torch.mean((ratio - 1.0 > config.train.clip_range).float()).detach()
                            )
                            info["clipfrac_lt_one"].append(
                                torch.mean((1.0 - ratio > config.train.clip_range).float()).detach()
                            )
                            info["policy_loss"].append(policy_loss.detach())
                            if config.train.beta > 0:
                                info["kl_loss"].append(kl_loss.detach())
                            info["loss"].append(loss.detach())

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            swanlab.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

