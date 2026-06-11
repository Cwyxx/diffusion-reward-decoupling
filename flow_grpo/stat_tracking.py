import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, dists=None, type='grpo'):
        """Compute per-prompt normalized advantages.

        Args:
            prompts: list of N prompt strings (same prompt appears multiple times due to num_image_per_prompt)
            rewards: (N,) or (N, num_train_timesteps) array of rewards
            dists: (N, feat_dim) array of image features (e.g. DINOv3), required for type='gardo'
            type: 'grpo' | 'gardo' | 'rwr' | 'sft' | 'dpo'

        Returns:
            advantages: same shape as rewards, normalized per-prompt (and per-timestep if 2D)
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        if dists is not None and type=='gardo':
            dists = np.array(dists)
            dists = dists / np.linalg.norm(dists, ord=2, axis=1, keepdims=True)
        # First pass: accumulate rewards for each prompt
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
        # Second pass: compute advantages using per-prompt statistics
        # axis=0 computes mean/std across samples, independently for each timestep
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            # mean shape: (1,) or (1, num_train_timesteps), per-prompt mean for each timestep
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if dists is not None and type=='gardo':
                # GARDO: diversity weight from the nearest-neighbor cosine distance within the group.
                # Samples below the group mean keep weight 1; samples above are reweighted by how
                # "novel" they are relative to the rest of the group.
                prompt_dists = dists[prompts == prompt]
                similarity_matrix = np.dot(prompt_dists, prompt_dists.T).clip(-1, 1)
                distance_matrix = 1 - similarity_matrix
                avg_distances = np.zeros(len(distance_matrix))
                for i in range(len(distance_matrix)):
                    other_mask = np.ones(len(distance_matrix), dtype=bool)
                    other_mask[i] = False
                    avg_distances[i] = np.min(distance_matrix[i, other_mask])
                prompt_dists = np.repeat(avg_distances[:, np.newaxis], prompt_rewards.shape[-1], axis=1)
                prompt_dists = np.where((prompt_rewards - mean) < 0, np.ones_like(prompt_dists), prompt_dists / np.mean(prompt_dists))
            if self.global_std:
                # std across ALL samples in this batch (all prompts), per-timestep
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                # std across samples of this specific prompt only, per-timestep
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type=='gardo':
                # No std normalization: the centered reward is scaled by the diversity weight instead.
                advantages[prompts == prompt] = (prompt_rewards - mean) * prompt_dists
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()