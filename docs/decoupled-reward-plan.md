# 去噪步数解耦奖励（Decoupled Reward）实现计划

## 目标

在 GRPO 训练中，为不同去噪阶段使用不同的 reward model：
- **早期去噪步**：使用 reward model A（如 aesthetic）
- **后期去噪步**：使用 reward model B（如 ocr）

## 当前代码流程分析

目前的奖励机制：
1. **采样阶段** (`train_sd3.py:618`): 对最终生成图像计算一个标量奖励
2. **奖励扩展** (`train_sd3.py:713`): 将这个标量 reward 通过 `.unsqueeze(1).repeat(1, num_train_timesteps)` 复制到所有去噪步
3. **优势计算** (`train_sd3.py:736`): 使用 `PerPromptStatTracker` 基于相同的 reward 计算所有步的 advantage
4. **训练** (`train_sd3.py:891-893`): 每个时间步 `j` 取 `sample["advantages"][:, j]` 作为该步的 advantage

关键点：**奖励只在最终图像上计算一次，然后均匀分配到所有时间步**。

## 设计方案：基于时间步的双奖励模型分配

在最终图像上分别计算两个 reward model 的奖励，然后按时间步分配不同的奖励值到不同的去噪步。

```
timestep 0 ... split_point ... num_steps-1
|--- reward_early ---|--- reward_late ---|
```

## 详细执行步骤

### Step 1: 修改配置系统 (`config/base.py` + `config/grpo.py`)

在 `config/base.py` 中增加以下配置项：

```python
# 双奖励模型配置
config.reward_fn_early = {}        # 早期步数使用的 reward model，格式同 reward_fn
config.reward_fn_late = {}         # 后期步数使用的 reward model，格式同 reward_fn
config.reward_split_ratio = 0.5    # 分界点比例，0.5 表示前50%步用 early，后50%步用 late
config.reward_decoupled = False    # 是否启用解耦奖励（兼容旧配置）
```

在 `config/grpo.py` 中添加新的配置函数，示例：

```python
def decoupled_reward_sd3():
    config = general_ocr_sd3()
    config.reward_decoupled = True
    config.reward_fn_early = {"aesthetic": 1.0}    # 早期关注美学质量
    config.reward_fn_late = {"ocr": 1.0}           # 后期关注文字准确度
    config.reward_split_ratio = 0.5
    # 原有的 reward_fn 仍可保留用于 eval
    return config
```

### Step 2: 修改 `train_sd3.py` — 初始化两个奖励函数

在 `train_sd3.py:434` 附近，添加条件初始化：

```python
# 原始代码
reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

# 新增
if config.reward_decoupled:
    reward_fn_early = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn_early)
    reward_fn_late = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn_late)
```

### Step 3: 修改采样阶段 — 计算双奖励

在 `train_sd3.py:618` 附近，修改奖励计算：

```python
# 原始代码
rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)

# 新增：解耦模式下提交两个 reward 计算任务
if config.reward_decoupled:
    rewards_early = executor.submit(reward_fn_early, images, prompts, prompt_metadata, only_strict=True)
    rewards_late = executor.submit(reward_fn_late, images, prompts, prompt_metadata, only_strict=True)
```

并在 `samples.append(...)` 中存储两组奖励：

```python
if config.reward_decoupled:
    samples.append({
        ...
        "rewards_early": rewards_early,
        "rewards_late": rewards_late,
    })
else:
    samples.append({
        ...
        "rewards": rewards,
    })
```

### Step 4: 修改奖励等待与拼接阶段

在 `train_sd3.py:640-651`，等待并处理双奖励：

```python
if config.reward_decoupled:
    for sample in samples:
        re, _ = sample["rewards_early"].result()
        rl, _ = sample["rewards_late"].result()
        sample["rewards_early"] = {
            key: torch.as_tensor(value, device=accelerator.device).float()
            for key, value in re.items()
        }
        sample["rewards_late"] = {
            key: torch.as_tensor(value, device=accelerator.device).float()
            for key, value in rl.items()
        }
```

### Step 5: 核心修改 — 按时间步分配不同奖励

替换 `train_sd3.py:709-713`，这是最关键的修改：

```python
if config.reward_decoupled:
    split_step = int(num_train_timesteps * config.reward_split_ratio)

    reward_early_avg = samples["rewards_early"]["avg"]  # (batch_size,)
    reward_late_avg = samples["rewards_late"]["avg"]     # (batch_size,)

    # 保存原始奖励用于日志
    samples["rewards"] = {}
    samples["rewards"]["ori_avg"] = (reward_early_avg + reward_late_avg) / 2  # 用于日志

    # 构造 (batch_size, num_train_timesteps) 的奖励张量
    timestep_rewards = torch.zeros(
        reward_early_avg.shape[0], num_train_timesteps, device=accelerator.device
    )
    timestep_rewards[:, :split_step] = reward_early_avg.unsqueeze(1).repeat(1, split_step)
    timestep_rewards[:, split_step:] = reward_late_avg.unsqueeze(1).repeat(
        1, num_train_timesteps - split_step
    )

    samples["rewards"]["avg"] = timestep_rewards  # (batch_size, num_train_timesteps)

    # 额外记录各 reward model 的详细分数
    for key, value in samples["rewards_early"].items():
        samples["rewards"][f"early_{key}"] = value
    for key, value in samples["rewards_late"].items():
        samples["rewards"][f"late_{key}"] = value

    del samples["rewards_early"]
    del samples["rewards_late"]
else:
    # 原始逻辑
    samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
    samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(
        1, num_train_timesteps
    )
```

### Step 6: PerPromptStatTracker 兼容性

查看 `stat_tracking.py` 代码，`update()` 方法使用 `axis=0` 和 `keepdims=True` 计算均值和标准差，已经可以处理 2D rewards `(N, T)`，会对每个时间步独立计算 per-prompt advantage。

**`flow_grpo/stat_tracking.py` 无需修改。**

### Step 7: 修改日志记录

在 `train_sd3.py:720-727`，添加双奖励的日志：

```python
if config.reward_decoupled:
    wandb.log(
        {
            "epoch": epoch,
            **{
                f"reward_early_{key}": value.mean()
                for key, value in gathered_rewards.items()
                if key.startswith("early_")
            },
            **{
                f"reward_late_{key}": value.mean()
                for key, value in gathered_rewards.items()
                if key.startswith("late_")
            },
            "reward_avg": gathered_rewards["ori_avg"].mean(),
        },
        step=global_step,
    )
```

### Step 8: 修改 `config/base.py` 添加默认值

确保新增配置项有默认值，不影响现有配置：

```python
config.reward_decoupled = False
config.reward_fn_early = {}
config.reward_fn_late = {}
config.reward_split_ratio = 0.5
```

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `config/base.py` | 添加 `reward_decoupled`, `reward_fn_early`, `reward_fn_late`, `reward_split_ratio` 配置项 |
| `config/grpo.py` | 添加使用双奖励的配置函数示例 |
| `scripts/train_sd3.py` | 1) 初始化两个 reward_fn; 2) 采样时计算双奖励; 3) 按时间步分配奖励; 4) 日志记录 |
| `flow_grpo/stat_tracking.py` | **无需修改**（已支持 2D rewards） |
| `flow_grpo/rewards.py` | **无需修改**（multi_score 已足够灵活） |

## 注意事项

1. **显存开销**：两个 reward model 同时加载会增加显存。如果显存紧张，可以让两个 reward 使用 remote server 模式（如 `pickscore_remote`）
2. **split_ratio 调参**：建议从 0.5 开始，根据实验结果调整。直觉上，早期步决定全局结构，后期步决定细节
3. **平滑过渡（可选增强）**：如果想要更平滑的过渡而非硬切分，可以用线性插值：

```python
# weight 从 1->0 线性变化
w = 1.0 - j / num_train_timesteps
timestep_rewards[:, j] = w * reward_early + (1 - w) * reward_late
```
