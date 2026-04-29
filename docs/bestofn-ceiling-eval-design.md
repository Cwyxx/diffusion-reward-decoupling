# Best-of-N 上限评估（Ceiling Evaluation）设计文档

## 1. 研究问题

Text-to-Image diffusion model 在大规模图文数据集上预训练，目标是数据分布拟合，本身不优化人类偏好。Post-Training 阶段引入 Human Preference Reward 微调模型，在 HPSv2 / PickScore 等指标上有明显提升。

**核心问题**：Post-Training 之后，模型的**生成能力上限**（不只是 Human Preference 维度）是否真的提升了？还是只是提高了 Human Preference Model 打高分图像的采样概率？

**评估方法**：用 Best-of-N 度量"上限"。对每个 prompt 生成 N 张图，取最好的一次结果。如果 Post-Training 只是改变采样分布而不扩展能力，base 与 RL'd 模型的 Best-of-N 曲线应随 N 增大而**收敛**；如果 Post-Training 真正扩展能力，曲线应**保持分离甚至发散**。

## 2. 评估配置

### 2.1 模型集合（共 7 个）

| 名称 | 说明 |
|---|---|
| base | SD-v1.5 原版（`runwayml/stable-diffusion-v1-5`） |
| dpo | Diffusion-DPO 公开 checkpoint |
| kto | Diffusion-KTO 公开 checkpoint |
| spo | SPO 公开 checkpoint |
| smpo | SmPO 公开 checkpoint |
| dro | Diffusion-DRO 公开 checkpoint |
| inpo | InPO 公开 checkpoint |

所有 RL checkpoint 均要求是基于 SD-v1.5 + Pick-a-Pic-v2 / PickScore 训练的公开发布版本。具体 HF repo 在 `evaluation/checkpoints/registry.py` 登记。

### 2.2 任务与数据集（共 3 个）

| 任务类型 | 数据集 | 路径 | Prompt 数 | 评分指标 |
|---|---|---|---|---|
| Human Preference | drawbench-unique | `dataset/drawbench-unique/test.txt` | ~200 | PickScore, HPSv3, DeQA, Aesthetic |
| Visual Text Rendering | ocr | `dataset/ocr/test.txt` | 1017 | OCR（二元：正确生成 / 错误） |
| Compositional Generation | geneval | `dataset/geneval/test_metadata.jsonl` | 553 | GenEval（二元） |

每张图**只用其 dataset 关心的 metric 评分**（不交叉），避免无谓评分成本。

### 2.3 推理超参（所有 method 统一）

- `num_inference_steps = 50`
- `guidance_scale = 7.5`
- `resolution = 512`
- `scheduler`：使用 diffusers pipeline 默认值（SD-v1.5 默认 `PNDMScheduler`），不主动覆盖
- `dtype`：fp32（3090 24GB 跑 SD-v1.5 fp32 推理显存充裕；去掉 fp16 数值噪声让 BoN 对比更干净）

某 method 原 paper 用了不同的推理超参（例如不同 scheduler）时，在 `verify-checkpoints.py` 阶段对比该 method 在默认 vs 原 paper 设定下的 PickScore；若差距 > 5% 标记 warning，由人工决定是否个例化。

### 2.4 Best-of-N 协议

- **N_max**：初始定 32（资源允许直接 32；紧张可从 16 起步，扩 N 通过 resumption 几乎免费）。
- **种子约定**：同一 (prompt, seed_index) 在所有 method 下使用相同初始噪声 →  paired comparison。每个 method 内 seed_index = 0..N_max-1。
- **Best-of-N 选择 — HP 类（连续 reward）**：每个 metric 独立选择（per-metric oracle）。
  ```
  BoN(N, M) = mean over prompts of  max_{i < N} M(x_i)
  ```
- **Best-of-N 选择 — OCR / GenEval（二元）**：naive `any()`。
  ```
  pass@N = mean over prompts of  any_{i < N}[ M(x_i) = 1 ]
  ```
- **N 取值**：聚合时计算 N ∈ {1, 2, ..., N_max} 全部值（成本可忽略），绘图时只显示 log-spaced 子集 {1, 2, 4, 8, 16, 32}。

### 2.5 输出

- 每个 (method, dataset, metric) 一条 BoN 曲线（log-x），不输出 headline 表格。
- 每条曲线对应一个原始 JSON `bestofn/curves.json`，可后续重新可视化。

## 3. 总体架构

三阶段流水线，沿用现有 `evaluation/` 目录的分层：

```
Stage 1: Generate                   Stage 2: Score              Stage 3: Aggregate
─────────────────────────────       ───────────────────────     ──────────────────────
generate-images-bestofn.py          score-images.py             aggregate-bestofn.py
   ↓                                  ↓                            ↓
images/{sid:05d}/{seed:05d}.png     evaluation_results.jsonl    bestofn/curves.json
manifest.json                                                    bestofn/plots/*.png
```

### 3.1 输出目录布局

```
${base_root}/bestofn-eval/sd-v1-5/
  └── ${method}/                                # base | dpo | kto | spo | smpo | dro | inpo
      └── ${dataset}/                           # drawbench-unique | ocr | geneval
          ├── images/{sample_id:05d}/{seed_index:05d}.png
          ├── manifest.json
          ├── evaluation_results.jsonl
          └── bestofn/
              ├── curves.json                   # 全 N 值结果 + 衍生指标
              └── plots/
                  └── {metric}_curve_log.png    # 7 method 一图
```

`${base_root}` 复用现有 `run-eval.sh` 中的根路径。子树 `bestofn-eval/sd-v1-5/` 与现有 `flow-grpo/sd-3-5-medium/` 完全平行，互不干扰。

### 3.2 Resumption（断点续跑）

**两个层次**：

1. **Generation resumption**（核心场景：N 从 32 扩到 64）：
   - 文件粒度。每张图独立路径 `images/{sid:05d}/{seed:05d}.png`。
   - 直接 `Image.save(path)` 写盘；崩溃残留的部分文件下次运行重生成（单图重生成成本低，不上原子写保险）。
   - 遍历所有 (sample_id, seed_index ∈ [0, N))，目标文件已存在则跳过；否则用 `torch.Generator().manual_seed(seed_index)` 生成。

2. **Scoring resumption**：
   - `evaluation_results.jsonl` schema 升级为 (sample_id, seed_index) 复合键：
     ```jsonl
     {"sample_id": 0, "seed_index": 0, "prompt": "...", "image_path": "...", "metadata": {...optional...}, "scores": {"pickscore": 21.3}}
     ```
   - 评分时已有该 metric 分则跳过；`--force` 强制重评。

### 3.3 Manifest

每个 `${method}/${dataset}/` 目录一个 `manifest.json`：

```json
{
  "method": "dpo",
  "dataset": "ocr",
  "checkpoint_id": "mhdang/dpo-sd1.5-text2image-v1",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "resolution": 512,
  "scheduler_class": "PNDMScheduler",
  "max_seed_generated": 31
}
```

`max_seed_generated` 仅作 informational 用途（快速读"已经生成到哪"），**文件系统是真相** —— 启动时仍以 `images/` 下的实际 PNG 为准遍历跳过。

启动时若 manifest 中超参与当前 CLI 不一致 → **拒绝续跑**，要求显式 `--force-regenerate` 或换目录。这避免了"扩 N 时悄悄改了 guidance_scale，前 32 张和后 32 张实际是不同分布"。

## 4. 模块设计

### 4.1 Checkpoint registry（`evaluation/checkpoints/`）

新增模块：

```
evaluation/checkpoints/
    __init__.py
    registry.py          # method_name -> CheckpointRecipe
    loaders.py           # 通用 loader: load_base / load_lora / load_unet / load_full
```

**`registry.py`**：

```python
@dataclass
class CheckpointRecipe:
    method: str
    base_model_id: str                 # "runwayml/stable-diffusion-v1-5"
    load_kind: Literal["base", "lora", "unet", "full"]
    repo_id: Optional[str]             # HF repo
    subfolder: Optional[str]
    extra_kwargs: Dict[str, Any]       # method-specific overrides (rare)

REGISTRY: Dict[str, CheckpointRecipe] = {
    "base": CheckpointRecipe(method="base", base_model_id=SD15, load_kind="base"),
    "dpo":  CheckpointRecipe(method="dpo",  base_model_id=SD15, load_kind="unet",
                             repo_id="mhdang/dpo-sd1.5-text2image-v1"),
    "kto":  ...,
    "spo":  ...,   # lora release with weight_name in extra_kwargs
    "smpo": ...,
    "dro":  ...,
    "inpo": ...,
}

def load_pipeline(method_name: str) -> StableDiffusionPipeline: ...
```

**`loaders.py`** 提供 4 个通用 loader，把 recipe 翻译为具体 diffusers 调用，返回统一 `StableDiffusionPipeline` 接口。

### 4.2 生成脚本（`evaluation/metrics/generate-images-bestofn.py`）

仿照现有 `generate-images.py`，差异如下：

| 维度 | 现有（SD3.5） | 新（SD-v1.5 BoN） |
|---|---|---|
| Pipeline 类 | `StableDiffusion3Pipeline` | `StableDiffusionPipeline` |
| Checkpoint 加载 | LoRA + 硬编码 target_modules | 通过 `evaluation/checkpoints/registry.py` |
| 每 prompt 图数 | 1 | N（`--n_max`） |
| 文件路径 | `images/{sid:05d}.png` | `images/{sid:05d}/{seed:05d}.png` |
| 写入策略 | 直接 save | 原子写 |
| 已存在文件 | 覆盖 | 跳过 |
| Manifest | 无 | 有，启动时检查 |
| Random generator | 每 batch `manual_seed(args.seed)` | 每 (sid, seed_index) `manual_seed(seed_index)` |

**Dataset loader 分支**：

```python
DATASET_LOADERS = {
    "drawbench-unique": load_txt,        # test.txt
    "ocr":              load_txt,        # test.txt
    "geneval":          load_jsonl,      # test_metadata.jsonl, 取 metadata["prompt"]
}
```

GenEval 的特殊性：scoring 时需要传 metadata（class、count 等）给 `geneval_score`。`evaluation_results.jsonl` 的行加 optional `metadata` 字段，drawbench/ocr 留空。

### 4.3 评分脚本（`evaluation/metrics/score-images.py` 修改）

**3 处修改**：

1. **AVAILABLE_METRICS 加入 `ocr` 和 `geneval`**：`flow_grpo.rewards` 已注册过这两个 scorer，仅需加进白名单。
2. **JSONL schema 升级**为 `(sample_id, seed_index)` 复合键 + optional `metadata` 字段。
3. **跳过已评分行**：默认行为；`--force` 覆盖。

**Conda env 切换**：沿用现有约定（hpsv3 → hpsv3 / deqa → internvl / visualquality_r1 → visualquality / 默认 alignprop）。`ocr` 和 `geneval` 的所属 env 在 prerequisite 阶段确认（见第 6 节）。

**Batch size**：默认从 2 提到 8（BoN 场景下评分图数为现有的 N 倍）；具体值在 verify 阶段量 OOM 边界。`SMALL_BATCH_METRICS = {"hpsv3", "visualquality_r1"}` 沿用 bs=1。

### 4.4 聚合脚本（`evaluation/metrics/aggregate-bestofn.py`，新增）

输入：`evaluation_results.jsonl`。
输出：

```
bestofn/
    curves.json     # {metric: {N: float, ...}, "ceiling_lift": float, "lift_over_base": ...}
    plots/
        {metric}_curve_log.png
```

**计算**：

- HP metric：按 prompt 分组 → 对每个 N ∈ [1, N_max] 算 `mean over prompts of max(scores[:N])`。
- OCR / GenEval：按 prompt 分组 → 对每个 N 算 `mean over prompts of any(scores[:N] > 0)`。
  - 假设：`flow_grpo.rewards.ocr_score` / `geneval_score` 在评分函数返回中给出 0/1（或可阈值化为 0/1）的标量。该假设在 pre-flight smoke test 阶段验证（评分非 NaN + 取值在 {0, 1} 或 [0, 1]）；若返回的是连续 score 而非二元，需要在 spec 加阈值化策略并重审。
- 衍生指标（在 curves.json 里多存两个数）：
  - `ceiling_lift = BoN(N_max) - BoN(1)`：method 自身的 N=1 vs N=N_max 差距。
  - `lift_over_base@N_max = method_BoN(N_max) - base_BoN(N_max)`：相对 base 的提升。

**绘图**：每个 metric 一张图，7 条线（一 method 一线），log-x，标注 N ∈ {1, 4, 16, 32}。

### 4.5 编排脚本

**`evaluation/run-bestofn.sh`**（与现有 `run-eval.sh` 平行，不替换）：

```bash
bash evaluation/run-bestofn.sh <cuda_device> <method> <dataset> <N_max>
# 例：
bash evaluation/run-bestofn.sh 0 dpo  drawbench-unique 32
bash evaluation/run-bestofn.sh 1 base ocr              32
```

职责：切默认 conda env → 调 `generate-images-bestofn.py` → 按 dataset 关心的 metric 切 conda env → 调 `score-images.py` → 调 `aggregate-bestofn.py`。

**矩阵分发**：手动通过 tmux + `CUDA_VISIBLE_DEVICES=k` 把 7 method × 3 dataset = 21 个组合分配到不同 GPU。**不引入 ray / multiprocessing.Pool**，与现有 `run-eval.sh` 风格一致。

## 5. Compute & 存储预算

- **图像生成**：7 method × 1770 prompts × 32 images = ~400K 张图。SD-v1.5 @ 512² 在 A100 上约 1–2 s/img → ~110–220 GPU-hours。8-GPU 节点 1–2 天可完成。
- **存储**：~400K × ~150 KB ≈ 60 GB。在 `${base_root}` 现有 partition 上确认空间。
- **评分**：每个 metric 评分速度差异较大（DeQA / VisualQuality 慢 ~5–10×）。整体评分时长估计 50–100 GPU-hours。

## 6. Verification gates 与 Risks

### 6.1 Pre-flight prerequisites（必须，按顺序）

> 评分环境（`ocr` / `geneval` 各自的 conda env / 依赖）的搭建不在本 spec 范围内 —— 不同 reward 各有独立环境依赖，由 implementation 阶段实操时按需准备，不作为 gate。

1. **Checkpoint 发现**：填好 `evaluation/checkpoints/registry.py`，并将 7 个 HF repo 在本地预先 cache（避免运行时网络抖动）。

2. **Smoke test gate `evaluation/checkpoints/verify-checkpoints.py`**：对每个 method
   - 加载 pipeline；
   - 用一个固定 prompt 生成 1 张图；
   - 用 PickScore 评分，确认非 NaN；
   - 若 method 原 paper 用了非默认 scheduler，**额外用原 paper scheduler 跑一次**，比较 PickScore 差距。差距 > 5% → flag warning，人工决定是否个例化。
   - 7 个全部通过才允许进入大规模生成阶段。

3. **数据集行数验证**：
   - drawbench-unique: 期望 ~200 prompts
   - ocr: 期望 1017 prompts
   - geneval: 期望 553 prompts
   行数不匹配 → 终止，不污染输出目录。

### 6.2 Sanity checks（在 aggregation 阶段自动检查）

1. 任意 (method, dataset, metric) 的 BoN 曲线必须在 N 上**单调非降**（数学上必然；不满足则代码 bug）。
2. RL methods 在 PickScore 上应**比 base 更快达到饱和**（训练对齐）。不满足则提示"checkpoint 是否真的训练于 PickScore？"。
3. `pass@1` from BoN 应等于 `mean over prompts of (seed_index=0 success indicator)` —— N=1 时两种公式一致。

### 6.3 Risks

| 风险 | 缓解 |
|---|---|
| 各 method checkpoint 加载格式不一 | Registry + smoke test gate |
| 默认 scheduler 与原 paper 不一致导致差异 | verify-checkpoints 的 5% 警戒线 |
| 磁盘 60 GB 占用 | pre-flight 检查 partition 剩余空间 |
| 110–220 GPU-hours 计算量 | tmux 多卡分片，resumption 容许中断 |
| RL paper checkpoint 实际训练于其他 reward（非 PickScore） | sanity check #2；spec 已要求 PickScore/Pick-a-Pic-v2 |

### 6.4 Out of scope

- 自行训练任一 RL method（仅用公开 checkpoint）。
- 跨任务评分（每张图只在所属 dataset 的 metric 上评分）。
- Bootstrap CI / 统计检验。
- Headline 表格（仅曲线输出）。
- N > 32 的初始承诺；通过 resumption 可后续扩展。

## 7. 任务拆解（feed 给 writing-plans）

最终 implementation 阶段会涉及的工作单元（粗粒度，writing-plans 阶段会进一步细化）：

1. `evaluation/checkpoints/{__init__,registry,loaders}.py`：填好 7 个 method recipe + 4 个 loader。
2. `evaluation/checkpoints/verify-checkpoints.py`：smoke test gate。
3. `evaluation/metrics/generate-images-bestofn.py`：BoN 生成脚本（resumption + manifest + 原子写）。
4. `evaluation/metrics/score-images.py` 修改：白名单加 ocr/geneval、schema 升级、跳过已评分行。
5. `evaluation/metrics/aggregate-bestofn.py`：曲线计算 + 绘图。
6. `evaluation/run-bestofn.sh`：单组合编排。
7. 跑 verify-checkpoints 通过 → 跑 base + 1 个 method × 1 个 dataset 小规模 dry run → 确认无误后铺全 21 组合。

## 8. 已确认决议

- **GenEval metadata 透传**：决定采用内嵌方案 —— `evaluation_results.jsonl` 加 optional `metadata` 字段（drawbench / ocr 留空，geneval 拷整条），scoring 自包含，不依赖反查 `test_metadata.jsonl`。
- **N_max 初始值**：32 一步到位。
- **多 GPU 分发方式**：手动 tmux + `CUDA_VISIBLE_DEVICES=k`，与现有 `run-eval.sh` 风格一致；不引入 ray / multiprocessing.Pool。
- **评分环境搭建**：不作 pre-flight gate；implementation 阶段按需准备，不同 reward 各有独立 conda env 依赖。
