# Qwen-Image-Bench → Best-of-N 评测体系整合设计

日期：2026-06-03
作者：Cwyxx（设计）/ Claude（执行）
状态：待用户复核

## 背景与目标

将 **Qwen-Image-Bench** 作为一个新的能力天花板基准，整合进现有 Best-of-N 评测管线
（`evaluation/run-bestofn.sh` 的 Generate → Score → Aggregate 三段式）。

Qwen-Image-Bench 本质：

- **Prompt 集**：1000 条（ID 1–1000）。正文**已在本地**
  `flow_grpo/Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl`（185MB，含各模型预生成
  response，本整合只用其 prompt/dims 字段）。每条记录含
  `{ID, prompt_cn, prompt_en, dims_cn, dims_en, ...}`——即 prompts 文件本身就带
  `dims_en`，**无需下载、无需再 join `bench_metadata.json`**。
- **评测方式**：用微调的 27B "Q-Judger"（HF `Qwen/Qwen-Image-Bench`，经 ms-swift
  PtEngine 推理）。对每张图、按该 prompt `dims_en` 涉及的每个 L1 维度，拼一个
  checklist prompt 让 judge 给每个 L3 facet 打 `0/1/2/N-A`；再 L3→L2→L1→Total
  自底向上聚合（映射 0→0、1→60、2→100），得到 0–100 的 **overall(Total)** 与 5 个
  **L1 维度分**（Quality / Aesthetics / Alignment / Real-world Fidelity /
  Creative Generation）。

固定推理参数：seed 42、temperature 0、top_k 1、top_p 1、repetition_penalty 1.05、
max_new_tokens 4096、enable_thinking True、**max_batch_size 1**（24 会 OOM）。

## 语言选择（用户拍板）

生成与判分统一使用 **`prompt_en` + `dims_en`**。理由：现有 BoN 跑 SD-v1.5 / SDXL /
SD-3.5-M，均为英文文本编码器，中文 prompt 会大面积失效。judge 也传 `prompt_en`、配
`dims_en` checklist，保证生成端与判分端一致。

## Best-of-N 语义（用户拍板）

对每条 prompt：

1. 生成 N 张候选图（family 默认分辨率/步数/CFG，沿用 `run-bestofn.sh`）。
2. 每张图按 Qwen-Image-Bench 标准算 **overall**（= 该 prompt 适用 L1 维度分的均值）。
3. 在前 n 张里取 **overall 最高的那张**作为该 prompt 在预算 n 下的代表图。
4. 用该代表图的分数报告 **Total + 5 个 L1 维度**，并对 n=1..N 扫出 BoN-at-n 曲线。

即：**用 overall 总分做选择器（argmax over N），再读出胜出图的维度剖面**。

设计性质（均已与用户确认）：

- 5 个维度分是 "Total 胜出图的伴随分"，**不是各自独立 max**，保证 5 维来自同一张真实图。
- Total 曲线随 n 单调上升（它就是 overall 的 max-over-n）；维度曲线**可能非单调**
  （换了 Total 胜出图，某维可能降），这是该设计的应有之义。
- 与"每维独立 max-over-N"相比，本设计不会拼出现实中不存在的"虚拟最优图"，更贴近
  Best-of-N 的真实用法（生成 N 张、按一个选择器挑 1 张交付）。

## 架构与组件

不改动任何现有数据集/指标。新增 1 个 `dataset` 与 1 个 `metric`，名均为
`qwen-image-bench`，复用 `flow_grpo/Qwen-Image-Bench` 里的现成评分逻辑。

### 1. Prompt 数据准备（`dataset/qwen-image-bench/`）

- `prepare.py`：从本地 `flow_grpo/Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl`
  抽 `{ID, prompt_en, dims_en}`，生成 `prompts.jsonl`，每行
  `{"prompt": <prompt_en>, "ID": int, "dims_en": str}`（按 ID 升序）。纯本地处理，
  无网络依赖。
- `generate-images-bestofn.py` 的 `_DATASET_LOADERS` 注册：
  `"qwen-image-bench": ("prompts.jsonl", _load_jsonl)`。`_load_jsonl` 已把整行塞入
  `metadata`，故 `metadata` 自带 `ID` 与 `dims_en`；`sample_id` 与 prompt 顺序对齐。
- 备注：很多 prompt 偏文字渲染/设计，SD 系预期得分偏低——这是能力天花板基准的预期
  表现，非缺陷。

### 2. 评审环境与模型（4×3090 多卡，无单 GPU 路径）

- ms-swift 与仓库 `transformers==4.40` 冲突，使用**独立 conda env `qwen-image-bench`**
  （ms-swift≥4.0 + judge 依赖，参考 `flow_grpo/Qwen-Image-Bench/requirements.txt`）。
- judge 模型 `Qwen/Qwen-Image-Bench`（27B，bf16 ≈ 54GB）**必须分片到 4×3090 24GB**
  （合计 96GB）。`MsSwiftJudge` 以 `device_map="auto"` 跨 4 卡加载；Stage2 该 metric
  的 `CUDA_VISIBLE_DEVICES` 设为全部 `--gpus`（不走单 GPU `score_gpu` 路径）。
- `max_batch_size=1`（OOM 约束）。

### 3. 打分器（`evaluation/metrics/score-images.py` 新增 in-place 分支）

仿 `wise` / `dpg-score-mplug` 的"独立路由"模式（27B 模型不走 `multi_score` 的
batched-on-cuda 契约）：

- 新函数 `_score_qwen_image_bench_in_place(output_dir, ...)`：
  - 复用 `flow_grpo/Qwen-Image-Bench` 的 `checklists.py`（SYSTEM_PROMPT /
    USER_PROMPT_TEMPLATE / DIM_TO_CHECKLIST / parse_dims_by_level1）、
    `score_utils.py`（extract_json / fix_score_json / compute_dimension_score）、
    `backends/ms_swift_backend.py`（MsSwiftJudge，max_batch_size=1、device_map auto）。
  - 模型只加载一次；遍历全部 `(sample_id, seed_index)` 图像。
  - 每张图：按其 `dims_en` 适用的 L1 维度跑 checklist 推理 → 解析 →
    `compute_dimension_score` → 得到 **适用 L1 维度分** 与 **overall = 适用维度均值**。
- 写入 `row["scores"]`（evaluation_results.jsonl，仅数值）：
  - `qwen-image-bench` = overall（**选择器键**，每张图都有）
  - `qwen-image-bench-quality` / `-aesthetics` / `-alignment` / `-fidelity` /
    `-creative`（**仅该 prompt 适用维度才写**）

### 3b. 保留 Q-Judger 原始判断（独立文件）

按用户要求，**每张图每个问题的判断结果单独落盘**，不塞进 evaluation_results.jsonl：

- 写 `${output_dir}/qwen_image_bench_judge_outputs.jsonl`，每行对应一张图：
  `{sample_id, seed_index, ID, image_path, prompt,
    judge_model_output: {<L1_dim>: <修正后的 L3 score json>},
    <dim>_judge_output: <judge 原始文本>}`
  —— 字段语义与 `flow_grpo/Qwen-Image-Bench/judge.py` 的 `_build_row_result` 一致。
- 该文件同样**支持断点续跑追加**：已写过的 `(sample_id, seed_index)` 不重复 judge。

### 3c. 断点续跑

- 复用 `score-images.py` 的 `_has_metric_score` / `METRIC_OUTPUT_KEYS` 机制。
  `METRIC_OUTPUT_KEYS["qwen-image-bench"] = ("qwen-image-bench",)`（overall 一定存在），
  按 `(sample_id, seed_index)` 检查 row 是否已含该键，有则跳过（数值 + 原始判断文件
  两端一致跳过）。
- 关键不变式：`dims_en` 按 ID（按 prompt）固定 → 同一 prompt 的 N 张图覆盖**相同**
  维度集合，故每个维度"该 prompt 要么全有、要么全无"，便于聚合按 prompt 取舍。

### 4. 选择式聚合（`evaluation/metrics/aggregate-bestofn.py` 新增分支）

现有 `bon_continuous` 是"每键各自 max"，不符合本设计。新增
`_aggregate_qwen_image_bench`（仿 `_aggregate_spatial_geneval`），在 `main()` 里以
`if any(k == "qwen-image-bench" for k in metrics)` 路由，并 `continue` 掉
`qwen-image-bench*` 各键避免落入通用 per-key max 路径：

- **Total 曲线**：对 overall 矩阵直接 `bon_continuous`（winner 的 overall 即 max
  overall，二者等价）。
- **新原语** `bon_select(overall_mat, dim_mat, n)`：每 prompt 在前 n 张里按
  `overall` 取 argmax → 读该胜出图的 `dim` 值 → 对"覆盖该维度的 prompt"求均值
  （其余 prompt 跳过）。
- 输出 6 条曲线键：`qwen-image-bench`(Total) + 5 个 `qwen-image-bench-<dim>`；
  写 `curves.json` / 各自 `csv` / 各自 `png`，外加一张 breakdown 合图（5 细线 + 1
  粗 Total），风格对齐 geneval / spatial-geneval。
- 不能用 `build_score_matrix`（对 NaN 报错）：维度矩阵按"适用 prompt 子集"单独构建；
  overall 矩阵则要求每 `(sid, seed)` 都有 overall（每张图必有，应满足完整性）。

### 5. `run-bestofn.sh` 接线

- `dataset` case 加：`qwen-image-bench) metric_list=(qwen-image-bench) ;;`
- `metric_env` 加：`[qwen-image-bench]=qwen-image-bench`
- Stage2 该 metric **强制多卡**：`score_cuda` 选 `${gpus}`（全部 4 卡），与
  dalleval-bias-gender/attribute 同一"全 GPU"分支并列加入 `case "${metric}"`。
- Stage3 走标准 aggregate（新分支自动接管），不进 unsafe / aigi / dalleval 的特殊跳过。

## 数据流总览

```
run-bestofn.sh "0,1,2,3" <method> qwen-image-bench <n_max>
  Stage1 Generate (env=alignprop, 多 GPU)
    └─ 每 prompt 生成 N 张 → evaluation_results.jsonl
       (row: sample_id, seed_index, prompt=prompt_en, metadata{ID,dims_en}, image_path)
  Stage2 Score (env=qwen-image-bench, 4 卡 device_map, batch=1, 断点续跑)
    ├─ 27B Q-Judger 逐图 → row.scores += {qwen-image-bench(overall), -<5 维>}
    └─ 原始判断 → qwen_image_bench_judge_outputs.jsonl（每图每问题）
  Stage3 Aggregate (env=alignprop, CPU)
    └─ 选择式 BoN → bestofn/curves.json + plots/*.png + csv/*.csv
       (Total + 5 维曲线 + breakdown 合图)
```

## 测试策略

- **单元（纯函数）**：`bon_select` 在小手造矩阵上验证——含非单调维度曲线、含
  非适用 prompt（应跳过）、Total 曲线与 `bon_continuous(overall)` 数值一致。
- **打分器解析**：用 `flow_grpo/Qwen-Image-Bench` 自带的解析链对一两条已知 judge
  原始输出做 round-trip（不需真跑 27B），验证 overall 与 5 维聚合数值正确。
- **端到端冒烟**：取 5–10 条 prompt、N=2、4 卡真跑 judge，确认三段贯通、
  断点续跑命中（第二次跑应全部 skip）、`curves.json` 6 键齐全且 Total 单调、
  `qwen_image_bench_judge_outputs.jsonl` 每图每问题齐全。

## 不做（YAGNI）

- 不实现 L2/L3 细分曲线（只到 Total + 5 个 L1；L3 明细仅存在原始判断文件里）。
- 不做"代理选 Top-K 再 judge"的省算力变体（用户已选全量 judge）。
- 不做中文（prompt_cn/dims_cn）路径（用户已选 EN）。
- 不改动其它数据集/指标的任何行为。
- 不保留单 GPU judge 路径（27B 必须 4 卡）。

## 待确认 / 风险

1. **ms-swift device_map 跨 4 卡**：`MsSwiftJudge` 现以 `TransformersEngine(model_path,
   max_batch_size=...)` 加载；需确认 ms-swift≥4.0 该接口能 `device_map="auto"` 把 27B
   分片到 4×3090（或改用相应环境变量/参数）。安装时对齐。
2. **判分吞吐**：batch=1 + thinking + 1000×N 张×多维 → 单次跑很慢；断点续跑是关键。
3. **ms-swift 版本/接口漂移**：`ms_swift_backend.py` 依赖 ms-swift≥4.0 的
   `TransformersEngine` 接口。
