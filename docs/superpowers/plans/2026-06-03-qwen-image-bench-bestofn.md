# Qwen-Image-Bench Best-of-N Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Qwen-Image-Bench（27B Q-Judger）接进 `evaluation/run-bestofn.sh`，新增一个 `dataset=qwen-image-bench` 与 `metric=qwen-image-bench`：用 `prompt_en`/`dims_en` 生成→逐图判分（Total + 5 个 L1 维度）→选择式 BoN 聚合（按 overall argmax 选图、带出维度剖面），并保留每图每问题的原始判断。

**Architecture:** 沿用现有三段式（Generate/Score/Aggregate）。判分器仿 `wise`/`dpg-score-mplug` 的 in-place 路由（27B 不走 `multi_score` 的 batched-on-cuda 契约），复用 `flow_grpo/Qwen-Image-Bench` 的 `checklists.py`/`score_utils.py`/`backends/ms_swift_backend.py`。纯逻辑（prompt 准备、judge 解析聚合、BoN 选择式聚合）全部 TDD；27B 模型相关部分用 4×3090 冒烟验证。

**Tech Stack:** Python, numpy, pandas, PIL, matplotlib, ms-swift≥4.0（独立 conda env `qwen-image-bench`），bash。

**关键路径前提：**
- 本地 prompt 文件：`flow_grpo/Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl`（每行含 `ID, prompt_en, dims_en, ...`）。
- 判分模型：`Qwen/Qwen-Image-Bench`（27B），4×3090 24GB `device_map="auto"`，`max_batch_size=1`。
- 仓库根（worktree）：所有相对路径以仓库根为基准。`evaluation/metrics/` 脚本内已有 `_REPO_ROOT` 注入 `sys.path`。

**约定：** 所有 `python`/`pytest` 命令在仓库根目录、conda env `alignprop` 下运行（纯逻辑测试不需要 GPU/ms-swift）。判分相关运行在 env `qwen-image-bench`、4×3090 机器上。

**Worktree dev-setup（已由 controller 完成，实现者无需重做）：** `flow_grpo/Qwen-Image-Bench/` 是父仓库未跟踪的独立上游克隆，只存在于主 checkout。worktree 里已建一个**只读软链** `flow_grpo/Qwen-Image-Bench → 主 checkout 真实目录**，使 `checklists.py`/`score_utils.py` 可被 import、Task 3 测试可在 worktree 跑通；该软链不提交。因此**本计划不修改上游 `flow_grpo/Qwen-Image-Bench` 内任何文件**——所有改动都落在父仓库跟踪树（`dataset/`、`evaluation/`）。judge 引擎封装在我们自己的树内新建（见 Task 4），仅 import 复用上游的 `checklists`/`score_utils`（只读）。

---

## File Structure

新增：
- `dataset/qwen-image-bench/prepare.py` — 从本地 jsonl 抽 `{prompt(en), ID, dims_en}` 生成 `prompts.jsonl`。
- `dataset/qwen-image-bench/prompts.jsonl` — 生成产物（入库，1000 行）。
- `dataset/qwen-image-bench/test_prepare.py` — prepare 纯逻辑测试。
- `evaluation/metrics/qwen_image_bench_judge.py` — judge 复用层（不 import swift）：构造 checklist 任务、从原始判断算 overall+5 维、图像加载/缩放。
- `evaluation/metrics/test_qwen_image_bench_judge.py` — 上面纯逻辑测试（用 stub judge）。
- `evaluation/metrics/test_qwen_image_bench_agg.py` — `bon_select` + `_aggregate_qwen_image_bench` 测试。

- `evaluation/metrics/qwen_image_bench_engine.py` — ms-swift TransformersEngine 封装（batch=1 + device_map=auto），等价上游 `MsSwiftJudge` 但落在我们树内（不改上游）。

修改：
- `evaluation/metrics/generate-images-bestofn.py` — 注册 dataset loader。
- `evaluation/metrics/score-images.py` — `AVAILABLE_METRICS` + `_score_qwen_image_bench_in_place` + dispatch。
- `evaluation/metrics/aggregate-bestofn.py` — `bon_select` + `_aggregate_qwen_image_bench` + 路由。
- `evaluation/run-bestofn.sh` — dataset case / metric_env / 多卡 case / 判分模型 env。

---

## Task 1: Prompt 数据准备（prepare.py + prompts.jsonl）

**Files:**
- Create: `dataset/qwen-image-bench/prepare.py`
- Create: `dataset/qwen-image-bench/test_prepare.py`
- Generate+commit: `dataset/qwen-image-bench/prompts.jsonl`

- [ ] **Step 1: 写失败测试**

Create `dataset/qwen-image-bench/test_prepare.py`:

```python
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prepare import build_prompts


def test_build_prompts_extracts_en_and_sorts(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [
        {"ID": 2, "prompt_en": "a red cube", "prompt_cn": "红色立方体",
         "dims_en": "Quality / Realism / Physical Logic", "dims_cn": "x", "junk": 1},
        {"ID": 1, "prompt_en": "a blue sphere", "prompt_cn": "蓝色球",
         "dims_en": "Aesthetics / Composition / Composition", "dims_cn": "y"},
    ]
    with open(src, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = build_prompts(str(src))

    assert [r["ID"] for r in out] == [1, 2]           # sorted by ID
    assert out[0]["prompt"] == "a blue sphere"        # prompt == prompt_en
    assert out[0]["dims_en"] == "Aesthetics / Composition / Composition"
    assert set(out[0].keys()) == {"prompt", "ID", "dims_en"}  # only the 3 fields
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest dataset/qwen-image-bench/test_prepare.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'prepare'`）

- [ ] **Step 3: 写实现**

Create `dataset/qwen-image-bench/prepare.py`:

```python
"""Build dataset/qwen-image-bench/prompts.jsonl from the local Qwen-Image-Bench
prompts file (flow_grpo/Qwen-Image-Bench/qwen_image_bench_hf_v0518.jsonl).

We use the ENGLISH prompt (prompt_en) + English dimension spec (dims_en), since
the Best-of-N pipeline targets English text-to-image models (SD-v1.5/SDXL/SD-3.5-M).
Each output row: {"prompt": <prompt_en>, "ID": int, "dims_en": str}, sorted by ID.
"""
import json
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SRC = os.path.join(
    _REPO_ROOT, "flow_grpo", "Qwen-Image-Bench", "qwen_image_bench_hf_v0518.jsonl"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "prompts.jsonl")


def build_prompts(src_path):
    """Read the source jsonl, return list of {prompt, ID, dims_en} sorted by ID."""
    out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "prompt": r["prompt_en"],
                "ID": int(r["ID"]),
                "dims_en": r["dims_en"],
            })
    out.sort(key=lambda x: x["ID"])
    return out


def main():
    rows = build_prompts(DEFAULT_SRC)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest dataset/qwen-image-bench/test_prepare.py -v`
Expected: PASS

- [ ] **Step 5: 生成真实 prompts.jsonl 并自检**

Run:
```bash
python dataset/qwen-image-bench/prepare.py
wc -l dataset/qwen-image-bench/prompts.jsonl
head -1 dataset/qwen-image-bench/prompts.jsonl
```
Expected: `1000 dataset/qwen-image-bench/prompts.jsonl`；首行是合法 JSON，含 `prompt`/`ID`/`dims_en`，`prompt` 为英文。

- [ ] **Step 6: 提交**

```bash
git add dataset/qwen-image-bench/prepare.py dataset/qwen-image-bench/test_prepare.py dataset/qwen-image-bench/prompts.jsonl
git commit -m "feat(qwen-image-bench): prompt prep (prompt_en/dims_en) + prompts.jsonl"
```

---

## Task 2: 注册 generate dataset loader

**Files:**
- Modify: `evaluation/metrics/generate-images-bestofn.py`（`_DATASET_LOADERS`，约 87–100 行）

- [ ] **Step 1: 加注册项**

在 `_DATASET_LOADERS` 字典里、`"anytext-zh"` 行之后加入一行：

```python
    "anytext-en":       ("test.jsonl",            _load_jsonl),
    "anytext-zh":       ("test.jsonl",            _load_jsonl),
    "qwen-image-bench": ("prompts.jsonl",         _load_jsonl),
```

（`_load_jsonl` 读取 `row["prompt"]` 作为提示词，并把整行存入 `metadata`，因此 `metadata` 自带 `ID` 与 `dims_en`。）

- [ ] **Step 2: 验证 loader 可用**

Run:
```bash
python -c "
import importlib.util, os
spec = importlib.util.spec_from_file_location('g', 'evaluation/metrics/generate-images-bestofn.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items = m.load_prompts('qwen-image-bench')
print('n=', len(items))
print('keys0=', sorted(items[0].keys()))
print('meta keys=', sorted(items[0]['metadata'].keys()))
assert len(items) == 1000
assert 'dims_en' in items[0]['metadata'] and 'ID' in items[0]['metadata']
print('OK')
"
```
Expected: `n= 1000`、metadata 含 `dims_en` 与 `ID`、打印 `OK`。

- [ ] **Step 3: 提交**

```bash
git add evaluation/metrics/generate-images-bestofn.py
git commit -m "feat(qwen-image-bench): register generate dataset loader"
```

---

## Task 3: judge 复用层（qwen_image_bench_judge.py，纯逻辑）

该模块**不 import swift/ms-swift**，只依赖 `flow_grpo/Qwen-Image-Bench` 的 `checklists.py`/`score_utils.py` 与 PIL，故可本地单测。judge 模型由调用方（Task 5）注入。

**Files:**
- Create: `evaluation/metrics/qwen_image_bench_judge.py`
- Create: `evaluation/metrics/test_qwen_image_bench_judge.py`

- [ ] **Step 1: 写失败测试**

Create `evaluation/metrics/test_qwen_image_bench_judge.py`:

```python
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from qwen_image_bench_judge import (
    DIM_KEY,
    build_tasks,
    load_and_resize_image,
    scores_from_raw,
)


def test_load_and_resize_keeps_small_and_shrinks_large(tmp_path):
    small = tmp_path / "s.png"
    Image.new("RGB", (512, 512), "red").save(small)
    assert load_and_resize_image(str(small)).size == (512, 512)

    big = tmp_path / "b.png"
    Image.new("RGB", (2048, 2048), "blue").save(big)
    assert load_and_resize_image(str(big)).size == (1024, 1024)


def test_build_tasks_one_per_applicable_l1():
    dims_en = ("Quality / Realism / Physical Logic; "
               "Aesthetics / Composition / Composition")
    img = Image.new("RGB", (64, 64))
    tasks = build_tasks("a cat", dims_en, img)
    l1s = [l1 for l1, _ in tasks]
    assert l1s == ["Quality", "Aesthetics"]
    # checklist text for the dim must be embedded in the user prompt
    quality_task = tasks[0][1]
    assert "Physical Logic" in quality_task["user_text"]
    assert quality_task["image"] is img


def test_scores_from_raw_maps_and_aggregates():
    # Quality: Realism has 2 facets (1->60, 2->100 => L2=80);
    #          Resolution has 1 facet (0->0 => L2=0). L1 = mean(80,0)=40.
    quality_raw = (
        '{"Realism": {"Physical Logic": {"score": 1}, '
        '"Material Texture": {"score": 2}}, '
        '"Resolution": {"Resolution": {"score": 0}}}'
    )
    # Aesthetics: single facet 2 -> 100 => L1 = 100.
    aesth_raw = '{"Composition": {"Composition": {"score": 2}}}'

    overall, dim_scores, parsed = scores_from_raw(
        {"Quality": quality_raw, "Aesthetics": aesth_raw}
    )
    assert dim_scores[DIM_KEY["Quality"]] == 40.0
    assert dim_scores[DIM_KEY["Aesthetics"]] == 100.0
    assert overall == 70.0          # mean(40, 100)
    assert "Quality" in parsed and "Aesthetics" in parsed


def test_scores_from_raw_unparseable_dim_skipped():
    overall, dim_scores, parsed = scores_from_raw(
        {"Quality": "not json at all"}
    )
    assert overall is None
    assert dim_scores == {}
    assert parsed == {}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest evaluation/metrics/test_qwen_image_bench_judge.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qwen_image_bench_judge'`）

- [ ] **Step 3: 写实现**

Create `evaluation/metrics/qwen_image_bench_judge.py`:

```python
"""Reusable Qwen-Image-Bench judging logic, decoupled from the ms-swift engine.

Builds the per-L1-dimension checklist inference tasks and turns raw judge text
into a per-image overall score + 5 L1 dimension scores. Imports only the
checklist/score helpers from flow_grpo/Qwen-Image-Bench (no swift dependency),
so it is unit-testable without the 27B model. The judge engine is injected by
the caller (score-images.py).
"""
import os
import sys

from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_QIB_DIR = os.path.join(_REPO_ROOT, "flow_grpo", "Qwen-Image-Bench")
if _QIB_DIR not in sys.path:
    sys.path.insert(0, _QIB_DIR)

from checklists import (  # noqa: E402
    DIM_TO_CHECKLIST,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_dims_by_level1,
)
from score_utils import (  # noqa: E402
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

# L1 dimension name -> score-key suffix used in evaluation_results.jsonl.
DIM_KEY = {
    "Quality": "quality",
    "Aesthetics": "aesthetics",
    "Alignment": "alignment",
    "Real-world Fidelity": "fidelity",
    "Creative Generation": "creative",
}


def load_and_resize_image(path):
    """Load RGB image; resize to 1024x1024 if any side > 1024 (matches judge.py)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
    img.load()
    return img


def build_tasks(prompt, dims_en, image):
    """Return [(level1_dim, task_dict), ...] for the L1 dimensions this prompt covers.

    task_dict matches MsSwiftJudge.generate_batch's item schema:
    {"system_prompt": str, "user_text": str, "image": PIL.Image}.
    """
    dims_by_l1 = parse_dims_by_level1(dims_en)
    tasks = []
    for level1_dim in dims_by_l1:
        if level1_dim not in DIM_TO_CHECKLIST:
            continue
        user_text = USER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            level1_dim=level1_dim,
            format_checklist=DIM_TO_CHECKLIST[level1_dim],
        )
        tasks.append((level1_dim, {
            "system_prompt": SYSTEM_PROMPT,
            "user_text": user_text,
            "image": image,
        }))
    return tasks


def scores_from_raw(raw_by_dim):
    """Turn {L1_dim: raw_judge_text} into (overall, dim_scores, parsed_by_dim).

    overall = mean of non-None L1 scores; dim_scores keyed by DIM_KEY suffix
    (only dims that parsed and yielded a non-None L1 score). Unparseable dims
    are skipped. parsed_by_dim holds the fixed L3 score JSON per dim (for the
    raw-judgments file).
    """
    parsed_by_dim = {}
    l1_by_dim = {}
    for level1_dim, raw in raw_by_dim.items():
        score_json = extract_json_from_response(raw)
        if score_json is None:
            continue
        fixed = fix_score_json(score_json, level1_dim)
        parsed_by_dim[level1_dim] = fixed
        l1_by_dim[level1_dim] = compute_dimension_score(fixed)["level1_score"]

    valid = [v for v in l1_by_dim.values() if v is not None]
    overall = sum(valid) / len(valid) if valid else None
    dim_scores = {
        DIM_KEY[d]: v for d, v in l1_by_dim.items()
        if d in DIM_KEY and v is not None
    }
    return overall, dim_scores, parsed_by_dim
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest evaluation/metrics/test_qwen_image_bench_judge.py -v`
Expected: PASS（4 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add evaluation/metrics/qwen_image_bench_judge.py evaluation/metrics/test_qwen_image_bench_judge.py
git commit -m "feat(qwen-image-bench): judge reuse layer (tasks + score aggregation)"
```

---

## Task 4: judge 引擎封装（tracked 树内，batch=1 + device_map）

不修改未跟踪的上游 `flow_grpo/Qwen-Image-Bench/backends/ms_swift_backend.py`（在 worktree 里无法干净提交、且属用户外部克隆）。改为在我们自己的树内新建一个等价封装，pin `max_batch_size=1`（24 在 3090 上 OOM）与 `device_map="auto"`（把 27B 分片到 4×3090）。该模块 import `swift`（ms-swift≥4.0，仅 GPU 机器的 `qwen-image-bench` env 有），故本机不导入、不单测，只做语法检查；真实加载在 Task 8 冒烟测试验证。

**Files:**
- Create: `evaluation/metrics/qwen_image_bench_engine.py`

- [ ] **Step 1: 写封装**

Create `evaluation/metrics/qwen_image_bench_engine.py`:

```python
"""ms-swift TransformersEngine wrapper for the 27B Qwen-Image-Bench Q-Judger.

Vendored into the tracked tree (instead of editing the untracked upstream
flow_grpo/Qwen-Image-Bench/backends/ms_swift_backend.py) so all integration code
lives in this repo. Mirrors upstream's MsSwiftJudge but pins max_batch_size=1
(24 OOMs on 3090s) and device_map="auto" to shard the 27B across the 4x3090
cards. Imports `swift` (ms-swift>=4.0), only present in the qwen-image-bench
conda env, so this module is not imported at unit-test time.

Fixed decoding to match Qwen-Image-Bench: seed 42, temperature 0, top_k 1,
top_p 1.0, repetition_penalty 1.05, enable_thinking True.
"""
from swift import TransformersEngine, RequestConfig, InferRequest


class QwenImageBenchJudge:
    def __init__(self, model_path, max_batch_size=1, max_new_tokens=4096,
                 device_map="auto"):
        # device_map="auto" shards the 27B across all CUDA_VISIBLE_DEVICES.
        # Older ms-swift TransformersEngine may not accept device_map -> fall
        # back to its default placement.
        try:
            self.engine = TransformersEngine(
                model_path, max_batch_size=max_batch_size, device_map=device_map,
            )
        except TypeError:
            self.engine = TransformersEngine(model_path, max_batch_size=max_batch_size)
        self.request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )
        # Enable Qwen3 thinking mode on the engine's default template.
        try:
            self.engine.default_template.template_meta.template_kwargs = {
                "enable_thinking": True
            }
        except AttributeError:
            pass

    def generate_batch(self, items):
        """Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}.
        Returns list of generated text strings (one per item)."""
        infer_requests = []
        for item in items:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_text"]},
            ]
            infer_requests.append(
                InferRequest(messages=messages, images=[item["image"]])
            )
        resp_list = self.engine.infer(infer_requests, self.request_config)
        return [r.choices[0].message.content for r in resp_list]
```

- [ ] **Step 2: 静态检查（不导入 swift）**

Run: `python -c "import ast; ast.parse(open('evaluation/metrics/qwen_image_bench_engine.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: 提交**

```bash
git add evaluation/metrics/qwen_image_bench_engine.py
git commit -m "feat(qwen-image-bench): ms-swift engine wrapper (batch=1, 4-GPU device_map)"
```

---

## Task 5: 打分器 in-place 分支（score-images.py）

判分器：遍历 `todo` 行，按 `metadata.dims_en` 跑 27B judge，写 `row["scores"]` 的 `qwen-image-bench`(overall) 与 `qwen-image-bench-<dim>`，并把每图每问题的原始判断追加到 `qwen_image_bench_judge_outputs.jsonl`。该文件兼作判分缓存：崩溃后重跑可复用已有判断、免再跑 27B。

**Files:**
- Modify: `evaluation/metrics/score-images.py`（`AVAILABLE_METRICS` 约 26–33 行；新增函数；`main` dispatch 约 1075–1133 行）

- [ ] **Step 1: 把 metric 加入 `AVAILABLE_METRICS`**

在 `AVAILABLE_METRICS` 列表末尾（`"anytext-ocr",` 之后）加：

```python
    "anytext-ocr",
    "qwen-image-bench",
]
```

- [ ] **Step 2: 新增判分器函数**

在 `_score_mhsc_in_place` 定义之后（约 984 行后）插入：

```python
# ------------------------- Qwen-Image-Bench 27B Q-Judger -------------------------
# Per image: for each L1 dimension the prompt covers (metadata.dims_en), run the
# fine-tuned 27B judge over its checklist, parse 0/1/2/N-A -> L3/L2/L1 scores, and
# write overall(Total) + 5 L1 dimension scores. The 27B model shards across all
# visible GPUs (device_map="auto"), max_batch_size=1. Raw per-question judgments
# are appended to qwen_image_bench_judge_outputs.jsonl, which doubles as a
# judgment cache so a crash-resumed run reuses prior judge calls.

_QIB_RAW_FILENAME = "qwen_image_bench_judge_outputs.jsonl"


def _qib_load_raw_cache(raw_path):
    """Return {(sample_id, seed_index): {L1_dim: raw_text}} from prior runs."""
    cache = {}
    if not os.path.exists(raw_path):
        return cache
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["sample_id"], rec.get("seed_index", 0))
            cache[key] = rec.get("raw_by_dim", {})
    return cache


def _qib_append_raw(raw_path, row, raw_by_dim, parsed_by_dim):
    """Append one image's raw + parsed judgments (crash-safe progress)."""
    meta = row.get("metadata") or {}
    rec = {
        "sample_id": row["sample_id"],
        "seed_index": row.get("seed_index", 0),
        "ID": meta.get("ID"),
        "image_path": row["image_path"],
        "prompt": row["prompt"],
        "raw_by_dim": raw_by_dim,           # {L1_dim: raw judge text}
        "parsed_by_dim": parsed_by_dim,     # {L1_dim: fixed L3 score json}
    }
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _score_qwen_image_bench_in_place(todo_rows, output_dir):
    from qwen_image_bench_judge import build_tasks, load_and_resize_image, scores_from_raw

    n = len(todo_rows)
    if n == 0:
        return

    raw_path = os.path.join(output_dir, _QIB_RAW_FILENAME)
    cache = _qib_load_raw_cache(raw_path)
    model_path = os.environ.get("QIB_JUDGE_MODEL", "Qwen/Qwen-Image-Bench")
    max_new_tokens = int(os.environ.get("QIB_MAX_NEW_TOKENS", "4096"))
    print(f"[qwen-image-bench] {n} images to score; model={model_path}; "
          f"raw cache hits available={len(cache)}")

    judge = None  # lazy: only construct the 27B engine if a row misses the cache
    for r in tqdm(todo_rows, desc="qwen-image-bench"):
        meta = r.get("metadata") or {}
        if "dims_en" not in meta:
            raise KeyError(
                f"qwen-image-bench needs metadata.dims_en; row for "
                f"{r['image_path']} has metadata keys {sorted(meta.keys())}"
            )
        key = (r["sample_id"], r.get("seed_index", 0))

        if key in cache:
            raw_by_dim = cache[key]
            parsed_by_dim = None  # already on disk; don't re-append
        else:
            if judge is None:
                from qwen_image_bench_engine import QwenImageBenchJudge  # noqa: E402
                judge = QwenImageBenchJudge(model_path=model_path, max_batch_size=1,
                                            max_new_tokens=max_new_tokens)
            img = load_and_resize_image(r["image_path"])
            tasks = build_tasks(r["prompt"], meta["dims_en"], img)
            outputs = judge.generate_batch([t for _, t in tasks])
            raw_by_dim = {l1: out for (l1, _), out in zip(tasks, outputs)}

        overall, dim_scores, parsed_by_dim_new = scores_from_raw(raw_by_dim)
        if key not in cache:
            _qib_append_raw(raw_path, r, raw_by_dim, parsed_by_dim_new)

        r["scores"]["qwen-image-bench"] = overall
        for suffix, val in dim_scores.items():
            r["scores"][f"qwen-image-bench-{suffix}"] = val
```

注：`qwen_image_bench_engine`/`qwen_image_bench_judge` 与 score-images.py 同在
`evaluation/metrics/`；score-images.py 以脚本方式运行（`python evaluation/metrics/score-images.py`），
Python 会自动把脚本所在目录加入 `sys.path[0]`，故 `from qwen_image_bench_judge import ...`/
`from qwen_image_bench_engine import ...` 可直接解析。`qwen_image_bench_judge` 在导入时再把
软链/真实的 `flow_grpo/Qwen-Image-Bench` 加入 `sys.path`，复用 `checklists`/`score_utils`（只读）。

- [ ] **Step 3: 在 `main` 里加 dispatch**

在 `main` 的 metric dispatch 链中、`anytext-ocr` 分支之后加：

```python
        if metric == "anytext-ocr":
            _score_anytext_in_place(todo)
            continue

        if metric == "qwen-image-bench":
            _score_qwen_image_bench_in_place(todo, args.output_dir)
            continue
```

- [ ] **Step 4: 静态检查 + dispatch 连通性（mock 模型，不跑 27B）**

Run:
```bash
python -c "import ast; ast.parse(open('evaluation/metrics/score-images.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

再做一个不依赖模型的缓存命中测试（验证缓存命中时不构造 judge）：
```bash
python - <<'PY'
import json, os, sys, tempfile
sys.path.insert(0, "evaluation/metrics")
import importlib.util
spec = importlib.util.spec_from_file_location("score_images", "evaluation/metrics/score-images.py")
# 仅导入函数定义而不触发 main：score-images.py 顶层 import 了 flow_grpo.rewards，
# 若该 import 过重可改为直接复制 _qib_load_raw_cache 验证。这里只验证缓存读取逻辑：
PY
echo "cache-roundtrip checked in Task 8 smoke test"
```
说明：score-images.py 顶层 `from flow_grpo.rewards import multi_score` 较重，缓存命中/judge-lazy 路径的完整验证放入 Task 8 冒烟测试（第二次重跑应 0 次 27B 调用）。本步只保证语法正确与 dispatch 接好。

- [ ] **Step 5: 提交**

```bash
git add evaluation/metrics/score-images.py
git commit -m "feat(qwen-image-bench): in-place 27B judge scorer with raw-judgment cache + resume"
```

---

## Task 6: 选择式 BoN 聚合（aggregate-bestofn.py）

**Files:**
- Modify: `evaluation/metrics/aggregate-bestofn.py`（新增 `bon_select`；新增 `_aggregate_qwen_image_bench` 与 breakdown 画图；`main` 路由）
- Create: `evaluation/metrics/test_qwen_image_bench_agg.py`

- [ ] **Step 1: 写失败测试**

Create `evaluation/metrics/test_qwen_image_bench_agg.py`:

```python
import importlib.util
import os

import numpy as np

_AGG = os.path.join(os.path.dirname(__file__), "aggregate-bestofn.py")
_spec = importlib.util.spec_from_file_location("agg_bestofn", _AGG)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_bon_select_picks_overall_argmax_and_reads_companion_dim():
    # 1 prompt, 3 seeds. overall maxes at seed1 from n>=2.
    overall = np.array([[10.0, 50.0, 30.0]])
    dim = np.array([[100.0, 0.0, 40.0]])
    # n=1 -> only seed0 -> dim 100
    assert agg.bon_select(overall, dim, 1) == 100.0
    # n=2 -> argmax overall in {10,50}=seed1 -> dim 0  (non-monotonic drop)
    assert agg.bon_select(overall, dim, 2) == 0.0
    # n=3 -> argmax overall still seed1 -> dim 0
    assert agg.bon_select(overall, dim, 3) == 0.0


def test_total_curve_equals_bon_continuous_of_overall():
    overall = np.array([[10.0, 50.0, 30.0], [70.0, 20.0, 60.0]])
    for n in (1, 2, 3):
        # selecting overall's companion == overall itself equals max-over-n
        assert agg.bon_select(overall, overall, n) == agg.bon_continuous(overall, n)


def _row(sid, seed, scores):
    return {"sample_id": sid, "seed_index": seed, "prompt": f"p{sid}",
            "image_path": f"/img/{sid}_{seed}.png", "metadata": {"ID": sid},
            "scores": scores}


def test_aggregate_only_averages_dim_over_covering_prompts(tmp_path):
    # prompt 0 covers quality+alignment; prompt 1 covers only quality.
    rows = [
        _row(0, 0, {"qwen-image-bench": 40.0, "qwen-image-bench-quality": 60.0,
                    "qwen-image-bench-alignment": 0.0}),
        _row(0, 1, {"qwen-image-bench": 80.0, "qwen-image-bench-quality": 100.0,
                    "qwen-image-bench-alignment": 60.0}),
        _row(1, 0, {"qwen-image-bench": 60.0, "qwen-image-bench-quality": 60.0}),
        _row(1, 1, {"qwen-image-bench": 20.0, "qwen-image-bench-quality": 20.0}),
    ]
    bestofn = tmp_path / "bestofn"
    plots = bestofn / "plots"
    csvd = bestofn / "csv"
    for d in (plots, csvd):
        os.makedirs(d, exist_ok=True)

    out = agg._aggregate_qwen_image_bench(rows, str(bestofn), str(plots), str(csvd))

    # Total at n=2: prompt0 max(40,80)=80; prompt1 max(60,20)=60 -> mean 70.
    assert out["qwen-image-bench"]["curve"][2] == 70.0
    # quality at n=2: prompt0 winner=seed1(overall80)->100; prompt1 winner=seed0(60)->60 -> mean 80.
    assert out["qwen-image-bench-quality"]["curve"][2] == 80.0
    # alignment only prompt0 covers it: winner seed1 -> 60. mean over 1 prompt = 60.
    assert out["qwen-image-bench-alignment"]["curve"][2] == 60.0
    assert out["qwen-image-bench-alignment"]["num_prompts"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest evaluation/metrics/test_qwen_image_bench_agg.py -v`
Expected: FAIL（`AttributeError: module ... has no attribute 'bon_select'`）

- [ ] **Step 3: 加 `bon_select` 原语**

在 `aggregate-bestofn.py` 的 `bon_continuous` 定义之后（约 133 行后）加：

```python
def bon_select(overall_mat: np.ndarray, dim_mat: np.ndarray, n: int) -> float:
    """Selection-based BoN: per prompt pick the image with the highest OVERALL
    score among the first n seeds, then read that winning image's `dim` value;
    mean over prompts.

    overall_mat and dim_mat must share shape and column ordering (seed_index).
    The Total curve is bon_select(overall, overall, n) == bon_continuous(overall, n);
    dimension curves may be NON-monotonic in n (the overall winner can change).
    """
    if not 1 <= n <= overall_mat.shape[1]:
        raise ValueError(f"n={n} out of range [1, {overall_mat.shape[1]}]")
    sel = np.argmax(overall_mat[:, :n], axis=1)            # (n_prompts,)
    picked = dim_mat[np.arange(dim_mat.shape[0]), sel]     # companion dim values
    return float(np.mean(picked))
```

- [ ] **Step 4: 加 `_aggregate_qwen_image_bench` 与 breakdown 画图**

在 `_aggregate_spatial_geneval` 之后（约 574 行后）加：

```python
QWEN_IMAGE_BENCH_DIMS = [
    ("qwen-image-bench-quality", "Quality"),
    ("qwen-image-bench-aesthetics", "Aesthetics"),
    ("qwen-image-bench-alignment", "Alignment"),
    ("qwen-image-bench-fidelity", "Real-world Fidelity"),
    ("qwen-image-bench-creative", "Creative Generation"),
]


def _aggregate_qwen_image_bench(rows, bestofn_dir, plots_dir, csv_dir):
    """Selection-based BoN for Qwen-Image-Bench.

    Total curve = mean over prompts of max over first-n OVERALL scores. Each of
    the 5 L1 dimension curves = mean over prompts (that cover the dim) of the
    OVERALL-winner image's dimension value (bon_select). Emits 6 curve entries +
    csv/png each + a breakdown plot (5 thin dim lines + 1 thick Total).
    """
    overall_mat = build_score_matrix(rows, "qwen-image-bench")
    if overall_mat is None:
        raise ValueError("No 'qwen-image-bench' (overall) scores found; run scoring first.")
    n_max = overall_mat.shape[1]

    total_curve = aggregate_curve(overall_mat, kind="continuous")
    out = {
        "qwen-image-bench": {
            "kind": "continuous",
            "n_max": n_max,
            "num_prompts": overall_mat.shape[0],
            "curve": total_curve,
            "ceiling_lift": total_curve[n_max] - total_curve[1],
            "aggregation": "Total = mean over prompts of max over first-n overall scores",
        }
    }
    plot_curve(total_curve, "qwen-image-bench", "continuous", None,
               os.path.join(plots_dir, "qwen-image-bench_curve_log.png"))
    write_curve_csv(total_curve, os.path.join(csv_dir, "qwen-image-bench_curve.csv"))

    dim_curves = {}
    for key, label in QWEN_IMAGE_BENCH_DIMS:
        sub = [r for r in rows if key in (r.get("scores") or {})]
        if not sub:
            continue
        o_sub = build_score_matrix(sub, "qwen-image-bench")
        d_sub = build_score_matrix(sub, key)
        curve = {n: bon_select(o_sub, d_sub, n) for n in range(1, d_sub.shape[1] + 1)}
        dim_curves[key] = curve
        out[key] = {
            "kind": "continuous",
            "n_max": d_sub.shape[1],
            "num_prompts": d_sub.shape[0],
            "curve": curve,
            "ceiling_lift": curve[d_sub.shape[1]] - curve[1],
            "aggregation": f"{label}: overall-winner companion score (selection-based BoN)",
        }
        write_curve_csv(curve, os.path.join(csv_dir, f"{key}_curve.csv"))
        plot_curve(curve, key, "continuous", None,
                   os.path.join(plots_dir, f"{key}_curve_log.png"))

    _plot_qwen_image_bench_breakdown(
        total_curve, dim_curves,
        os.path.join(plots_dir, "qwen-image-bench_breakdown_curve_log.png"))
    return out


def _plot_qwen_image_bench_breakdown(total_curve, dim_curves, out_path):
    ns = sorted(total_curve.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("tab10")
    for i, (key, label) in enumerate(QWEN_IMAGE_BENCH_DIMS):
        if key not in dim_curves:
            continue
        ys = [dim_curves[key][n] for n in ns]
        ax.plot(ns, ys, marker="o", markersize=2.5, linewidth=1.2,
                color=cmap(i), label=label, alpha=0.85)
    ax.plot(ns, [total_curve[n] for n in ns], marker="o", markersize=4,
            linewidth=2.4, color="black", label="Total (overall)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (samples per prompt)")
    ax.set_ylabel("Best-of-N score (0-100)")
    ax.set_title("Qwen-Image-Bench BoN: per-dimension + Total")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 5: 在 `main` 里加路由**

在 `main` 的 `if any(k.startswith("spatial-geneval") ...)` 块之后加：

```python
    if any(k.startswith("spatial-geneval") for k in metrics):
        out.update(_aggregate_spatial_geneval(rows, bestofn_dir, plots_dir, csv_dir))
    if "qwen-image-bench" in metrics:
        out.update(_aggregate_qwen_image_bench(rows, bestofn_dir, plots_dir, csv_dir))
```

并在 `for metric in metrics:` 通用循环开头，`spatial-geneval` 跳过那一行旁边加 `qwen-image-bench*` 跳过（避免落入通用 per-key max）：

```python
    for metric in metrics:
        if metric.startswith("spatial-geneval") or metric == "_spatial_geneval_correct":
            continue
        if metric == "qwen-image-bench" or metric.startswith("qwen-image-bench-"):
            continue
```

- [ ] **Step 6: 跑测试确认通过**

Run: `python -m pytest evaluation/metrics/test_qwen_image_bench_agg.py -v`
Expected: PASS（3 个测试全过）

- [ ] **Step 7: 提交**

```bash
git add evaluation/metrics/aggregate-bestofn.py evaluation/metrics/test_qwen_image_bench_agg.py
git commit -m "feat(qwen-image-bench): selection-based BoN aggregation (Total + 5 L1 dims)"
```

---

## Task 7: run-bestofn.sh 接线

**Files:**
- Modify: `evaluation/run-bestofn.sh`（dataset 帮助文本与 case 约 42/96 行；metric_env 约 121 行；Stage2 多卡 case 约 174–177 行；判分模型 env）

- [ ] **Step 1: dataset case 加分支**

在 `case "${dataset}" in` 的 `anytext-en|anytext-zh) ...` 之后加：

```bash
    anytext-en|anytext-zh) metric_list=(anytext-ocr) ;;
    # Qwen-Image-Bench: 1000 English prompts (prompt_en). Scored by the 27B
    # Q-Judger -> per-image overall(Total) + 5 L1 dims; selection-based BoN.
    qwen-image-bench) metric_list=(qwen-image-bench) ;;
```

并把 `dataset=${3:?...}` 帮助串里的可选项补上 `qwen-image-bench`（在 `anytext-zh` 后追加 `, qwen-image-bench`）。

- [ ] **Step 2: metric_env 加映射**

在 `declare -A metric_env=( ... )` 里，`[anytext-ocr]=dpg-bench` 之后加：

```bash
    [anytext-ocr]=dpg-bench
    # 27B judge needs ms-swift; isolate from the transformers==4.40 main env.
    [qwen-image-bench]=qwen-image-bench
```

- [ ] **Step 3: Stage2 多卡分支**

把 Stage2 的 `case "${metric}" in` 改成让 `qwen-image-bench` 也用全部 GPU：

```bash
    case "${metric}" in
        dalleval-bias-gender|dalleval-bias-attribute|qwen-image-bench) score_cuda="${gpus}" ;;
        *)                                            score_cuda="${score_gpu}" ;;
    esac
```

- [ ] **Step 4: 判分模型 env 默认值**

在 Stage2 循环之前（紧跟 vLLM 探活块之后、`for metric in ...` 之前）加一个 dataset 门控块：

```bash
# Qwen-Image-Bench judge: 27B Q-Judger via ms-swift, sharded across all --gpus
# (device_map=auto). Override QIB_JUDGE_MODEL to point at a local checkpoint.
if [[ "${dataset}" == "qwen-image-bench" ]]; then
    : "${QIB_JUDGE_MODEL:=Qwen/Qwen-Image-Bench}"
    export QIB_JUDGE_MODEL
    echo "Qwen-Image-Bench judge model: ${QIB_JUDGE_MODEL}"
fi
```

- [ ] **Step 5: 语法检查**

Run: `bash -n evaluation/run-bestofn.sh && echo "bash syntax OK"`
Expected: `bash syntax OK`

- [ ] **Step 6: 提交**

```bash
git add evaluation/run-bestofn.sh
git commit -m "feat(qwen-image-bench): wire dataset/metric/env/multi-GPU into run-bestofn.sh"
```

---

## Task 8: 端到端冒烟测试（4×3090 机器，env=qwen-image-bench）

此任务在目标 GPU 机器上跑，验证 ms-swift 加载/分片、判分、缓存续跑、聚合贯通。需要已安装 conda env `qwen-image-bench`（ms-swift≥4.0 + judge 依赖）。

- [ ] **Step 1: 准备 5 条 prompt 的迷你集**

Run:
```bash
mkdir -p /tmp/qib_smoke
head -5 dataset/qwen-image-bench/prompts.jsonl > dataset/qwen-image-bench/prompts.smoke.jsonl
```
临时把 loader 指到 smoke 文件：在 `_DATASET_LOADERS` 里临时改
`"qwen-image-bench": ("prompts.smoke.jsonl", _load_jsonl)`（冒烟后改回）。
或更简单：直接用完整集但 `--n_max 2` 并在生成后 `head` 截断 evaluation_results.jsonl —— 推荐用 smoke 文件法，干净。

- [ ] **Step 2: 三段式真跑（N=2）**

Run:
```bash
bash evaluation/run-bestofn.sh "0,1,2,3" base-sd3 qwen-image-bench 2
```
Expected:
- Stage1 生成 5×2=10 张图，写 `evaluation_results.jsonl`。
- Stage2 `[qwen-image-bench] 10 images to score; model=Qwen/Qwen-Image-Bench`，27B 跨 4 卡加载成功，逐图判分；生成 `qwen_image_bench_judge_outputs.jsonl`（10 行，每行 `raw_by_dim`/`parsed_by_dim` 齐全）。
- Stage3 写 `bestofn/curves.json`，含 `qwen-image-bench`(Total) 及出现过的 `qwen-image-bench-<dim>` 键；`bestofn/plots/qwen-image-bench_breakdown_curve_log.png` 存在。

- [ ] **Step 3: 校验 Total 单调 + 6 键齐整**

Run:
```bash
python - <<'PY'
import json, glob
d = glob.glob("**/bestofn-eval/**/qwen-image-bench/bestofn/curves.json", recursive=True)
c = json.load(open(sorted(d)[-1]))
tot = c["qwen-image-bench"]["curve"]
ns = sorted(int(k) for k in tot)
vals = [tot[str(n)] for n in ns]
assert all(vals[i] <= vals[i+1] + 1e-9 for i in range(len(vals)-1)), "Total must be monotonic"
print("Total curve:", vals, "OK; dim keys:", [k for k in c if k.startswith("qwen-image-bench-")])
PY
```
Expected: Total 单调不降；打印出现的维度键。

- [ ] **Step 4: 校验断点续跑（第二次跑 0 次 27B 调用）**

Run（重复 Stage2 评分）:
```bash
OUT=$(ls -d */bestofn-eval/*/base-sd3/qwen-image-bench 2>/dev/null | head -1 || true)
# 用实际 output_dir 替换 $OUT；env 切到 qwen-image-bench
python evaluation/metrics/score-images.py --output_dir "<OUTPUT_DIR>" --metrics qwen-image-bench
```
Expected: 打印 `0/10 rows todo`（所有行已有 `qwen-image-bench` 分），判分器 `return`，**不构造 27B 引擎**、不新增 raw 行。
（额外验证缓存路径：手工删掉 `evaluation_results.jsonl` 里某行的 `qwen-image-bench*` 分数后重跑 —— 应命中 `qwen_image_bench_judge_outputs.jsonl` 缓存、`raw cache hits available` 非 0、不重跑该图的 27B。）

- [ ] **Step 5: 清理冒烟产物，提交（若改过 loader 记得改回）**

```bash
rm -f dataset/qwen-image-bench/prompts.smoke.jsonl
git checkout evaluation/metrics/generate-images-bestofn.py 2>/dev/null || true   # 若临时改过 loader
git status   # 确认无遗留临时改动
```
（冒烟测试本身不产生需提交的源码改动；若一切正常则本任务无提交。）

---

## Self-Review

**Spec coverage（逐节核对）：**
- 语言 prompt_en/dims_en → Task 1（prepare 抽 prompt_en/dims_en）。✓
- 本地文件、无下载/无 join → Task 1（直接读本地 jsonl，dims_en 自带）。✓
- 4×3090 device_map、batch=1、无单 GPU 路径 → Task 4（引擎封装 `qwen_image_bench_engine`）+ Task 7 Step3（全 GPU）。✓
- 不修改未跟踪上游 QIB（worktree 隔离）→ Task 4 改为树内封装；只读复用 `checklists`/`score_utils`（软链/真实目录）。✓
- BoN 选择式语义（overall argmax 选图、带出维度、Total 单调/维度可非单调）→ Task 6（`bon_select` + 测试）。✓
- per-image overall + 5 L1 维度，仅适用维度才写 → Task 3（`scores_from_raw`）+ Task 5。✓
- 原始判断单独文件 + 兼作缓存 → Task 5（`qwen_image_bench_judge_outputs.jsonl`）。✓
- 断点续跑（row 级 + 判断级缓存）→ Task 5 + Task 8 Step4。✓
- run-bestofn.sh 四处接线 → Task 7。✓

**Placeholder scan：** 无 TBD/TODO 占位；每个改代码步骤含完整代码或精确命令。Task 8 是运行时验证（需 GPU），步骤为精确命令而非占位。✓

**Type/名称一致性：**
- `build_tasks` / `scores_from_raw` / `load_and_resize_image` / `DIM_KEY`（Task 3）与 Task 5 调用一致。✓
- `DIM_KEY` 后缀（quality/aesthetics/alignment/fidelity/creative）与 Task 6 `QWEN_IMAGE_BENCH_DIMS` 键、score-key 前缀 `qwen-image-bench-<suffix>` 一致。✓
- `bon_select(overall_mat, dim_mat, n)`（Task 6 定义）与测试、`_aggregate_qwen_image_bench` 调用签名一致。✓
- raw 文件名 `qwen_image_bench_judge_outputs.jsonl` 在 Task 5/Task 8 一致。✓
- 续跑键：`_has_metric_score` 默认 `("qwen-image-bench",)`（未加 `METRIC_OUTPUT_KEYS` 条目，刻意——overall 必写，足以判定），与 Task 5 行为一致。✓

**已知风险（执行时留意）：**
- ms-swift `TransformersEngine` 的 `device_map` kwarg：若不支持，Task 4 的 `except TypeError` 回退到默认放置；仍不能 4 卡分片时，需按所装 ms-swift 版本改用其多卡参数（Task 8 Step2 会暴露）。
- overall 解析失败写入 `None` 会让 Task 6 的 `build_score_matrix` 报 NaN 错（清晰提示重跑该指标）；确定性解码下应极少发生。
