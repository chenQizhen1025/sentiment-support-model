# SFT 训练框架说明

## 1. 当前框架做了什么

目前当前项目里已经搭好了一套面向心理支持对话任务的 SFT 训练框架，目标模型为 `Qwen 4B` 系列，训练方法为 `LoRA` 微调。

这套框架已经包含：

- 数据预处理入口
- `stage1 / stage2` 训练数据构建脚本
- `Qwen-4B + LoRA` 训练脚本
- 训练配置文件
- 本地模型下载脚本
- 自动评测脚本与人工评测规则

## 2. 训练框架的核心文件

### 2.1 数据预处理与数据构建

- `scripts/data_preprocessing/inspect_datasets.py`
  用于检查数据集规模、轮次数、空内容等基础统计。

- `scripts/data_preprocessing/standardize_sources.py`
  用于把不同来源的数据统一成内部标准格式。

- `scripts/data_preprocessing/basic_filter.py`
  用于做基础结构过滤，剔除空内容、无 assistant、过短回复等明显异常样本。

- `scripts/data_preprocessing/build_sft_datasets.py`
  用于构建 `sft_stage1` 和 `sft_stage2` 的 `train/dev/test` 数据集。

### 2.2 训练相关文件

- `scripts/training/train_qwen_lora_sft.py`
  主训练脚本，负责加载模型、读取数据、套用 LoRA 配置并执行 SFT。

- `configs/sft_qwen4b_lora.yaml`
  主训练配置文件，包含模型路径、数据路径、训练超参数和 LoRA 参数。

- `scripts/training/run_stage1_qwen4b_lora.ps1`
  启动 `stage1` 训练。

- `scripts/training/run_stage2_qwen4b_lora.ps1`
  启动 `stage2` 训练。

### 2.3 模型准备相关文件

- `scripts/training/download_qwen_model.py`
  用于下载 Qwen 模型到本地目录。

- `scripts/training/run_download_qwen_model.ps1`
  模型下载启动入口。

### 2.4 评测相关文件

- `scripts/evaluation/auto_eval.py`
  自动评测脚本，参考 LMAPP 的 BLEU / ROUGE 思路实现。

- `data/processed/eval/auto_eval_samples.jsonl`
  自动评测样例数据。

- `data/processed/eval/manual_eval_rubric.json`
  人工评测维度定义。

## 3. 数据分阶段设计

### Stage 1

`stage1` 主要使用当前项目中的高质量单轮和高质量多轮数据，目的是先把模型调成更符合大学生心理支持场景的风格。

输出目录：

- `data/processed/sft_stage1/`

### Stage 2

`stage2` 在 `stage1` 基础上继续加入 `SoulChat` 多轮共情数据，目的是增强多轮承接和连续对话能力。

输出目录：

- `data/processed/sft_stage2/`

## 4. 当前默认模型

当前配置已经切到本地模型路径：

- `models/Qwen/Qwen3-4B-Instruct-2507`

对应来源仓库：

- `Qwen/Qwen3-4B-Instruct-2507`

## 5. 当前状态

现在这套框架已经具备：

- 预处理数据
- 构建训练集
- 启动训练
- 准备模型
- 做自动评测

也就是说，工程骨架已经完整，后续可以直接围绕训练实验继续推进。
