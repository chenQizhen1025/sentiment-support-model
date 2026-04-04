# 2026-04-01 工作总记录

本文档用于汇总今天在当前项目根目录下实际完成的工作，方便后续继续研究代码、复现实验、排查环境和理解当前项目状态。

## 1. 今日工作目标

今天的主要工作目标有四个：

1. 在当前项目根目录下搭建项目基础结构。
2. 把数据预处理链路从“规划”推进到“实际可运行”。
3. 搭建 `Qwen 4B + LoRA` 的 SFT 训练框架。
4. 让训练框架能够使用现有 `DL` 环境中的 GPU，并支持未来迁移到其他 GPU 环境。

## 2. 今日完成的主要工作

### 2.1 项目目录初始化

今天已经在当前项目根目录下补齐并使用了以下目录：

- `configs/`
- `docs/`
- `scripts/`
- `scripts/data_preprocessing/`
- `scripts/training/`
- `scripts/evaluation/`
- `data/raw/`
- `data/interim/`
- `data/interim/standardized/`
- `data/interim/filtered/`
- `data/interim/deduped/`
- `data/processed/sft_stage1/`
- `data/processed/sft_stage2/`
- `data/processed/eval/`
- `data/reports/`
- `experiments/`
- `models/`

这些目录现在已经形成了一个完整的工程结构，后续继续做预处理、训练和评测都可以直接往下接。

### 2.2 文档与说明文件

今天新增或更新了以下文档：

- `docs/created_files_summary.md`
- `docs/sft_training_framework.md`
- `docs/sft_work_report.md`
- `docs/evaluation_framework.md`
- `docs/gpu_portability_guide.md`
- `docs/work_log.md`

这些文档分别用于：

- 说明新建文件的用途
- 说明 SFT 训练框架结构
- 记录今日工作结果
- 说明评测框架与指标
- 说明如何迁移到其他 GPU 环境
- 记录工作日志

### 2.3 数据预处理脚本搭建与运行

今天已经确认并使用的预处理脚本有：

- `scripts/data_preprocessing/inspect_datasets.py`
- `scripts/data_preprocessing/standardize_sources.py`
- `scripts/data_preprocessing/basic_filter.py`
- `scripts/data_preprocessing/build_sft_datasets.py`

这些脚本完成了如下流程：

1. 数据统计与检查
2. 格式标准化
3. 基础过滤
4. 构建 Stage 1 / Stage 2 SFT 数据集

### 2.4 训练框架搭建

今天搭建并可用的训练相关文件有：

- `scripts/training/train_qwen_lora_sft.py`
- `scripts/training/run_preprocessing_pipeline.ps1`
- `scripts/training/run_sft_train.ps1`
- `scripts/training/run_stage1_qwen4b_lora.ps1`
- `scripts/training/run_stage2_qwen4b_lora.ps1`
- `scripts/training/download_qwen_model.py`
- `scripts/training/run_download_qwen_model.ps1`
- `scripts/training/run_check_torch_cuda.ps1`
- `configs/sft_qwen4b_lora.yaml`
- `configs/deepspeed_zero2.json`
- `configs/runtime_profiles.json`
- `requirements-sft.txt`

现在的训练框架已经具备：

- 数据读取
- tokenizer 与 chat template 处理
- Qwen 模型加载
- LoRA 注入
- SFT 训练
- 单卡/多卡启动入口
- Python 环境和 GPU 参数化切换

### 2.5 自动评测与人工评测框架

今天新增了以下评测相关文件：

- `scripts/evaluation/auto_eval.py`
- `data/processed/eval/auto_eval_samples.jsonl`
- `data/processed/eval/manual_eval_rubric.json`
- `docs/evaluation_framework.md`

当前评测框架参考了 LMAPP 的思路，保留了以下自动指标：

- `BLEU-1`
- `BLEU-2`
- `BLEU-3`
- `BLEU-4`
- `ROUGE-1`
- `ROUGE-2`
- `ROUGE-L`
- `total_score`

同时保留了更适合你项目的人工评测维度：

- 共情性
- 贴合度
- 自然度
- 帮助性
- 安全性
- 多轮连贯性

## 3. 今日实际运行过的流程与结果

### 3.1 数据检查结果

已运行：

- `inspect_datasets.py`

生成：

- `data/reports/dataset_inventory_report.json`

关键结果：

- SoulChat 训练集：`232,517` 条对话
- SoulChat 验证集：`25,836` 条对话
- 当前项目单轮数据：`18,000` 条
- 当前项目多轮处理版：`19,200` 条
- 当前项目多轮原始版：`18,000` 条

### 3.2 数据标准化结果

已运行：

- `standardize_sources.py`

生成：

- `data/interim/standardized/current_single_standardized.jsonl`
- `data/interim/standardized/current_multi_processed_standardized.jsonl`
- `data/interim/standardized/current_multi_raw_standardized.jsonl`
- `data/interim/standardized/soulchat_train_standardized.jsonl`
- `data/interim/standardized/soulchat_val_standardized.jsonl`

统一后的核心字段是：

- `sample_id`
- `source`
- `topic`
- `messages`

### 3.3 基础过滤结果

已运行：

- `basic_filter.py`

生成：

- `data/reports/basic_filter_report.json`

过滤后保留情况：

- `current_single`: `17,994`
- `current_multi_processed`: `19,200`
- `current_multi_raw`: `17,999`
- `soulchat_train`: `232,059`
- `soulchat_val`: `25,779`

### 3.4 SFT 数据集构建结果

已运行：

- `build_sft_datasets.py`

生成：

- `data/reports/sft_stage1_dataset_report.json`
- `data/reports/sft_stage2_dataset_report.json`

#### Stage 1

输出文件：

- `data/processed/sft_stage1/train.jsonl`
- `data/processed/sft_stage1/dev.jsonl`
- `data/processed/sft_stage1/test.jsonl`

样本规模：

- `train`: `102,568`
- `dev`: `5,697`
- `test`: `5,697`

#### Stage 2

输出文件：

- `data/processed/sft_stage2/train.jsonl`
- `data/processed/sft_stage2/dev.jsonl`
- `data/processed/sft_stage2/test.jsonl`

样本规模：

- `train`: `1,545,745`
- `dev`: `85,872`
- `test`: `85,872`

## 4. 模型准备情况

### 4.1 本地模型下载

最开始尝试过使用 Hugging Face 下载模型，但因为网络代理问题被阻断，后续改为使用 ModelScope。

最终已经成功下载：

- `Qwen/Qwen3-4B-Instruct-2507`

本地路径：

- `models/Qwen/Qwen3-4B-Instruct-2507`

当前训练配置已经指向本地模型目录，不再依赖在线拉取。

### 4.2 训练目标模型

当前默认目标模型是：

- `Qwen3-4B-Instruct-2507`

训练方式：

- `LoRA`
- `SFT`

## 5. GPU 训练环境处理结果

### 5.1 发生过的问题

今天中途我一度尝试在 `base` 环境里补装 CUDA 版 PyTorch，这条路没有继续使用，因为你提醒了已有的 `DL` 环境。

这个提醒是正确的，后续已经完全切回 `DL` 环境处理。

### 5.2 最终采用的训练环境

最终确认的可用 GPU 训练环境是：

- Conda 环境：`DL`
- Python 路径：通过启动参数传入，默认可使用 `python`
- Torch：`2.8.0+cu128`
- CUDA：`True`
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`

这一结果已经通过：

- `scripts/training/run_check_torch_cuda.ps1`

实际验证通过。

### 5.3 训练入口切换

所有主要启动脚本都已经切到 `DL` 环境：

- `run_preprocessing_pipeline.ps1`
- `run_sft_train.ps1`
- `run_stage1_qwen4b_lora.ps1`
- `run_stage2_qwen4b_lora.ps1`
- `run_download_qwen_model.ps1`
- `run_check_torch_cuda.ps1`

也就是说，当前默认路径已经是 GPU 可用的环境，而不是 base 环境。

## 6. 为未来切换其他 GPU 做的调整

今天不仅打通了当前这张 3060，还顺手把训练框架做成了可迁移版本。

已完成的改动包括：

- 新增通用启动脚本：`scripts/training/run_sft_train.ps1`
- 新增运行配置模板：`configs/runtime_profiles.json`
- 新增迁移说明：`docs/gpu_portability_guide.md`
- 将 GPU 编号改成运行时参数
- 将 Python 环境改成运行时参数
- 支持单卡直接运行
- 支持多卡使用 `accelerate launch`
- 在训练配置中加入 `ddp_find_unused_parameters: false`

这意味着后续你切换到其他单卡或多卡环境时，不需要重写训练代码，只需要改：

- `PythonPath`
- `GpuIds`
- `NumProcesses`
- 少量 batch / gradient accumulation / max_length 配置

## 7. 今日修复的问题

### 7.1 文档乱码问题

今天后半段发现部分文档因为 Windows 下编码链路问题出现了乱码。

已经重新修复为正常中文的文档包括：

- `docs/sft_work_report.md`
- `docs/sft_training_framework.md`
- `docs/evaluation_framework.md`
- `scripts/training/install_torch_cuda_note.md`
- `docs/gpu_portability_guide.md`
- `configs/runtime_profiles.json`

现在这些文件都已经可以正常查看。

### 7.2 训练依赖问题

今天已经补齐了 `DL` 环境中缺失的训练依赖：

- `transformers==4.45.2`
- `accelerate==0.34.2`
- `peft==0.12.0`
- `datasets==2.21.0`
- `sentencepiece`
- `jieba`
- `nltk`
- `rouge-chinese`
- `huggingface_hub`
- `tqdm`

## 8. 当前项目已经具备的能力

截至今天结束时，当前仓库已经具备：

- 多源心理支持对话数据预处理能力
- `stage1 / stage2` SFT 训练集
- 本地化 Qwen 4B 模型
- 基于 LoRA 的 SFT 训练框架
- 可用的 GPU 训练环境
- 可迁移到其他 GPU 的通用启动方式
- 自动评测和人工评测基础框架

## 9. 当前还没做的事

虽然训练框架已经搭好、GPU 环境也已可用，但今天还没有正式做：

- `stage1` 小规模试训
- `stage1` 全量正式训练
- `stage2` 正式训练
- 你自己的完整评测集构建
- 训练后结果对比与人工评审

## 10. 后续建议的推进顺序

建议你下一步按下面顺序继续：

1. 先跑 `stage1` 小规模试训
2. 看显存、loss、checkpoint 是否正常
3. 再跑 `stage1` 正式训练
4. 构建你自己的评测集
5. 用评测框架比较 base / stage1 / stage2
6. 最后再决定是否继续做 DPO

## 11. 最重要的当前结论

用一句话总结今天的工作成果：

**项目已经从“前期规划阶段”进入了“可以正式开始 LoRA 训练实验”的状态。**

也就是说，你现在已经不是在准备搭框架，而是已经具备：

- 数据
- 模型
- GPU
- 训练脚本
- 评测框架
- 迁移能力

后续的核心工作就从“搭基础设施”转向“开始训练和评估效果”。
