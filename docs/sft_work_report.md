# SFT 工作报告

## 1. 本轮已经完成的工作

本轮已经把当前项目从“只有预处理规划”推进到了“具备可训练能力”的状态，主要完成了三部分：

- 数据预处理链路落地并实际运行
- `stage1 / stage2` SFT 数据集生成
- `Qwen-4B + LoRA` 训练框架搭建

## 2. 数据预处理实际执行结果

### 2.1 数据检查

已生成：

- `data/reports/dataset_inventory_report.json`

核心统计结果：

- SoulChat 训练集：`232,517` 条对话
- SoulChat 验证集：`25,836` 条对话
- 当前项目单轮数据：`18,000` 条
- 当前项目多轮处理版：`19,200` 条
- 当前项目多轮原始版：`18,000` 条

### 2.2 标准化

已生成：

- `data/interim/standardized/current_single_standardized.jsonl`
- `data/interim/standardized/current_multi_processed_standardized.jsonl`
- `data/interim/standardized/current_multi_raw_standardized.jsonl`
- `data/interim/standardized/soulchat_train_standardized.jsonl`
- `data/interim/standardized/soulchat_val_standardized.jsonl`

### 2.3 基础过滤

已生成：

- `data/reports/basic_filter_report.json`

过滤后保留情况：

- `current_single`：`17,994`
- `current_multi_processed`：`19,200`
- `current_multi_raw`：`17,999`
- `soulchat_train`：`232,059`
- `soulchat_val`：`25,779`

## 3. 已生成的训练数据

### 3.1 Stage 1

已生成：

- `data/processed/sft_stage1/train.jsonl`
- `data/processed/sft_stage1/dev.jsonl`
- `data/processed/sft_stage1/test.jsonl`

样本规模：

- `train`: `102,568`
- `dev`: `5,697`
- `test`: `5,697`

### 3.2 Stage 2

已生成：

- `data/processed/sft_stage2/train.jsonl`
- `data/processed/sft_stage2/dev.jsonl`
- `data/processed/sft_stage2/test.jsonl`

样本规模：

- `train`: `1,545,745`
- `dev`: `85,872`
- `test`: `85,872`

## 4. 已搭建的训练框架

已经新增并可用的训练相关文件包括：

- `scripts/training/train_qwen_lora_sft.py`
- `configs/sft_qwen4b_lora.yaml`
- `scripts/training/run_stage1_qwen4b_lora.ps1`
- `scripts/training/run_stage2_qwen4b_lora.ps1`
- `scripts/training/download_qwen_model.py`
- `scripts/training/run_download_qwen_model.ps1`

## 5. 模型准备情况

Qwen 4B 模型已经通过 ModelScope 下载到本地：

- `models/Qwen/Qwen3-4B-Instruct-2507`

## 6. 当前环境状态

当前环境已经安装好了大部分训练和评测依赖，但当前 `torch` 是 CPU 版：

- `torch 2.8.0+cpu`

这意味着：

- 脚本可以导入和运行
- 但当前还不能直接使用 RTX 3060 做 GPU 训练

## 7. 当前结论

截至目前，这个项目已经具备：

- 预处理多源心理支持数据
- 生成 SFT 训练集
- 本地准备 Qwen 4B 模型
- 启动 LoRA 训练框架
- 搭建自动评测与人工评测基础

也就是说，当前已经进入“可以正式做训练实验”的前夜，剩下的关键问题主要是补齐 CUDA 版 PyTorch。

## 8. GPU ????

????????????????? GPU ?????????????????????????????

- `scripts/training/run_sft_train.ps1`
- `configs/runtime_profiles.json`
- `docs/gpu_portability_guide.md`
