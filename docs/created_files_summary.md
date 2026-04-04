# 已创建文件工作说明

本文档用于说明我在当前项目根目录下新建的目录、文档和脚本分别完成了哪些工作，方便后续继续推进数据预处理与 SFT 数据构建。

## 1. 本次新增的目录结构

本次主要完成了项目初始化目录搭建，目的是把后续的数据预处理工作、结果文件和记录文档分层管理。

已创建的主要目录包括：

- `configs/`
- `docs/`
- `scripts/`
- `scripts/data_preprocessing/`
- `data/raw/`
- `data/raw/current_project/`
- `data/raw/soulchat/`
- `data/interim/`
- `data/interim/standardized/`
- `data/interim/filtered/`
- `data/interim/deduped/`
- `data/processed/sft_stage1/`
- `data/processed/sft_stage2/`
- `data/processed/eval/`
- `data/reports/`
- `experiments/`

这些目录的作用分别是：

- `data/raw/`：存放原始数据，不直接改动。
- `data/interim/`：存放标准化、过滤、去重后的中间结果。
- `data/processed/`：存放最终可用于训练和评测的数据。
- `data/reports/`：存放统计报告、过滤报告、去重报告。
- `scripts/data_preprocessing/`：存放数据预处理脚本。
- `docs/`：存放项目说明、计划、工作记录和总结。
- `configs/`：存放数据注册表、项目配置。
- `experiments/`：为后续训练实验记录预留。

## 2. 已创建文档说明

### 2.1 `README.md`

该文件完成了项目级别的基础说明，主要明确了：

- 当前项目的目标是做“共情支持回复模型”的数据预处理和 SFT 数据构建。
- 当前阶段暂时只做 `SFT`，不做 `DPO`。
- 主要数据源包括：
  - `SoulChat` 多轮共情对话数据
  - 当前项目中的高质量单轮和多轮心理健康数据
- 整体处理流程包括：
  - 数据盘点
  - 格式统一
  - 基础清洗
  - 质量过滤
  - 去重
  - 构建 Stage 1 / Stage 2 SFT 数据

它的作用是让这个目录一打开就能知道项目当前在做什么，以及后续工作主线是什么。

### 2.2 `docs/work_log.md`

该文件是工作记录文档，主要用于按日期记录当前阶段做了什么。

目前记录的内容包括：

- 初始化项目工作目录
- 确认本地已有的数据资源
- 创建数据处理目录结构
- 确定当前阶段以“数据预处理”为主
- 明确后续会围绕标准化、过滤、去重和 SFT 数据构建继续推进

它的作用是保留过程记录，后续你写论文或整理项目进度时会比较方便。

### 2.3 `docs/data_preprocessing_plan.md`

该文件是数据预处理阶段的计划文档，主要梳理了后续要完成的工作流程。

里面已经明确了：

- 项目目标是为 `Qwen-4B + LoRA` 准备高质量 SFT 数据
- 主要数据来源是什么
- 数据预处理要分哪些阶段
- 每个阶段的产出文件是什么

它更像是一份实施说明书，后续做脚本和跑数据时可以对照执行。

### 2.4 `docs/data_inventory.md`

该文件是当前已识别数据集的清单文档。

目前已经整理进去的数据包括：

- `SoulChat` 的处理后训练集和验证集
- 当前项目中的单轮高质量数据
- 当前项目中的多轮处理版数据
- 当前项目中的多轮原始版数据

每条数据都标注了用途，例如：

- 主多轮共情数据
- 领域锚点单轮数据
- 领域锚点多轮数据

这个文件的作用是帮助你快速知道“当前有哪些数据可用、它们各自扮演什么角色”。

## 3. 已创建配置文件说明

### 3.1 `configs/dataset_registry.yaml`

该文件是数据注册表，用于统一记录项目中要使用的数据路径和角色。

目前已经登记的信息包括：

- 项目根目录
- 当前项目范围：`sft_only`
- 目标模型：`qwen_4b_family`
- 语言：`zh`
- 统一使用的 system prompt
- 各个数据集的路径和用途

已登记数据源包括：

- `soulchat_train`
- `soulchat_val`
- `current_single`
- `current_multi_processed`
- `current_multi_raw`

这个文件的作用是为后续脚本提供统一入口，避免脚本里到处分散写死路径。

## 4. 已创建脚本说明

### 4.1 `scripts/data_preprocessing/inspect_datasets.py`

这是数据检查脚本，用于对当前选定的数据源做初步统计和结构检查。

它目前完成的工作包括：

- 读取 `SoulChat` 的训练集和验证集
- 读取当前项目的单轮和多轮数据
- 统计不同数据集的样本数量
- 对多轮数据统计：
  - 平均轮次
  - 最小轮次
  - 最大轮次
  - 首轮角色分布
  - 末轮角色分布
  - 空内容数量
- 对单轮数据统计：
  - 空问题数量
  - 空回答数量
  - 文本长度分布

脚本会把结果输出到：

- `data/reports/dataset_inventory_report.json`

这个脚本的作用是：在真正开始清洗和标准化之前，先把数据情况摸清楚。

### 4.2 `scripts/data_preprocessing/standardize_sources.py`

这是数据标准化脚本，用于把不同来源的数据统一转换成内部标准格式。

目前它已经完成的设计是：

- 为所有样本补统一的 system prompt
- 将单轮数据转成带 `messages` 的格式
- 将多轮数据统一成包含以下字段的格式：
  - `sample_id`
  - `source`
  - `topic`
  - `messages`

它会生成标准化后的文件到：

- `data/interim/standardized/current_single_standardized.jsonl`
- `data/interim/standardized/current_multi_processed_standardized.jsonl`
- `data/interim/standardized/current_multi_raw_standardized.jsonl`
- `data/interim/standardized/soulchat_train_standardized.jsonl`
- `data/interim/standardized/soulchat_val_standardized.jsonl`

这个脚本的作用是：把不同来源的数据先“整理成同一种内部表示”，方便后续统一过滤和去重。

### 4.3 `scripts/data_preprocessing/basic_filter.py`

这是基础过滤脚本，用于对标准化后的数据做结构性清洗。

它当前已经实现的过滤规则包括：

- 样本至少要有 3 条消息
- 第一条必须是 `system`
- 第一条对话消息必须是 `user`
- 不能有空内容
- 单条消息长度不能超过 4000 字符
- 样本中必须包含 assistant 回复
- assistant 回复长度不能少于 6 个字符

输出路径为：

- `data/interim/filtered/*.jsonl`
- `data/reports/basic_filter_report.json`

这个脚本的作用是先删掉明显不合格、结构异常或质量过低的样本，为后续更细的质量过滤打底。

## 5. 目前这些文件已经完成到什么程度

截至目前，已经完成的是：

- 项目目录初始化
- 数据处理流程文档化
- 数据清单登记
- 数据注册表建立
- 三个基础预处理脚本的搭建

也就是说，当前已经把“数据预处理阶段的基础工程框架”搭起来了。

但还没有正式完成的工作包括：

- 实际跑完整的数据标准化输出
- 实际跑完整的基础过滤输出
- 做跨数据源去重
- 构建最终的 Stage 1 / Stage 2 SFT 数据
- 构建你自己的评测集

## 6. 当前阶段的结论

本次新建文件的核心贡献，不是直接生成最终训练集，而是先完成了以下基础工作：

- 把当前仓库组织成一个可持续推进的数据工程目录
- 明确了项目当前只做 `SFT` 的工作边界
- 明确了 `SoulChat + 当前项目心理健康数据` 的数据角色分工
- 建立了数据预处理的文档、配置和脚本骨架
- 为后续实际执行“标准化、清洗、过滤、去重、构建训练集”做好了准备

## 7. 后续建议

接下来最适合继续推进的顺序是：

1. 运行 `inspect_datasets.py`，生成真实统计报告
2. 运行 `standardize_sources.py`，生成标准化数据
3. 运行 `basic_filter.py`，生成基础过滤结果
4. 在此基础上继续开发质量过滤与去重脚本
5. 最后再构建 `sft_stage1` 和 `sft_stage2` 数据集

