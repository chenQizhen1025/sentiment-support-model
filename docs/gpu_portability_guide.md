# GPU 迁移与切换说明

## 1. 当前调整目标

这次调整的目标，是把训练框架从“写死在当前本地 3060 环境”改成“后续可以切换到其他 GPU 或其他机器”的版本。

也就是说，后续你如果换到：

- 另一张本地显卡
- 远程单卡服务器
- 远程多卡服务器

都不需要重写训练脚本本身，只需要修改启动参数即可。

## 2. 当前已做的调整

已经完成的改动包括：

- 新增通用启动脚本：`scripts/training/run_sft_train.ps1`
- 保留 `stage1` 和 `stage2` 简化入口
- 把 GPU 编号改成运行时参数，而不是写死
- 把 Python 环境路径改成运行时参数，而不是写死
- 支持单卡直接启动，也支持多卡使用 `accelerate launch`
- 在训练配置中补充 `ddp_find_unused_parameters: false`，更适合 LoRA + 多卡 DDP
- 新增运行配置参考：`configs/runtime_profiles.json`

## 3. 默认用法

### 本地单卡

```powershell
D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1 -Stage sft_stage1
```

### 指定 GPU 编号

```powershell
D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1 -Stage sft_stage1 -GpuIds 1
```

### 指定其他 Python 环境

```powershell
D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1 -Stage sft_stage1 -PythonPath "E:\envs\dl\python.exe"
```

### 多卡训练

```powershell
D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1 -Stage sft_stage2 -GpuIds "0,1" -NumProcesses 2 -UseAccelerate
```

## 4. 迁移到其他 GPU / 服务器时改什么

通常只需要改这几个量：

- `PythonPath`
- `GpuIds`
- `NumProcesses`
- 视显存情况调整 `configs/sft_qwen4b_lora.yaml` 里的：
  - `per_device_train_batch_size`
  - `gradient_accumulation_steps`
  - `max_length`
  - `fp16 / bf16`

## 5. 什么时候算真正完成迁移

后续如果你把项目接到其他 GPU 环境，至少要确认这几点：

- `torch.cuda.is_available()` 为 `True`
- `run_sft_train.ps1` 能正常启动
- 模型路径可访问
- 显存能撑住当前 batch 配置

## 6. 当前结论

训练框架现在已经不再只适配当前这张 3060，而是已经具备切换到其他 GPU 环境的基础能力。后续你主要改“运行参数”和“少量训练配置”，不需要推倒重来。
