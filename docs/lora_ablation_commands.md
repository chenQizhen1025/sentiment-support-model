# LoRA Ablation Commands

## QV Baseline

### Stage 1
powershell -ExecutionPolicy Bypass -File "D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1" -Stage sft_stage1 -ConfigPath "D:\Sentiment-SUPPORT\configs\sft_qwen4b_lora_qv.yaml"

### Stage 2
powershell -ExecutionPolicy Bypass -File "D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1" -Stage sft_stage2 -ConfigPath "D:\Sentiment-SUPPORT\configs\sft_qwen4b_lora_qv.yaml"

## Full LoRA (QKVO + MLP)

### Stage 1
powershell -ExecutionPolicy Bypass -File "D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1" -Stage sft_stage1 -ConfigPath "D:\Sentiment-SUPPORT\configs\sft_qwen4b_lora_full.yaml"

### Stage 2
powershell -ExecutionPolicy Bypass -File "D:\Sentiment-SUPPORT\scripts\training\run_sft_train.ps1" -Stage sft_stage2 -ConfigPath "D:\Sentiment-SUPPORT\configs\sft_qwen4b_lora_full.yaml"

## Current 6GB GPU Suggestion

Run Stage 1 first. If you hit CUDA OOM, reduce `data.max_length` and/or `lora.r` in the config before running Stage 2.
