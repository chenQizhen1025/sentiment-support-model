# Work Log

## 2026-04-01
- Initialized the Sentiment-SUPPORT project workspace.
- Confirmed the local presence of SoulChat processed data and current-project quality_data files.
- Created project folders for configs, docs, scripts, reports, interim data, and processed outputs.
- Planned the preprocessing workflow around: inventory, standardization, basic cleaning, quality filtering, deduplication, and staged SFT dataset construction.
- Added the SFT dataset builder for stage1 and stage2 outputs.
- Added the Qwen-4B LoRA SFT training framework, configs, dependency list, and PowerShell launch scripts.
- Added an SFT framework work document describing the new pipeline and how to continue.
- Ran dataset inspection and generated dataset_inventory_report.json with real source statistics.
- Ran source standardization and basic filtering for SoulChat and current-project datasets.
- Built stage1 and stage2 SFT datasets and generated the corresponding dataset reports.
- Added sft_work_report.md summarizing preprocessing outputs and the Qwen-4B LoRA training framework.
- Updated the SFT dependency list and local training config to better fit the current Windows + RTX 3060 environment.
- Added the evaluation framework based on the LMAPP BLEU/ROUGE evaluation idea and a manual scoring rubric.
- Switched the default training target to the valid Qwen 4B repo Qwen/Qwen3-4B-Instruct-2507 and added a dedicated model download script.
- Downloaded Qwen/Qwen3-4B-Instruct-2507 successfully through ModelScope and updated the config to the real local model path.
- Recorded the current environment status: model ready, but PyTorch is CPU-only so GPU training is not yet available.
- Switched all training and preprocessing launcher scripts to the CUDA-enabled DL environment.
- Confirmed that the DL environment can access the RTX 3060 with torch 2.8.0+cu128.
- Generalized the SFT launchers to support switching Python envs, GPU IDs, and single-/multi-GPU execution for future hardware changes.
