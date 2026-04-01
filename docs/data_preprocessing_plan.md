# Data Preprocessing Plan

## Goal
Prepare high-quality SFT-ready data for an empathetic support model based on Qwen-4B with LoRA.

## Data sources
- SoulChat processed multi-turn data
- Current-project single-turn and multi-turn mental-health data

## Main stages
1. Inventory all source datasets
2. Standardize all datasets into a unified messages-based schema
3. Run basic structural cleaning and anomaly checks
4. Run rule-based and optional LLM-assisted quality filtering
5. Run internal and cross-source deduplication
6. Build Stage 1 and Stage 2 SFT train/dev/test splits

## Deliverables
- Inventory report
- Standardized datasets
- Filtered datasets
- Deduplicated datasets
- Stage 1 / Stage 2 SFT datasets
- Independent eval dataset design document
