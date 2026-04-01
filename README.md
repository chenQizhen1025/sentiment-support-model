# Sentiment-SUPPORT

This project focuses on data preprocessing and SFT data construction for an empathetic support model.

## Current scope
- Use SoulChatCorpus as the main multi-turn empathy dataset
- Use the current project's single-turn and multi-turn mental-health datasets as domain-anchor data
- Complete data preprocessing first
- Build SFT-ready train/dev/test datasets later

## Current dataset roots
- `data/SoulChatCorpus/`
- `data/quality_data/`
- `data/processed/train_dedup.jsonl`
- `data/processed/val_dedup.jsonl`

## Workflow
1. Inventory datasets
2. Standardize schema
3. Basic cleaning
4. Quality filtering
5. Cross-source deduplication
6. Build Stage 1 / Stage 2 SFT datasets
