# GitHub Dataset Packaging

- `models/` is intentionally excluded from Git tracking.
- `data/interim/` and `data/raw/` are local preprocessing artifacts and are not pushed.
- `data/processed/sft_stage2/train.jsonl` is kept locally for training but excluded from GitHub because it exceeds GitHub-friendly single-file size limits.
- GitHub upload uses the split files below instead.

## Stage2 Train Chunks
- `train.part-000.jsonl` (1501086510 bytes)
- `train.part-001.jsonl` (660202830 bytes)

## Rebuild Command

```powershell
$output = 'data\\processed\\sft_stage2\\train.jsonl'
Get-ChildItem 'data\\processed\\sft_stage2' -Filter 'train.part-*.jsonl' | Sort-Object Name | Get-Content | Set-Content -LiteralPath $output -Encoding UTF8
```

