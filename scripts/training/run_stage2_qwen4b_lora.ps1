$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
& "$root\scripts\training\run_sft_train.ps1" -Stage sft_stage2
