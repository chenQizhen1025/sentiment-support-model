param(
    [ValidateSet("sft_stage1", "sft_stage2")]
    [string]$Stage = "sft_stage1",
    [string]$PythonPath = "python",
    [string]$ConfigPath = "",
    [string]$GpuIds = "0",
    [int]$NumProcesses = 1,
    [switch]$UseAccelerate
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if ($GpuIds) {
    $env:CUDA_VISIBLE_DEVICES = $GpuIds
}

$pythonResolved = if ($PythonPath -eq "python") { "python" } else { (Resolve-Path $PythonPath).Path }
$configResolved = if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    Join-Path $root "configs\sft_qwen4b_lora.yaml"
} else {
    if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { (Join-Path $root $ConfigPath) }
}
$pythonDir = if ($pythonResolved -eq "python") { Split-Path (Get-Command python).Source -Parent } else { Split-Path $pythonResolved -Parent }
$acceleratePath = Join-Path $pythonDir "Scripts\accelerate.exe"
$trainScript = Join-Path $root "scripts\training\train_qwen_lora_sft.py"

Write-Host "Python:" $pythonResolved
Write-Host "Stage:" $Stage
Write-Host "Config:" $configResolved
Write-Host "CUDA_VISIBLE_DEVICES:" $env:CUDA_VISIBLE_DEVICES
Write-Host "NumProcesses:" $NumProcesses

if ($UseAccelerate -or $NumProcesses -gt 1) {
    & $acceleratePath launch --num_processes $NumProcesses --mixed_precision fp16 $trainScript --config $configResolved --stage $Stage
}
else {
    & $pythonResolved $trainScript --config $configResolved --stage $Stage
}
