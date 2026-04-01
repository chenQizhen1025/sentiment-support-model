param(
    [ValidateSet("sft_stage1", "sft_stage2")]
    [string]$Stage = "sft_stage1",
    [string]$PythonPath = "D:\Anaconda\envs\DL\python.exe",
    [string]$ConfigPath = "D:\Sentiment-SUPPORT\configs\sft_qwen4b_lora.yaml",
    [string]$GpuIds = "0",
    [int]$NumProcesses = 1,
    [switch]$UseAccelerate
)

$root = "D:\Sentiment-SUPPORT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if ($GpuIds) {
    $env:CUDA_VISIBLE_DEVICES = $GpuIds
}

$pythonDir = Split-Path $PythonPath -Parent
$acceleratePath = Join-Path $pythonDir "Scripts\accelerate.exe"
$trainScript = Join-Path $root "scripts\training\train_qwen_lora_sft.py"

Write-Host "Python:" $PythonPath
Write-Host "Stage:" $Stage
Write-Host "CUDA_VISIBLE_DEVICES:" $env:CUDA_VISIBLE_DEVICES
Write-Host "NumProcesses:" $NumProcesses

if ($UseAccelerate -or $NumProcesses -gt 1) {
    & $acceleratePath launch --num_processes $NumProcesses --mixed_precision fp16 $trainScript --config $ConfigPath --stage $Stage
}
else {
    & $PythonPath $trainScript --config $ConfigPath --stage $Stage
}
