param(
    [string]$PythonPath = "python"
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = if ($PythonPath -eq "python") { "python" } else { (Resolve-Path $PythonPath).Path }

& $python "$root\scripts\training\download_qwen_model.py"
