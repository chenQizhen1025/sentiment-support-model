param(
    [string]$PythonPath = "python"
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = if ($PythonPath -eq "python") { "python" } else { (Resolve-Path $PythonPath).Path }

& $python "$root\scripts\data_preprocessing\inspect_datasets.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "$root\scripts\data_preprocessing\standardize_sources.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "$root\scripts\data_preprocessing\basic_filter.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "$root\scripts\data_preprocessing\build_sft_datasets.py"
exit $LASTEXITCODE
