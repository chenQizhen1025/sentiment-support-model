param(
    [string]$PythonPath = "python"
)

$python = if ($PythonPath -eq "python") { "python" } else { (Resolve-Path $PythonPath).Path }
& $python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device_count=', torch.cuda.device_count()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
