# Start training on GPU (run in Windows Terminal or PowerShell)
$ErrorActionPreference = "Stop"
$env:SSL_CERT_FILE = $null
$env:REQUESTS_CA_BUNDLE = $null
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"

Set-Location $PSScriptRoot

Write-Host "Checking CUDA..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting training..."
python run_train.py
exit $LASTEXITCODE
