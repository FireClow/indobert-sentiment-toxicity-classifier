# Install PyTorch CUDA from local wheel (run from project folder)
$wheel = "C:\Users\kenji\OneDrive - Bina Nusantara\Documents\Kuliah\Aplikasi\Aplikasipytorch\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"

if (-not (Test-Path $wheel)) {
    Write-Error "File not found: $wheel"
    Write-Host "Pastikan file .whl ada di folder Aplikasi\Aplikasipytorch (bukan hanya di folder Aplikasi)."
    exit 1
}

pip install $wheel
python -c "import torch; print('Version:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
