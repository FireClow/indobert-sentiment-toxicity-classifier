"""
Entry point for training. Prefer this over `python -m training.train` on Windows.

Usage (PowerShell):
  cd "C:\\Users\\kenji\\Sentiment and Toxicity Classification of Indonesian Social Media Comments Using IndoBERT"
  .\\train_gpu.ps1
"""
from __future__ import annotations

import os

# Clear broken SSL cert paths that crash httpx/huggingface_hub on some Windows setups.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if __name__ == "__main__":
    # Split data before importing PyTorch (fixes Windows 0xC0000005 crash).
    from training.config import TrainConfig
    from training.data_prep import load_or_create_splits

    print("Preparing data splits (no PyTorch)...", flush=True)
    load_or_create_splits(TrainConfig())

    from training.train import main

    main()
