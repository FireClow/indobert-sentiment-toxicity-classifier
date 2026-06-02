"""
Entry point for evaluation. Prefer this over `python -m training.evaluate` on Windows.

Usage (PowerShell):
  cd "C:\\Users\\kenji\\Sentiment and Toxicity Classification of Indonesian Social Media Comments Using IndoBERT"
  python run_evaluate.py
"""
from __future__ import annotations

import os

# Clear broken SSL cert paths that crash httpx/huggingface_hub on some Windows setups.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

if __name__ == "__main__":
    from training.config import TrainConfig
    from training.evaluate import evaluate

    evaluate(TrainConfig())
