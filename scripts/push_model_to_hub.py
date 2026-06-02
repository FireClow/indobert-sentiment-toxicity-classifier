"""
Upload trained artifacts to Hugging Face Hub (alternative to Git LFS).

Usage:
  set HF_TOKEN=hf_...
  python scripts/push_model_to_hub.py --repo-id YOUR_USERNAME/indobert-sentiment-toxicity-multitask
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_model" / "model"
TOKENIZER_DIR = ROOT / "saved_model" / "tokenizer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. FireClow/indobert-sentiment-toxicity-multitask",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running.")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo_id, exist_ok=True, repo_type="model")

    for name in ("best_model.pt", "label_mappings.pt"):
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Uploading {path}...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
        )

    for path in TOKENIZER_DIR.glob("*"):
        if path.is_file() and path.name != ".gitkeep":
            print(f"Uploading tokenizer/{path.name}...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"tokenizer/{path.name}",
                repo_id=args.repo_id,
                repo_type="model",
            )

    print(f"Done. Set Streamlit secret HF_MODEL_REPO={args.repo_id}")


if __name__ == "__main__":
    main()
