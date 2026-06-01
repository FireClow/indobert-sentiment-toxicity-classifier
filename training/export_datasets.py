"""Export raw datasets for training into dataset/raw/."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("dataset/raw")
ABUSIVE_URL = (
    "https://raw.githubusercontent.com/okkyibrohim/"
    "id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv"
)
SMSA_PARQUET_FILES = {
    "train": "data/train-00000-of-00001-026a27ce8c45085d.parquet",
    "validation": "data/validation-00000-of-00001-495bab7e43ecc265.parquet",
    "test": "data/test-00000-of-00001-8e9d52f9ccaaffe4.parquet",
}
SMSA_HF_REPO = "kornwtp/smsa-ind-classification"
LABEL_MAP = {0: "positive", 1: "neutral", 2: "negative"}


def _download_hf_parquet(repo: str, filename: str) -> pd.DataFrame:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def export_abusive_dataset(output_path: Path = RAW_DIR / "indonesian_abusive_twitter.csv") -> Path:
    print("Downloading Indonesian Abusive/Hate Speech dataset...")
    df = pd.read_csv(ABUSIVE_URL, encoding="latin-1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {output_path}")
    return output_path


def export_smsa_dataset(output_path: Path = RAW_DIR / "indonlu_smsa.csv") -> Path:
    print("Downloading IndoNLU SmSA from Hugging Face mirror...")
    frames = []
    for split, filename in SMSA_PARQUET_FILES.items():
        df = _download_hf_parquet(SMSA_HF_REPO, filename)
        df = df.rename(columns={"texts": "text", "labels": "label"})
        if df["label"].dtype != object:
            df["label"] = df["label"].map(LABEL_MAP)
        else:
            df["label"] = df["label"].astype(str).str.lower().str.strip()
        frames.append(df[["text", "label"]])
        print(f"  {split}: {len(df)} rows")

    df = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows -> {output_path}")
    return output_path


def export_indotoxic_dataset(output_path: Path = RAW_DIR / "indotoxic2024.csv") -> Path:
    print("Downloading IndoDiscourse (IndoToxic2024 successor) from Hugging Face...")
    df = _download_hf_parquet("JUU198123/IndoDiscourse", "data/train-00000-of-00001.parquet")

    rows = []
    for _, row in df.iterrows():
        toxicity_list = row["toxicity"]
        if isinstance(toxicity_list, list) and len(toxicity_list) > 0:
            toxic_votes = sum(int(v) for v in toxicity_list)
            label = "1" if toxic_votes > len(toxicity_list) / 2 else "0"
        else:
            label = "0"
        rows.append({"text": row["text"], "label": label})

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows -> {output_path}")
    return output_path


def main() -> None:
    os.environ.pop("SSL_CERT_FILE", None)
    os.environ.pop("REQUESTS_CA_BUNDLE", None)

    export_abusive_dataset()
    export_smsa_dataset()
    try:
        export_indotoxic_dataset()
    except Exception as exc:
        print(f"Skipped optional IndoDiscourse export: {exc}")
    print("All required exports completed.")


if __name__ == "__main__":
    main()
