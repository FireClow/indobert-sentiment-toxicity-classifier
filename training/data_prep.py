"""Dataset load/split without PyTorch (avoids Windows native crash with torch + sklearn)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.preprocessing import preprocess_text
from training.config import SENTIMENT_LABEL2ID, TOXICITY_LABEL2ID, TrainConfig

REQUIRED_COLUMNS = {"text", "sentiment_label", "toxicity_label"}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Dataset missing required columns: {missing_str}")


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
    df["toxicity_label"] = df["toxicity_label"].astype(str).str.strip().str.lower()
    return df


def _prepared_cache_path(config: TrainConfig) -> Path:
    return Path(config.dataset_path).with_name(".prepared_dataframe.pkl")


def _splits_cache_path(config: TrainConfig) -> Path:
    return Path(config.dataset_path).with_name(".splits.pkl")


def load_and_prepare_dataframe(config: TrainConfig) -> pd.DataFrame:
    cache_path = _prepared_cache_path(config)
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}...", flush=True)
        return pd.read_pickle(cache_path)

    print(f"Reading {config.dataset_path}...", flush=True)
    df = pd.read_csv(config.dataset_path)
    _validate_columns(df)
    df = _normalize_labels(df)
    print("Preprocessing text...", flush=True)
    texts = df["text"].fillna("").astype(str).tolist()
    df["text"] = [preprocess_text(text) for text in texts]
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["sentiment_id"] = df["sentiment_label"].map(SENTIMENT_LABEL2ID)
    df["toxicity_id"] = df["toxicity_label"].map(TOXICITY_LABEL2ID)
    df = df.dropna(subset=["sentiment_id", "toxicity_id"]).reset_index(drop=True)
    df["sentiment_id"] = df["sentiment_id"].astype(int)
    df["toxicity_id"] = df["toxicity_id"].astype(int)
    df["stratify_key"] = df["sentiment_label"] + "__" + df["toxicity_label"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Cached prepared dataset to {cache_path}", flush=True)
    return df


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def split_dataframe(df: pd.DataFrame, config: TrainConfig) -> SplitData:
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=df["stratify_key"],
    )
    val_ratio_on_train_val = config.val_size / (1.0 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_on_train_val,
        random_state=config.random_seed,
        stratify=train_val_df["stratify_key"],
    )
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


def load_or_create_splits(config: TrainConfig) -> SplitData:
    """Load train/val/test splits from cache, or create them without importing PyTorch."""
    splits_path = _splits_cache_path(config)
    if splits_path.exists():
        print(f"Loading cached splits from {splits_path}...", flush=True)
        with splits_path.open("rb") as f:
            return pickle.load(f)

    print("Loading dataset for split...", flush=True)
    df = load_and_prepare_dataframe(config)
    print(f"Dataset ready: {len(df)} rows", flush=True)
    print("Splitting dataset (no PyTorch)...", flush=True)
    splits = split_dataframe(df, config)
    del df
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("wb") as f:
        pickle.dump(splits, f)
    print(
        f"Cached splits: train={len(splits.train_df)} val={len(splits.val_df)} test={len(splits.test_df)}",
        flush=True,
    )
    return splits
