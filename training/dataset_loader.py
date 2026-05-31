from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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


def load_and_prepare_dataframe(config: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(config.dataset_path)
    _validate_columns(df)
    df = _normalize_labels(df)
    df["text"] = df["text"].fillna("").astype(str).map(preprocess_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["sentiment_id"] = df["sentiment_label"].map(SENTIMENT_LABEL2ID)
    df["toxicity_id"] = df["toxicity_label"].map(TOXICITY_LABEL2ID)
    df = df.dropna(subset=["sentiment_id", "toxicity_id"]).reset_index(drop=True)
    df["sentiment_id"] = df["sentiment_id"].astype(int)
    df["toxicity_id"] = df["toxicity_id"].astype(int)
    df["stratify_key"] = df["sentiment_label"] + "__" + df["toxicity_label"]
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


class MultitaskCommentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.sentiments = df["sentiment_id"].tolist()
        self.toxicities = df["toxicity_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["sentiment_labels"] = torch.tensor(self.sentiments[idx], dtype=torch.long)
        item["toxicity_labels"] = torch.tensor(self.toxicities[idx], dtype=torch.long)
        return item

