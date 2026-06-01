from torch.utils.data import Dataset

from training.data_prep import SplitData, load_and_prepare_dataframe, load_or_create_splits, split_dataframe

__all__ = [
    "MultitaskCommentDataset",
    "SplitData",
    "load_and_prepare_dataframe",
    "load_or_create_splits",
    "split_dataframe",
]


class MultitaskCommentDataset(Dataset):
    """Keep a view on the dataframe instead of copying all texts into Python lists."""

    def __init__(self, df, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch

        row = self.df.iloc[idx]
        text = row["text"]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["sentiment_labels"] = torch.tensor(int(row["sentiment_id"]), dtype=torch.long)
        item["toxicity_labels"] = torch.tensor(int(row["toxicity_id"]), dtype=torch.long)
        return item
