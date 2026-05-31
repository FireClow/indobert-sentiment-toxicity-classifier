from pathlib import Path

import pandas as pd

from app.preprocessing import preprocess_text


def _load_abusive_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "Tweet" if "Tweet" in df.columns else "text"
    abusive_col = "HS" if "HS" in df.columns else "toxic"
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["toxicity_label"] = df[abusive_col].map(lambda x: "toxic" if int(x) == 1 else "non_toxic")
    out["sentiment_label"] = "neutral"
    out["source"] = "abusive_hate_speech"
    return out


def _load_smsa_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else "tweet"
    label_col = "label" if "label" in df.columns else "sentiment"
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["sentiment_label"] = (
        df[label_col].astype(str).str.lower().replace({"pos": "positive", "neg": "negative", "net": "neutral"})
    )
    out["toxicity_label"] = "non_toxic"
    out["source"] = "smsa"
    return out


def _load_optional_toxic_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else "comment"
    label_col = "label" if "label" in df.columns else "toxicity"
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["toxicity_label"] = (
        df[label_col].astype(str).str.lower().replace({"1": "toxic", "0": "non_toxic"})
    )
    out["sentiment_label"] = "neutral"
    out["source"] = "indotoxic2024"
    return out


def prepare_final_dataset(
    abusive_path: str,
    smsa_path: str,
    output_path: str = "dataset/final_dataset.csv",
    indotoxic_path: str | None = None,
) -> pd.DataFrame:
    frames = [_load_abusive_dataset(abusive_path), _load_smsa_dataset(smsa_path)]
    if indotoxic_path:
        frames.append(_load_optional_toxic_dataset(indotoxic_path))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str).map(preprocess_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "sentiment_label", "toxicity_label"])
    df = df.reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    # Update these paths based on your raw dataset file names.
    abusive_file = "dataset/raw/indonesian_abusive_twitter.csv"
    smsa_file = "dataset/raw/indonlu_smsa.csv"
    indotoxic_file = "dataset/raw/indotoxic2024.csv"

    optional_path = indotoxic_file if Path(indotoxic_file).exists() else None
    result_df = prepare_final_dataset(abusive_file, smsa_file, indotoxic_path=optional_path)
    print(f"Prepared dataset with {len(result_df)} rows -> dataset/final_dataset.csv")

