from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from training.config import SENTIMENT_ID2LABEL, TOXICITY_ID2LABEL, TrainConfig
from training.data_prep import load_or_create_splits
from training.dataset_loader import MultitaskCommentDataset
from training.train import IndoBERTMultitaskModel
from transformers import AutoTokenizer


def _collect_predictions(model, data_loader, device):
    model.eval()
    sent_preds, sent_trues = [], []
    tox_preds, tox_trues = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            sent_labels = batch["sentiment_labels"].to(device)
            tox_labels = batch["toxicity_labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            sent_batch_preds = torch.argmax(outputs["sentiment_logits"], dim=1)
            tox_batch_preds = torch.argmax(outputs["toxicity_logits"], dim=1)

            sent_preds.extend(sent_batch_preds.cpu().numpy().tolist())
            sent_trues.extend(sent_labels.cpu().numpy().tolist())
            tox_preds.extend(tox_batch_preds.cpu().numpy().tolist())
            tox_trues.extend(tox_labels.cpu().numpy().tolist())

    return sent_trues, sent_preds, tox_trues, tox_preds


def _plot_confusion_matrix(cm, labels, title, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.output_tokenizer_dir)
    splits = load_or_create_splits(config)
    test_dataset = MultitaskCommentDataset(splits.test_df, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = IndoBERTMultitaskModel(config.model_name).to(device)
    checkpoint = torch.load(Path(config.output_model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    sent_trues, sent_preds, tox_trues, tox_preds = _collect_predictions(model, test_loader, device)
    sentiment_labels = [SENTIMENT_ID2LABEL[i] for i in sorted(SENTIMENT_ID2LABEL)]
    toxicity_labels = [TOXICITY_ID2LABEL[i] for i in sorted(TOXICITY_ID2LABEL)]

    print("\n=== Sentiment Metrics ===")
    print(f"Accuracy: {accuracy_score(sent_trues, sent_preds):.4f}")
    print(classification_report(sent_trues, sent_preds, target_names=sentiment_labels, digits=4))

    print("\n=== Toxicity Metrics ===")
    print(f"Accuracy: {accuracy_score(tox_trues, tox_preds):.4f}")
    print(classification_report(tox_trues, tox_preds, target_names=toxicity_labels, digits=4))

    sent_cm = confusion_matrix(sent_trues, sent_preds)
    tox_cm = confusion_matrix(tox_trues, tox_preds)

    _plot_confusion_matrix(
        sent_cm,
        sentiment_labels,
        "Sentiment Confusion Matrix",
        "assets/confusion_matrix_sentiment.png",
    )
    _plot_confusion_matrix(
        tox_cm,
        toxicity_labels,
        "Toxicity Confusion Matrix",
        "assets/confusion_matrix_toxicity.png",
    )

    combined_canvas = np.concatenate([sent_cm, np.zeros((3, 1), dtype=int), np.pad(tox_cm, ((0, 1), (0, 1)))], axis=1)
    _plot_confusion_matrix(
        combined_canvas,
        ["S0", "S1", "S2", "", "T0", "T1", ""],
        "Combined Confusion Matrix Overview",
        "assets/confusion_matrix.png",
    )
    print("Saved confusion matrix images to assets/.")


if __name__ == "__main__":
    evaluate(TrainConfig())

