from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


class IndoBERTMultitaskModel(torch.nn.Module):
    def __init__(self, model_name: str, sentiment_num_labels: int, toxicity_num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.sentiment_head = torch.nn.Linear(hidden_size, sentiment_num_labels)
        self.toxicity_head = torch.nn.Linear(hidden_size, toxicity_num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        sentiment_logits = self.sentiment_head(pooled)
        toxicity_logits = self.toxicity_head(pooled)
        return {"sentiment_logits": sentiment_logits, "toxicity_logits": toxicity_logits}


def _load_label_maps(artifacts_dir: Path) -> dict[str, Any]:
    label_map_path = artifacts_dir / "label_mappings.pt"
    if label_map_path.exists():
        return torch.load(label_map_path, map_location="cpu")
    return {
        "sentiment_id2label": {0: "negative", 1: "neutral", 2: "positive"},
        "toxicity_id2label": {0: "non_toxic", 1: "toxic"},
    }


def load_model_and_tokenizer(
    model_dir: str = "saved_model/model",
    tokenizer_dir: str = "saved_model/tokenizer",
    base_model_name: str = "indobenchmark/indobert-base-p1",
    sentiment_num_labels: int = 3,
    toxicity_num_labels: int = 2,
) -> tuple[IndoBERTMultitaskModel, PreTrainedTokenizerBase, str, dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_dir = Path(model_dir)
    tokenizer_path = Path(tokenizer_dir)

    if tokenizer_path.exists() and any(tokenizer_path.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = IndoBERTMultitaskModel(
        model_name=base_model_name,
        sentiment_num_labels=sentiment_num_labels,
        toxicity_num_labels=toxicity_num_labels,
    )

    checkpoint_path = artifacts_dir / "best_model.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

    model.to(device)
    model.eval()
    label_maps = _load_label_maps(artifacts_dir)
    return model, tokenizer, device, label_maps

