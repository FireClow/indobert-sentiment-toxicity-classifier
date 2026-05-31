from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, BertTokenizer, PreTrainedTokenizerBase


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
    
    # Jika tidak ada file dictionary dari hasil training, gunakan default ini
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

    # 1. Load Base Model & Tokenizer dari HuggingFace
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = IndoBERTMultitaskModel(
        model_name=base_model_name,
        sentiment_num_labels=sentiment_num_labels,
        toxicity_num_labels=toxicity_num_labels,
    )

    # 2. VALIDASI KRITIKAL: Pastikan file weights hasil training benar-benar ada
    checkpoint_path = artifacts_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"❌ CRITICAL ERROR: File bobot model tidak ditemukan di path: {checkpoint_path.absolute()}\n"
            "Sistem dihentikan karena jika dilanjutkan, model hanya akan menggunakan angka acak (random weights). "
            "Pastikan folder 'saved_model/model' ada di dalam root project dan berisi file 'best_model.pt'."
        )

    # 3. Load Weights dengan sistem Fallback
    state = torch.load(checkpoint_path, map_location="cpu")
    
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # Berjaga-jaga jika tim ML menyimpan langsung state_dict-nya (tanpa dictionary wrapper)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    
    label_maps = _load_label_maps(artifacts_dir)
    return model, tokenizer, device, label_maps