import os

# Clear broken SSL cert paths (Windows) before huggingface_hub / httpx init.
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

# Optional: set in Streamlit Cloud secrets if model is on Hugging Face Hub instead of Git LFS.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
DEFAULT_BASE_MODEL = "indobenchmark/indobert-base-p1"


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _download_from_hub(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(path)


def _resolve_file(filename: str, local_dir: Path, hub_subpath: str | None = None) -> Path:
    local_path = local_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if HF_MODEL_REPO:
        hub_name = hub_subpath or filename
        return _download_from_hub(HF_MODEL_REPO, hub_name, local_dir)

    raise FileNotFoundError(
        f"Required artifact not found: {local_path}\n"
        "For Streamlit Cloud: ensure Git LFS files are pulled (best_model.pt in repo) "
        "or set HF_MODEL_REPO in app secrets to a Hugging Face model repo."
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_label_maps(artifacts_dir: Path) -> dict[str, Any]:
    label_path = artifacts_dir / "label_mappings.pt"
    if label_path.exists():
        return _torch_load(label_path)
    if HF_MODEL_REPO:
        label_path = _download_from_hub(HF_MODEL_REPO, "label_mappings.pt", artifacts_dir)
        return _torch_load(label_path)
    return {
        "sentiment_id2label": {0: "negative", 1: "neutral", 2: "positive"},
        "toxicity_id2label": {0: "non_toxic", 1: "toxic"},
    }


def _load_tokenizer(tokenizer_dir: Path, base_model_name: str) -> PreTrainedTokenizerBase:
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        return AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if HF_MODEL_REPO:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_MODEL_REPO,
            local_dir=str(tokenizer_dir),
            allow_patterns=["tokenizer*.json", "*.txt", "*.model"],
        )
        if tokenizer_config.exists():
            return AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return AutoTokenizer.from_pretrained(base_model_name)


def load_model_and_tokenizer(
    model_dir: str = "saved_model/model",
    tokenizer_dir: str = "saved_model/tokenizer",
    base_model_name: str = DEFAULT_BASE_MODEL,
    sentiment_num_labels: int = 3,
    toxicity_num_labels: int = 2,
) -> tuple[IndoBERTMultitaskModel, PreTrainedTokenizerBase, str, dict[str, Any]]:
    root = _repo_root()
    artifacts_dir = root / model_dir
    tok_dir = root / tokenizer_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = _resolve_file("best_model.pt", artifacts_dir)
    tokenizer = _load_tokenizer(tok_dir, base_model_name)

    model = IndoBERTMultitaskModel(
        model_name=base_model_name,
        sentiment_num_labels=sentiment_num_labels,
        toxicity_num_labels=toxicity_num_labels,
    )

    state = _torch_load(checkpoint_path)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    label_maps = _load_label_maps(artifacts_dir)
    return model, tokenizer, device, label_maps
