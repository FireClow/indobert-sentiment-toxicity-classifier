import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.config import SENTIMENT_ID2LABEL, TOXICITY_ID2LABEL, TrainConfig
from training.data_prep import load_or_create_splits
from training.dataset_loader import MultitaskCommentDataset, SplitData

# Avoid Hugging Face progress-bar / stderr issues on some Windows terminals.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _configure_runtime() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _import_transformers():
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    return AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class IndoBERTMultitaskModel(torch.nn.Module):
    def __init__(self, model_name: str, sentiment_num_labels: int = 3, toxicity_num_labels: int = 2):
        super().__init__()
        AutoModel, _, _ = _import_transformers()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.sentiment_head = torch.nn.Linear(hidden_size, sentiment_num_labels)
        self.toxicity_head = torch.nn.Linear(hidden_size, toxicity_num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {
            "sentiment_logits": self.sentiment_head(pooled),
            "toxicity_logits": self.toxicity_head(pooled),
        }


def set_cpu_seed(seed: int) -> None:
    """Seed Python/NumPy only. Avoid torch.manual_seed before dataloaders (Windows crash)."""
    random.seed(seed)
    np.random.seed(seed)


def set_cuda_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(config: TrainConfig, tokenizer, splits: SplitData):
    print("Creating PyTorch datasets...", flush=True)

    train_dataset = MultitaskCommentDataset(splits.train_df, tokenizer, config.max_length)
    val_dataset = MultitaskCommentDataset(splits.val_df, tokenizer, config.max_length)
    test_dataset = MultitaskCommentDataset(splits.test_df, tokenizer, config.max_length)
    print("Creating DataLoaders...", flush=True)

    # Do not touch CUDA here; pin_memory=False avoids driver init during dataset load.
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_eval(model, data_loader, device, sent_w, tox_w):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

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
            sent_loss = ce(outputs["sentiment_logits"], sent_labels)
            tox_loss = ce(outputs["toxicity_logits"], tox_labels)
            loss = sent_w * sent_loss + tox_w * tox_loss
            total_loss += loss.item()

    return total_loss / max(1, len(data_loader))


def train(config: TrainConfig):
    _configure_runtime()
    print("Training started...", flush=True)

    # Keep CUDA uninitialized while loading CSV / building datasets (fixes Windows crash).
    set_cpu_seed(config.random_seed)

    _, AutoTokenizer, get_linear_schedule_with_warmup = _import_transformers()

    print("Loading train/val/test splits...", flush=True)
    splits = load_or_create_splits(config)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Building dataloaders...", flush=True)
    train_loader, val_loader, test_loader = build_dataloaders(config, tokenizer, splits)
    print(
        f"Dataloaders ready: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}",
        flush=True,
    )

    device = resolve_device()
    set_cuda_seed(config.random_seed)
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading model to GPU...", flush=True)
    model = IndoBERTMultitaskModel(config.model_name).to(device)
    print("Model ready.", flush=True)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    Path(config.output_model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_tokenizer_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            sent_labels = batch["sentiment_labels"].to(device)
            tox_labels = batch["toxicity_labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sent_loss = criterion(outputs["sentiment_logits"], sent_labels)
            tox_loss = criterion(outputs["toxicity_logits"], tox_labels)
            loss = config.sentiment_loss_weight * sent_loss + config.toxicity_loss_weight * tox_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} "
                    f"batch {batch_idx + 1}/{len(train_loader)} loss={loss.item():.4f}",
                    flush=True,
                )

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = run_eval(
            model=model,
            data_loader=val_loader,
            device=device,
            sent_w=config.sentiment_loss_weight,
            tox_w=config.toxicity_loss_weight,
        )
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": asdict(config),
                },
                os.path.join(config.output_model_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.", flush=True)
                break

    tokenizer.save_pretrained(config.output_tokenizer_dir)
    torch.save(
        {
            "sentiment_id2label": SENTIMENT_ID2LABEL,
            "toxicity_id2label": TOXICITY_ID2LABEL,
        },
        os.path.join(config.output_model_dir, "label_mappings.pt"),
    )
    print("Training finished. Artifacts saved.", flush=True)
    return test_loader


def main() -> None:
    train(TrainConfig())


if __name__ == "__main__":
    main()
