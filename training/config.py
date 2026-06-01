from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "indobenchmark/indobert-base-p1"
    dataset_path: str = "dataset/final_dataset.csv"
    output_model_dir: str = "saved_model/model"
    output_tokenizer_dir: str = "saved_model/tokenizer"
    max_length: int = 128
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    sentiment_loss_weight: float = 1.0
    toxicity_loss_weight: float = 1.0
    random_seed: int = 42
    early_stopping_patience: int = 2
    val_size: float = 0.1
    test_size: float = 0.1


SENTIMENT_LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABEL2ID.items()}

TOXICITY_LABEL2ID = {"non_toxic": 0, "toxic": 1}
TOXICITY_ID2LABEL = {v: k for k, v in TOXICITY_LABEL2ID.items()}

