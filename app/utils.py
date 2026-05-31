from dataclasses import dataclass


SENTIMENT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_LABEL2ID = {v: k for k, v in SENTIMENT_ID2LABEL.items()}

TOXICITY_ID2LABEL = {0: "non_toxic", 1: "toxic"}
TOXICITY_LABEL2ID = {v: k for k, v in TOXICITY_ID2LABEL.items()}


@dataclass(frozen=True)
class PredictionResult:
    cleaned_text: str
    sentiment_label: str
    sentiment_confidence: float
    sentiment_probabilities: dict[str, float]
    toxicity_label: str
    toxicity_confidence: float
    toxicity_probabilities: dict[str, float]


def safe_float(value: float) -> float:
    return round(float(value), 4)

