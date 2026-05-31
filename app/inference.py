import torch

from app.preprocessing import preprocess_text
from app.utils import PredictionResult, safe_float


def predict_text(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    device: str,
    sentiment_id2label: dict[int, str],
    toxicity_id2label: dict[int, str],
    max_length: int = 128,
) -> PredictionResult:
    cleaned = preprocess_text(text)
    encoded = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)[0]
        toxicity_probs = torch.softmax(outputs["toxicity_logits"], dim=-1)[0]

    sentiment_id = int(torch.argmax(sentiment_probs).item())
    toxicity_id = int(torch.argmax(toxicity_probs).item())

    sentiment_map = {
        sentiment_id2label[i]: safe_float(sentiment_probs[i].item())
        for i in range(sentiment_probs.shape[0])
    }
    toxicity_map = {
        toxicity_id2label[i]: safe_float(toxicity_probs[i].item())
        for i in range(toxicity_probs.shape[0])
    }

    return PredictionResult(
        cleaned_text=cleaned,
        sentiment_label=sentiment_id2label[sentiment_id],
        sentiment_confidence=safe_float(sentiment_probs[sentiment_id].item()),
        sentiment_probabilities=sentiment_map,
        toxicity_label=toxicity_id2label[toxicity_id],
        toxicity_confidence=safe_float(toxicity_probs[toxicity_id].item()),
        toxicity_probabilities=toxicity_map,
    )

