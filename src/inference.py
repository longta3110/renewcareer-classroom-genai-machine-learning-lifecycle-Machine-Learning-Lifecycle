from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader import clean_text
from src.model import SentimentModel
from src.utils import load_config


def load_model(model_path: str | Path | None = None):
    """Load the configured trained model."""
    if model_path is None:
        model_path = load_config()["model"]["artifact_path"]
    return SentimentModel.load(model_path)


def predict(text: str, model_path: str | Path | None = None) -> dict[str, float | str]:
    """Predict sentiment and confidence for a single text string."""
    model = load_model(model_path)
    cleaned_text = clean_text(text)
    prediction = int(model.predict([cleaned_text])[0])
    sentiment = "positive" if prediction == 1 else "negative"

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(np.max(probabilities))

    return {"sentiment": sentiment, "confidence": confidence}


def batch_predict(texts: list[str], model_path: str | Path | None = None) -> list[dict[str, float | str]]:
    """Predict sentiment for many text strings using one loaded model."""
    model = load_model(model_path)
    cleaned_texts = [clean_text(text) for text in texts]
    predictions = model.predict(cleaned_texts)
    probabilities = model.predict_proba(cleaned_texts) if hasattr(model, "predict_proba") else None

    results = []
    for index, prediction in enumerate(predictions):
        confidence = float(np.max(probabilities[index])) if probabilities is not None else 1.0
        results.append(
            {
                "sentiment": "positive" if int(prediction) == 1 else "negative",
                "confidence": confidence,
            }
        )
    return results
