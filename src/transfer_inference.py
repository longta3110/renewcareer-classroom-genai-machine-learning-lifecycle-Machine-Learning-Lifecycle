"""Offline serving of an explicitly trained transformer checkpoint."""
import json
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import resolve_project_path


@lru_cache(maxsize=2)
def load_service(path="models/transformer"):
    path = resolve_project_path(path)
    if not (path / "training.json").exists():
        raise FileNotFoundError("Train the transformer pipeline before starting the API.")
    info = json.loads((path / "training.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    return tokenizer, model, info


def predict_texts(texts, path="models/transformer"):
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Provide at least one non-empty text string")
    tokenizer, model, info = load_service(path)
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            tokens = tokenizer(texts[start:start + 16], padding=True, truncation=True,
                               max_length=info["max_length"], return_tensors="pt")
            scores, labels = model(**tokens).logits.softmax(-1).max(-1)
            output.extend({"sentiment": model.config.id2label[label], "confidence": score}
                          for label, score in zip(labels.tolist(), scores.tolist()))
    return output
