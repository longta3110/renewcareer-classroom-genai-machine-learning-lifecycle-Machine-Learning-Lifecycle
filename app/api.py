from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import batch_predict, predict
from src.model import train_model
from src.utils import load_config, resolve_project_path


app = Flask(__name__)


def ensure_model_exists() -> None:
    """Train the lightweight model on first API use if no artifact exists."""
    model_path = resolve_project_path(load_config()["model"]["artifact_path"])
    if not model_path.exists():
        train_model()


@app.get("/health")
def health():
    model_path = resolve_project_path(load_config()["model"]["artifact_path"])
    return jsonify(
        {
            "status": "ok",
            "model_available": model_path.exists(),
            "model_path": str(model_path),
        }
    )


@app.post("/predict")
def predict_endpoint():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request JSON must include a non-empty 'text' string."}), 400

    ensure_model_exists()
    return jsonify(predict(text))


@app.post("/predict-batch")
def predict_batch_endpoint():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(item, str) and item.strip() for item in texts):
        return jsonify({"error": "Request JSON must include 'texts' as a list of non-empty strings."}), 400

    ensure_model_exists()
    return jsonify({"predictions": batch_predict(texts)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
