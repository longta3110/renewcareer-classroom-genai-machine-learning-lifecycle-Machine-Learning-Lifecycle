"""Serve the trained transformer without training inside HTTP requests."""
import os
import time

from flask import Flask, Response, g, jsonify, request
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from src.transfer_inference import load_service, predict_texts

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
REQUESTS = Counter("sentiment_requests_total", "API requests", ["endpoint", "status"])
LATENCY = Histogram("sentiment_request_seconds", "API request latency", ["endpoint"])


@app.before_request
def begin_request():
    g.started = time.monotonic()


@app.after_request
def record_request(response):
    endpoint = request.endpoint or "unknown"
    REQUESTS.labels(endpoint, str(response.status_code)).inc()
    LATENCY.labels(endpoint).observe(time.monotonic() - g.started)
    return response


@app.get("/metrics")
def monitoring():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    try:
        _, _, info = load_service()
    except (OSError, ValueError):
        return jsonify(status="unavailable", model_available=False), 503
    return jsonify(status="ok", model_available=True, model_version=info["run_id"])


def valid_text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 20000


def infer(texts):
    try:
        return predict_texts(texts)
    except (OSError, ValueError):
        app.logger.exception("Model unavailable")
        return None


@app.post("/predict")
def predict_endpoint():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not valid_text(payload.get("text")):
        return jsonify(error="Provide a non-empty text string, at most 20000 characters."), 400
    result = infer([payload["text"]])
    if result is None:
        return jsonify(error="Model unavailable; train and deploy a checkpoint first."), 503
    return jsonify(result[0])


@app.post("/predict-batch")
def predict_batch_endpoint():
    payload = request.get_json(silent=True)
    texts = payload.get("texts") if isinstance(payload, dict) else None
    if not isinstance(texts, list) or not 1 <= len(texts) <= 64 or not all(map(valid_text, texts)):
        return jsonify(error="Provide 1-64 non-empty strings in texts, each at most 20000 characters."), 400
    result = infer(texts)
    if result is None:
        return jsonify(error="Model unavailable; train and deploy a checkpoint first."), 503
    return jsonify(predictions=result)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
