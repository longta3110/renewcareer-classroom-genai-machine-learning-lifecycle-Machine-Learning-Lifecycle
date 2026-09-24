"""Evaluate the selected checkpoint on the untouched test split."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.transfer_model import Reviews, metrics, predict_loader
from src.utils import resolve_project_path, save_json


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = resolve_project_path("models/transformer")
    info = json.loads((output / "training.json").read_text())
    frame = pd.read_csv(resolve_project_path("data/processed/imdb/test.csv"))
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True)
    loader = DataLoader(Reviews(frame, tokenizer, info["max_length"]), batch_size=16)
    labels, predicted, confidence = predict_loader(model, loader, torch.device("cpu"))
    scores = metrics(labels, predicted)
    save_json(scores, "reports/test_metrics.json")
    frame["prediction"], frame["confidence"] = predicted, confidence
    report = resolve_project_path("reports")
    frame[frame.label != frame.prediction].to_csv(report / "errors.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(labels, predicted, labels=[0, 1], display_labels=["negative", "positive"])
    plt.tight_layout()
    plt.savefig(report / "confusion_matrix.png")
    plt.close()
    mlflow.set_tracking_uri("sqlite:///" + resolve_project_path("mlflow.db").as_posix())
    with mlflow.start_run(run_id=info["run_id"]):
        mlflow.log_metrics({f"test_{key}": value for key, value in scores.items()})
        mlflow.log_artifacts(str(report), artifact_path="evaluation")
    print(scores)


if __name__ == "__main__":
    main()
