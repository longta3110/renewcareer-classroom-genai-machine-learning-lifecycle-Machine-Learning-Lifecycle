from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.model import train_model
from src.utils import load_config, resolve_project_path, save_json


def evaluate_model(model, test_df: pd.DataFrame, text_column: str = "review", label_column: str = "sentiment") -> dict:
    """Evaluate a classifier on a held-out sentiment test set."""
    y_true = test_df[label_column]
    y_pred = model.predict(test_df[text_column])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_saved_model(config_path: str | Path | None = None) -> dict:
    """Load the configured model and test split, then save metrics."""
    from src.model import SentimentModel

    config = load_config(config_path) if config_path else load_config()
    model_path = resolve_project_path(config["model"]["artifact_path"])
    if not model_path.exists():
        train_model(config_path)

    test_df = pd.read_csv(resolve_project_path("data/processed/test.csv"))
    model = SentimentModel.load(model_path)
    metrics = evaluate_model(
        model,
        test_df,
        text_column=config["data"].get("text_column", "review"),
        label_column=config["data"].get("label_column", "sentiment"),
    )
    save_json(metrics, config["evaluation"].get("metrics_path", "eval_metrics.json"))
    return metrics


if __name__ == "__main__":
    print(evaluate_saved_model())
