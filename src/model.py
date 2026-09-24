from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import prepare_data
from src.utils import ensure_parent_dir, load_config, resolve_project_path


@dataclass
class SentimentModel:
    """Train and persist a sentiment classifier."""

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    analyzer: str = "word"
    max_iter: int = 1000

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=self.ngram_range,
                        analyzer=self.analyzer,
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=self.max_iter, class_weight="balanced")),
            ]
        )

    def train(self, train_df: pd.DataFrame, text_column: str = "review", label_column: str = "sentiment") -> Pipeline:
        pipeline = self.build_pipeline()
        pipeline.fit(train_df[text_column], train_df[label_column])
        return pipeline

    def save(self, pipeline: Pipeline, path: str | Path) -> Path:
        output_path = ensure_parent_dir(path)
        joblib.dump(pipeline, output_path)
        return output_path

    @staticmethod
    def load(path: str | Path) -> Pipeline:
        return joblib.load(resolve_project_path(path))


def train_model(config_path: str | Path | None = None) -> Path:
    """Train the configured sentiment model and save it to disk."""
    config = load_config(config_path) if config_path else load_config()
    data_config = config["data"]
    model_config = config["model"]

    train_path = resolve_project_path("data/processed/train.csv")
    if not train_path.exists():
        prepare_data(config_path)

    train_df = pd.read_csv(train_path)
    model = SentimentModel(
        max_features=model_config.get("max_features", 5000),
        ngram_range=tuple(model_config.get("ngram_range", [1, 2])),
        analyzer=model_config.get("analyzer", "word"),
        max_iter=model_config.get("max_iter", 1000),
    )
    pipeline = model.train(
        train_df,
        text_column=data_config.get("text_column", "review"),
        label_column=data_config.get("label_column", "sentiment"),
    )
    return model.save(pipeline, model_config["artifact_path"])


if __name__ == "__main__":
    artifact_path = train_model()
    print(f"Saved model: {artifact_path}")
