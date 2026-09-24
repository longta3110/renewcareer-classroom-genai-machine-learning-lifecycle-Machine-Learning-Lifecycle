from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_parent_dir, load_config, resolve_project_path


LABEL_MAP = {"negative": 0, "positive": 1, 0: 0, 1: 1}


def clean_text(text: object) -> str:
    """Normalize review text for classical NLP features."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_label(label: object) -> int:
    """Map positive/negative labels to integers."""
    normalized = str(label).strip().lower()
    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    raise ValueError(f"Unsupported sentiment label: {label!r}")


def load_reviews(path: str | Path, text_column: str = "review", label_column: str = "sentiment") -> pd.DataFrame:
    """Load a sentiment CSV and validate required columns."""
    dataset_path = resolve_project_path(path)
    df = pd.read_csv(dataset_path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df[[text_column, label_column]].copy()


def preprocess_reviews(
    df: pd.DataFrame,
    text_column: str = "review",
    label_column: str = "sentiment",
) -> pd.DataFrame:
    """Clean review text, encode labels, and drop empty rows."""
    processed = df[[text_column, label_column]].copy()
    processed[text_column] = processed[text_column].map(clean_text)
    processed[label_column] = processed[label_column].map(encode_label)
    processed = processed[processed[text_column].str.len() > 0].reset_index(drop=True)
    return processed


@dataclass
class DataPreprocessor:
    """Load, clean, split, and save sentiment review data."""

    data_path: str | Path
    text_column: str = "review"
    label_column: str = "sentiment"
    test_size: float = 0.2
    random_state: int = 42

    def load_data(self) -> pd.DataFrame:
        return load_reviews(self.data_path, self.text_column, self.label_column)

    def preprocess(self) -> pd.DataFrame:
        return preprocess_reviews(self.load_data(), self.text_column, self.label_column)

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        processed = self.preprocess()
        train_df, test_df = train_test_split(
            processed,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=processed[self.label_column],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save_splits(
        self,
        train_path: str | Path = "data/processed/train.csv",
        test_path: str | Path = "data/processed/test.csv",
    ) -> tuple[Path, Path]:
        train_df, test_df = self.split()
        train_output = ensure_parent_dir(train_path)
        test_output = ensure_parent_dir(test_path)
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        return train_output, test_output


def prepare_data(config_path: str | Path | None = None) -> tuple[Path, Path]:
    """Prepare train/test CSV files from configured raw data."""
    config = load_config(config_path) if config_path else load_config()
    data_config = config["data"]
    preprocessor = DataPreprocessor(
        data_path=data_config["dataset_path"],
        text_column=data_config.get("text_column", "review"),
        label_column=data_config.get("label_column", "sentiment"),
        test_size=data_config.get("test_size", 0.2),
        random_state=data_config.get("random_state", 42),
    )
    return preprocessor.save_splits()


if __name__ == "__main__":
    train_path, test_path = prepare_data()
    print(f"Saved train split: {train_path}")
    print(f"Saved test split: {test_path}")
