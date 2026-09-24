"""Prepare reproducible, disjoint splits without changing transformer punctuation."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, resolve_project_path, save_json


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


def validate(frame):
    if not {"text", "label"} <= set(frame.columns):
        raise ValueError("Expected text and label columns")
    frame = frame[["text", "label"]].dropna().copy()
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    frame["text"] = frame.text.map(normalize)
    frame = frame[frame.text.str.len() > 0]
    conflicts = frame.groupby("text").label.nunique()
    frame = frame[~frame.text.isin(conflicts[conflicts > 1].index)]
    return frame.drop_duplicates("text").reset_index(drop=True)


def sample(frame, limit, seed):
    if not limit or limit >= len(frame):
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(frame, train_size=limit, stratify=frame.label, random_state=seed)
    return selected.reset_index(drop=True)


def split_reviews(train, test, config):
    train, test = validate(train), validate(test)
    train = train[~train.text.isin(test.text)]
    train = sample(train, config["train_limit"], config["seed"])
    train, validation = train_test_split(train, test_size=config["validation_size"],
                                         stratify=train.label, random_state=config["seed"])
    test = sample(test, config["test_limit"], config["seed"])
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test}


def main():
    root = resolve_project_path("data/raw/imdb")
    splits = split_reviews(pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv"),
                          load_config("params.yaml")["data"])
    output = resolve_project_path("data/processed/imdb")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    save_json({name: {"rows": len(frame), "labels": frame.label.value_counts().to_dict()}
               for name, frame in splits.items()}, "reports/data_summary.json")


if __name__ == "__main__":
    main()
