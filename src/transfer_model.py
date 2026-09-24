"""Fine-tune a pretrained HuggingFace classifier and select by validation F1."""
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import load_config, resolve_project_path, save_json


class Reviews(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.tokens = tokenizer(frame.text.tolist(), truncation=True, padding="max_length", max_length=max_length)
        self.labels = frame.label.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: torch.tensor(value[index]) for key, value in self.tokens.items()},
                "labels": torch.tensor(self.labels[index], dtype=torch.long)}


def metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}


def predict_loader(model, loader, device):
    model.eval()
    labels, predictions, confidence = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels.extend(batch.pop("labels").tolist())
            probs = model(**{key: value.to(device) for key, value in batch.items()}).logits.softmax(-1)
            scores, classes = probs.max(-1)
            predictions.extend(classes.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    return labels, predictions, confidence


def train(config=None):
    config = config or load_config("params.yaml")
    args = config["train"]
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args["model_name"], num_labels=2, id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1})
    if args["freeze_encoder"]:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    model.to(device)
    root = resolve_project_path("data/processed/imdb")
    loaders = {name: DataLoader(Reviews(pd.read_csv(root / f"{name}.csv"), tokenizer, args["max_length"]),
                                batch_size=args["batch_size"], shuffle=name == "train")
               for name in ("train", "validation")}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args["learning_rate"])
    output = resolve_project_path("models/transformer")
    mlflow.set_tracking_uri("sqlite:///" + resolve_project_path("mlflow.db").as_posix())
    mlflow.set_experiment(config["tracking"]["experiment"])
    best, history = -1.0, []
    with mlflow.start_run() as run:
        mlflow.log_params({**args, **{f"data_{k}": v for k, v in config["data"].items()}})
        for epoch in range(args["epochs"]):
            model.train()
            if args["freeze_encoder"]:
                model.base_model.eval()
            total_loss = 0.0
            for batch in loaders["train"]:
                optimizer.zero_grad()
                loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * batch["labels"].shape[0]
            labels, predicted, _ = predict_loader(model, loaders["validation"], device)
            scores = {f"validation_{key}": value for key, value in metrics(labels, predicted).items()}
            scores["train_loss"] = total_loss / len(loaders["train"].dataset)
            mlflow.log_metrics(scores, step=epoch)
            history.append({"epoch": epoch + 1, **scores})
            print(history[-1], flush=True)
            if scores["validation_f1"] > best:
                best = scores["validation_f1"]
                model.save_pretrained(output)
                tokenizer.save_pretrained(output)
        save_json({"run_id": run.info.run_id, "best_validation_f1": best,
                   "max_length": args["max_length"], "history": history, "parameters": args}, output / "training.json")
        mlflow.log_metric("best_validation_f1", best)
        mlflow.log_artifacts(str(output), artifact_path="model")
        save_json({"best_validation_f1": best}, "reports/train_metrics.json")
    return output


if __name__ == "__main__":
    train()
