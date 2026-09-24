import pandas as pd

from src.evaluate import evaluate_model
from src.model import SentimentModel


def test_sentiment_model_can_train_and_predict():
    train_df = pd.DataFrame(
        {
            "review": [
                "wonderful excellent movie",
                "amazing and enjoyable story",
                "terrible boring movie",
                "awful and disappointing story",
            ],
            "sentiment": [1, 1, 0, 0],
        }
    )
    model = SentimentModel(max_features=100).train(train_df)
    predictions = model.predict(["excellent story", "boring awful movie"])
    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})


def test_evaluate_model_returns_standard_metrics():
    train_df = pd.DataFrame(
        {
            "review": ["great movie", "excellent film", "bad movie", "awful film"],
            "sentiment": [1, 1, 0, 0],
        }
    )
    model = SentimentModel(max_features=100).train(train_df)
    metrics = evaluate_model(model, train_df)
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"}.issubset(metrics)
    assert 0.0 <= metrics["accuracy"] <= 1.0
