from pathlib import Path

import pandas as pd

from src.inference import batch_predict, predict
from src.model import SentimentModel


def test_predict_returns_sentiment_and_confidence(tmp_path):
    train_df = pd.DataFrame(
        {
            "review": ["great excellent movie", "bad awful movie"],
            "sentiment": [1, 0],
        }
    )
    model_path = tmp_path / "model.joblib"
    helper = SentimentModel(max_features=50)
    pipeline = helper.train(train_df)
    helper.save(pipeline, model_path)

    result = predict("excellent movie", model_path=model_path)
    assert result["sentiment"] in {"positive", "negative"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_batch_predict_returns_one_result_per_text(tmp_path):
    train_df = pd.DataFrame(
        {
            "review": ["great excellent movie", "bad awful movie"],
            "sentiment": [1, 0],
        }
    )
    model_path = Path(tmp_path) / "model.joblib"
    helper = SentimentModel(max_features=50)
    helper.save(helper.train(train_df), model_path)

    results = batch_predict(["great film", "awful film"], model_path=model_path)
    assert len(results) == 2
