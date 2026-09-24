import pandas as pd
import pytest

from src.transfer_data import split_reviews, validate


def test_splits_are_disjoint_balanced_and_repeatable():
    train = pd.DataFrame({"text": [f"review {i}" for i in range(40)], "label": [0, 1] * 20})
    test = pd.DataFrame({"text": [f"review {i}" for i in range(30, 50)], "label": [0, 1] * 10})
    config = {"train_limit": 20, "test_limit": 10, "validation_size": .2, "seed": 42}
    splits = split_reviews(train, test, config)
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [16, 4, 10]
    assert not set(splits["train"].text) & set(splits["validation"].text)
    assert not set(splits["train"].text) & set(splits["test"].text)
    assert not set(splits["validation"].text) & set(splits["test"].text)
    repeated = split_reviews(train, test, config)
    for name, frame in splits.items():
        pd.testing.assert_frame_equal(frame, repeated[name])
        assert set(frame.label) == {0, 1}


def test_invalid_labels_and_conflicting_duplicates():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"text": ["x"], "label": [2]}))
    frame = validate(pd.DataFrame({"text": ["x", "x", "<br> ", "Good!"], "label": [0, 1, 0, 1]}))
    assert frame.text.tolist() == ["Good!"]


@pytest.mark.parametrize("payload", [None, [], "text", {"text": ""}, {"text": 7}])
def test_api_rejects_invalid_json(payload):
    from app.api import app
    assert app.test_client().post("/predict", json=payload).status_code == 400


def test_api_batch_limits_and_unavailable_model(monkeypatch):
    import app.api as api
    client = api.app.test_client()
    for texts in ([], ["ok"] * 65, [None], [" "]):
        assert client.post("/predict-batch", json={"texts": texts}).status_code == 400
    monkeypatch.setattr(api, "infer", lambda texts: None)
    assert client.post("/predict", json={"text": "Hello"}).status_code == 503


def test_api_prediction_contract(monkeypatch):
    import app.api as api
    monkeypatch.setattr(api, "predict_texts", lambda texts: [{"sentiment": "positive", "confidence": .75} for _ in texts])
    response = api.app.test_client().post("/predict-batch", json={"texts": ["good", "great"]})
    assert response.status_code == 200
    assert len(response.json["predictions"]) == 2
    assert response.json["predictions"][0]["confidence"] == .75


def test_transformer_save_reload_and_inference_offline(tmp_path):
    import json
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from src.transfer_inference import predict_texts

    torch.manual_seed(42)
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    tokenizer.save_pretrained(tmp_path)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}))
    model.save_pretrained(tmp_path)
    (tmp_path / "training.json").write_text(json.dumps({"max_length": 16, "run_id": "offline-test"}))
    result = predict_texts(["good movie", "bad movie"], str(tmp_path))
    assert len(result) == 2
    assert all(item["sentiment"] in {"positive", "negative"} for item in result)
    assert all(.5 <= item["confidence"] <= 1 for item in result)
    assert result == predict_texts(["good movie", "bad movie"], str(tmp_path))


def test_health_exposes_model_unavailability(monkeypatch):
    import app.api as api

    def unavailable():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(api, "load_service", unavailable)
    response = api.app.test_client().get("/health")
    assert response.status_code == 503
    assert response.json["model_available"] is False


def test_tuning_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import src.tune as tuning

    def resolve(path):
        return tmp_path / Path(path)

    def fake_train(config):
        rate = config["train"]["learning_rate"]
        output = resolve("models/transformer")
        output.mkdir(parents=True, exist_ok=True)
        (output / "training.json").write_text(json.dumps({
            "best_validation_f1": .9 if rate == .001 else .6, "run_id": str(rate)}))

    def save(data, path):
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))

    monkeypatch.setattr(tuning, "resolve_project_path", resolve)
    monkeypatch.setattr(tuning, "load_config", lambda path: {"train": {"learning_rate": .001}})
    monkeypatch.setattr(tuning, "train", fake_train)
    monkeypatch.setattr(tuning, "save_json", save)
    tuning.main()
    winner = json.loads(resolve("models/transformer/training.json").read_text())
    assert winner["run_id"] == "0.001"
    assert len(json.loads(resolve("reports/tuning.json").read_text())["trials"]) == 2
