import pandas as pd

from src.data_loader import clean_text, encode_label, preprocess_reviews


def test_clean_text_normalizes_review_text():
    assert clean_text("<b>Amazing!!!</b> Movie") == "amazing movie"


def test_encode_label_maps_sentiments():
    assert encode_label("positive") == 1
    assert encode_label("negative") == 0


def test_preprocess_reviews_returns_clean_encoded_dataframe():
    df = pd.DataFrame(
        {
            "review": ["Great movie!", "Bad movie."],
            "sentiment": ["positive", "negative"],
        }
    )
    processed = preprocess_reviews(df)
    assert processed["review"].tolist() == ["great movie", "bad movie"]
    assert processed["sentiment"].tolist() == [1, 0]
