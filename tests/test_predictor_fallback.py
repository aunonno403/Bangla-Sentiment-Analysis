import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from prediction import SentimentPredictor


def test_kandte_maps_to_sad():
    predictor = SentimentPredictor(
        model_path='models/best_model.joblib',
        vectorizer_path='models/vectorizer.joblib',
        label_encoder_path='models/label_encoder.joblib',
    )

    label, confidence = predictor.predict('আমি কাঁদতে চাই')
    assert label == 'Sad'
    assert confidence is not None


def test_kanna_maps_to_sad():
    predictor = SentimentPredictor(
        model_path='models/best_model.joblib',
        vectorizer_path='models/vectorizer.joblib',
        label_encoder_path='models/label_encoder.joblib',
    )

    label, confidence = predictor.predict('মনে অনেক কান্না হচ্ছে')
    assert label == 'Sad'
    assert confidence is not None


def test_smoke_phrase_maps_to_sad():
    predictor = SentimentPredictor(
        model_path='models/best_model.joblib',
        vectorizer_path='models/vectorizer.joblib',
        label_encoder_path='models/label_encoder.joblib',
    )

    label, confidence = predictor.predict('আমার কান্নার মত লাগছে')
    assert label == 'Sad'
    assert confidence is not None
