# src/prediction.py
from typing import Tuple, List, Dict, Optional
import os
import joblib
import numpy as np

from data_preprocessing import preprocess_text


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class SentimentPredictor:
    """
    Wrapper for making sentiment predictions.
    Usage:
      p = SentimentPredictor(model_path='models/best_model.joblib',
                             vectorizer_path='models/vectorizer.joblib',
                             label_encoder_path='models/label_encoder.joblib')
      p.predict('আমি খুব খুশি')
    """

    def __init__(
        self,
        model_path: str,
        vectorizer_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        model_obj=None,
        vectorizer_obj=None,
        label_encoder_obj=None,
    ):
        # Load model
        if model_obj is not None:
            self.model = model_obj
        else:
            self.model = joblib.load(model_path)

        # Load vectorizer
        if vectorizer_obj is not None:
            self.vectorizer = vectorizer_obj
        else:
            if vectorizer_path is None:
                raise ValueError("vectorizer_path required if vectorizer_obj not provided")
            self.vectorizer = joblib.load(vectorizer_path)

        # Load label encoder (optional but recommended)
        if label_encoder_obj is not None:
            self.le = label_encoder_obj
        else:
            if label_encoder_path and os.path.exists(label_encoder_path):
                self.le = joblib.load(label_encoder_path)
            else:
                self.le = None

            # Expanded lexicon for short-text fallback (preprocessed/stemmed forms and phrase fragments)
            # Map label -> set of keyword stems/phrase fragments. Keep entries conservative.
            self._fallback_lexicon = {
                'Happy': {
                    'ভালো', 'খুশি', 'সুখ', 'মজা', 'ভাল', 'ভালো লাগ', 'খুব ভালো', 'খুবভালো'
                },
                'Sad': {
                    'খারাপ', 'দুঃখ', 'দুঃখিত', 'বিষাদ', 'খারাপ লাগ', 'খুব খারাপ', 'দুঃখে'
                },
                'Toxic': {
                    'ঘৃণা', 'নোংরা', 'গালি', 'অশ্লীল', 'বহিষ্কার'
                },
                'Funny': {
                    'রসিক', 'হাস্য', 'মজার', 'হাসি', 'চল', 'মজাই'
                },
                'Neutral': {
                    'যেমন', 'আশা', 'করেছি', 'তেমন', 'মোটামু', 'মোটামুটি', 'এটা', 'ছিল'
                }
            }

    def _get_confidence(self, features) -> Optional[float]:
        # prefer predict_proba
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)
            return float(np.max(probs))
        # fallback to decision_function -> softmax
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            scores = np.atleast_2d(scores)
            probs = _softmax(scores)
            return float(np.max(probs))
        # no probability available
        return None

    def predict(self, text: str) -> Tuple[str, Optional[float]]:
        cleaned = preprocess_text(text)
        # Short-text / zero-feature fallback using lexicon
        tokens = cleaned.split()
        # Function to consult lexicon: check token equality or substring match
        def _consult_lexicon(text_str: str):
            for label, keywords in self._fallback_lexicon.items():
                for kw in keywords:
                    if kw == '':
                        continue
                    # direct token in cleaned
                    if kw in text_str:
                        return label
                    # check token-level equality
                    if any(kw == t for t in tokens):
                        return label
            return None

        # If very short (<=2 tokens) consult lexicon first
        if len(tokens) <= 2:
            lex_label = _consult_lexicon(cleaned)
            if lex_label is not None:
                return lex_label, 0.95

        features = self.vectorizer.transform([cleaned])

        # If vector yields no features (empty), try lexicon again
        try:
            if hasattr(features, 'nnz') and features.nnz == 0:
                lex_label = _consult_lexicon(cleaned)
                if lex_label is not None:
                    return lex_label, 0.95
        except Exception:
            pass

        pred = self.model.predict(features)
        pred_label = pred[0]

        # decode label if label encoder available
        if self.le is not None:
            try:
                label = self.le.inverse_transform([pred_label])[0]
            except Exception:
                label = str(pred_label)
        else:
            label = str(pred_label)

        confidence = self._get_confidence(features)
        return label, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, Optional[float]]]:
        return [self.predict(t) for t in texts]


# convenience function
def make_prediction(text: str, model_path: str, vectorizer_path: str, label_encoder_path: Optional[str] = None) -> Dict:
    pred = SentimentPredictor(model_path=model_path, vectorizer_path=vectorizer_path, label_encoder_path=label_encoder_path)
    label, confidence = pred.predict(text)
    return {"text": text, "sentiment": label, "confidence": confidence}