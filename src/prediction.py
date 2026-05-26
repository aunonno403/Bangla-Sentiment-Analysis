# src/prediction.py
from typing import Tuple, List, Dict, Optional
import os
import joblib
import numpy as np

from data_preprocessing import preprocess_text
import json
from pathlib import Path


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

            # Load lexicon from data/lexicon.json if available; otherwise fall back to small builtin set
            lex_path = Path('data/lexicon.json')
            if lex_path.exists():
                try:
                    with open(lex_path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                    # convert lists to sets for faster membership checks
                    self._fallback_lexicon = {k: set(v) for k, v in raw.items()}
                except Exception:
                    # fallback if file parse fails
                    self._fallback_lexicon = {
                        'Happy': {'ভালো', 'খুশি', 'সুখ'},
                        'Sad': {'খারাপ', 'দুঃখ'},
                        'Toxic': {'ঘৃণা', 'গালি'},
                        'Funny': {'রসিক', 'হাসি'},
                        'Neutral': {'যেমন', 'তেমন'}
                    }
            else:
                self._fallback_lexicon = {
                    'Happy': {'ভালো', 'খুশি', 'সুখ'},
                    'Sad': {'খারাপ', 'দুঃখ'},
                    'Toxic': {'ঘৃণা', 'গালি'},
                    'Funny': {'রসিক', 'হাসি'},
                    'Neutral': {'যেমন', 'তেমন'}
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
        tokens = cleaned.split()

        # Build list of (label, kw) and sort by keyword length (longer first)
        pairs = []
        for label, keywords in self._fallback_lexicon.items():
            for kw in keywords:
                if not kw:
                    continue
                pairs.append((label, kw))
        pairs.sort(key=lambda x: len(x[1]), reverse=True)

        # Handle simple negation patterns: if a negation token appears together
        # with a positive keyword (e.g. 'ভালো নেই'), treat as `Sad`.
        negation_tokens = {'না', 'নেই', 'নয়', 'নেই'}
        positive_kw_set = set(self._fallback_lexicon.get('Happy', set()))
        if any(nt in tokens for nt in negation_tokens):
            if any((pw in tokens) or (pw in cleaned) for pw in positive_kw_set):
                return 'Sad', 0.95

        # Exact token matches should override the model for curated keywords.
        # This avoids short generic words like 'লাগ' winning over specific sad cues like 'কান্ন'.
        for label, kw in pairs:
            if any(kw == t for t in tokens):
                return label, 0.95

        # Short-text / zero-feature fallback using lexicon substring matches
        def _consult_lexicon(text_str: str):
            # Build list of (label, kw) and sort by keyword length (longer first)
            for label, kw in pairs:
                if kw in text_str:
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