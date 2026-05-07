"""
Prediction pipeline for trained sentiment model.

Handles:
- Loading trained model
- Preprocessing input text
- Making predictions
- Displaying results with confidence
"""

from src.data_preprocessing import preprocess_text
from src.model_training import load_model
from typing import Tuple, Dict


class SentimentPredictor:
    """Wrapper for making sentiment predictions."""
    
    def __init__(self, model_path: str, vectorizer):
        """
        Initialize predictor with trained model and vectorizer.
        
        Args:
            model_path: Path to saved model (pkl file)
            vectorizer: TF-IDF vectorizer (must be fitted)
        """
        self.model = load_model(model_path)
        self.vectorizer = vectorizer
        self.label_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment of input text.
        
        Args:
            text: Raw Bangla text
            
        Returns:
            Tuple of (sentiment_label, confidence)
        """
        # Preprocess
        cleaned = preprocess_text(text)
        
        # Vectorize
        features = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features).max()
        
        label = self.label_map.get(prediction, 'Unknown')
        
        return label, confidence
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of Bangla texts
            
        Returns:
            List of (sentiment_label, confidence) tuples
        """
        return [self.predict(text) for text in texts]


def make_prediction(text: str, model_path: str, vectorizer) -> Dict:
    """
    Standalone prediction function.
    
    Args:
        text: Input Bangla text
        model_path: Path to trained model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Dictionary with prediction and confidence
    """
    predictor = SentimentPredictor(model_path, vectorizer)
    label, confidence = predictor.predict(text)
    
    return {
        'text': text,
        'sentiment': label,
        'confidence': confidence
    }


if __name__ == "__main__":
    print("Prediction module loaded successfully.")
