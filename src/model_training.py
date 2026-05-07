"""
Model training and evaluation for Bangla sentiment analysis.

Includes:
- Model training with hyperparameter tuning
- Evaluation metrics (precision, recall, F1, confusion matrix)
- Model persistence
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Tuple, Dict, Any


def train_logistic_regression(X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
    """
    Train Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train: Training features (sparse matrix from TF-IDF)
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200, 500]
    }
    
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    results = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_
    }
    
    if X_test is not None and y_test is not None:
        y_pred = best_model.predict(X_test)
        results.update({
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        })
    
    return results


def train_svm(X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
    """Train SVM model."""
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    results = {'model': model}
    
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        results.update({
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    
    return results


def train_naive_bayes(X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
    """Train Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    results = {'model': model}
    
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        results.update({
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    
    return results


def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def print_evaluation_metrics(results: Dict[str, Any]) -> None:
    """Print evaluation metrics nicely."""
    if 'accuracy' in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        if 'classification_report' in results:
            print(f"\nClassification Report:\n{results['classification_report']}")


if __name__ == "__main__":
    print("Model training module loaded successfully.")
