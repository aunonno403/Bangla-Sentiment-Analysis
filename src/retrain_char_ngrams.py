"""Retrain model using character n-grams (char_wb) to improve short/rare-token coverage.

Usage: python src/retrain_char_ngrams.py
"""
import sys
sys.path.insert(0, 'src')
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import data_preprocessing as dp


def load_data():
    train_path = 'data/processed/train_cleaned.csv'
    test_path = 'data/processed/test_cleaned.csv'
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
    else:
        df_train = pd.read_csv('data/processed/train.csv')
        df_train = dp.preprocess_dataframe(df_train, text_column='Comment')
        df_train['cleaned_text'] = df_train['cleaned_text'].fillna('')

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
    else:
        df_test = None

    return df_train, df_test


def main():
    os.makedirs('models', exist_ok=True)

    df_train, df_test = load_data()

    # Ensure cleaned_text column
    if 'cleaned_text' not in df_train.columns:
        df_train = dp.preprocess_dataframe(df_train, text_column='Comment')
    df_train['cleaned_text'] = df_train['cleaned_text'].fillna('')

    if df_test is not None:
        if 'cleaned_text' not in df_test.columns:
            df_test = dp.preprocess_dataframe(df_test, text_column='Comment')
        df_test['cleaned_text'] = df_test['cleaned_text'].fillna('')

    # Label encoding
    le = LabelEncoder()
    le.fit(df_train['Sentiment'].astype(str))
    y = le.transform(df_train['Sentiment'].astype(str))

    # Build TF-IDF with character n-grams
    # char_wb with ngram_range (3,5) captures substrings and will help unseen rare tokens
    max_features = 20000
    # Use module-level identity_preprocessor from data_preprocessing for picklability
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer='char_wb',
        ngram_range=(3, 5),
        lowercase=False,
        preprocessor=dp.identity_preprocessor
    )

    X_train_full = vectorizer.fit_transform(df_train['cleaned_text'])
    X_test_vec = vectorizer.transform(df_test['cleaned_text']) if df_test is not None else None

    # If no external test, split train
    if df_test is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X_train_full, y
        y_val = le.transform(df_test['Sentiment'].astype(str))
        X_val = X_test_vec

    # Keep all features (selector k = n_features)
    k = X_train.shape[1]
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)

    # Train models
    models = {
        'logreg': LogisticRegression(max_iter=2000, C=1.0),
        'linear_svc': LinearSVC(max_iter=2000),
        'mnb': MultinomialNB()
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_val_sel)
        f1_macro = f1_score(y_val, preds, average='macro')
        results[name] = (clf, f1_macro)
        print(f"Trained {name}, macro-f1={f1_macro:.4f}")

    # Pick best
    best_name, (best_model, best_score) = max(results.items(), key=lambda kv: kv[1][1])
    print('Best model:', best_name, best_score)

    # Save artifacts (distinct filenames to avoid overwriting previous experiments)
    joblib.dump(vectorizer, 'models/vectorizer_char_ngrams.joblib')
    joblib.dump(selector, 'models/selector_char_ngrams.joblib')
    joblib.dump(best_model, 'models/best_model_char_ngrams.joblib')
    joblib.dump(le, 'models/label_encoder_char_ngrams.joblib')

    # Save training report
    preds_val = best_model.predict(X_val_sel)
    preds_val_dec = le.inverse_transform(preds_val)
    y_val_dec = le.inverse_transform(y_val)
    report = classification_report(y_val_dec, preds_val_dec, digits=4, zero_division=0)
    with open('models/training_report_char_ngrams.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print('Saved char-ngrams artifacts to models/ and training_report_char_ngrams.txt')


if __name__ == '__main__':
    main()
