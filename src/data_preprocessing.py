"""
Data preprocessing for Bangla text.

Handles:
- Text normalization and cleaning
- Tokenization
- Stemming using bengali-stemmer
- Feature extraction (TF-IDF)
"""

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bengali_stemmer.stemmer import BengaliStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple


# Initialize Bengali stemmer
stemmer = BengaliStemmer()


def normalize_bangla_text(text: str) -> str:
    """
    Normalize Bangla text: handle unicode, remove diacritics, etc.
    
    Args:
        text: Raw Bangla text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep Bangla text and basic punctuation
    # Keep Bangla Unicode range (0x0980-0x09FF) and basic Latin
    text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s\.,!?\'\"-]', '', text)
    
    return text


def tokenize_bangla(text: str) -> List[str]:
    """
    Tokenize Bangla text into words.
    
    Args:
        text: Normalized Bangla text
        
    Returns:
        List of tokens
    """
    try:
        tokens = word_tokenize(text, language='bengali')
    except:
        # Fallback: simple split
        tokens = text.split()
    
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove Bengali stopwords.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        Filtered tokens
    """
    try:
        bengali_stopwords = set(stopwords.words('bengali'))
    except:
        bengali_stopwords = set()
    
    return [token for token in tokens if token not in bengali_stopwords]


def stem_bangla(tokens: List[str]) -> List[str]:
    """
    Apply Bengali stemming to tokens.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        Stemmed tokens
    """
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """
    Complete preprocessing pipeline: normalize → tokenize → stem.
    
    Args:
        text: Raw Bangla text
        
    Returns:
        Preprocessed text (space-separated tokens)
    """
    # Normalize
    normalized = normalize_bangla_text(text)
    
    # Tokenize
    tokens = tokenize_bangla(normalized)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Stem
    tokens = stem_bangla(tokens)
    
    # Rejoin
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to entire DataFrame.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        DataFrame with 'cleaned_text' column added
    """
    df_copy = df.copy()
    df_copy['cleaned_text'] = df_copy['text'].apply(preprocess_text)
    
    return df_copy


def create_tfidf_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None, max_features: int = 5000):
    """
    Create TF-IDF features from preprocessed text.
    
    Args:
        df_train: Training DataFrame with 'cleaned_text' column
        df_test: Test DataFrame (optional)
        max_features: Maximum number of features
        
    Returns:
        Tuple of (vectorizer, train_features, test_features or None)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.95, min_df=2)
    
    X_train = vectorizer.fit_transform(df_train['cleaned_text'])
    X_test = vectorizer.transform(df_test['cleaned_text']) if df_test is not None else None
    
    return vectorizer, X_train, X_test


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "এটি একটি দুর্দান্ত পণ্য এবং আমি খুবই সন্তুষ্ট"
    print(f"Original: {sample_text}")
    print(f"Preprocessed: {preprocess_text(sample_text)}")
