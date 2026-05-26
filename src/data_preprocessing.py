"""
Data preprocessing for Bangla text.

Handles:
- Text normalization and cleaning
- Tokenization
- Stemming using bangla-stemmer
- Feature extraction (TF-IDF)
"""

import re
import logging
import pandas as pd
from bnlp import BasicTokenizer
from nltk.corpus import stopwords
from bangla_stemmer.stemmer import stemmer as bangla_stemmer_module
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# Initialize bnlp tokenizer and stemmer
tokenizer = BasicTokenizer()
stemmer = bangla_stemmer_module.BanglaStemmer()


# Module-level helpers so objects referencing them are pickleable
def whitespace_tokenizer(text: str):
    return text.split()


def identity_preprocessor(x: str):
    return x


def normalize_bangla_text(text: str) -> str:
    """
    Normalize Bangla text: handle unicode, remove diacritics, etc.
    
    Args:
        text: Raw Bangla text
        
    Returns:
        Normalized text
    """
    # Handle NaN/None/non-string inputs
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep Bangla text and basic punctuation
    # Keep Bangla Unicode range (0x0980-0x09FF) and basic Latin
    text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s\.,!?\'\"-]', '', text)
    
    return text


# Conservative mapping of common variant tokens to normalized/stem-like forms
# Use this to capture variants that the stemmer or tokenizer may not normalize.
# Update when new variants are observed.
NORMALIZATION_MAP = {
    'মোটামুটি': 'মোটামু',
    'মোটামুটিা': 'মোটামু',
    'আজকের': 'আজক',
    'আজকে': 'আজ',
}


def tokenize_bangla(text: str) -> List[str]:
    """
    Tokenize Bangla text into words using bnlp BasicTokenizer.
    
    Args:
        text: Normalized Bangla text
        
    Returns:
        List of tokens
    """
    try:
        tokens = tokenizer(text)
    except Exception as e:
        logger.warning(f"Error during tokenization: {e}. Using fallback split.")
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
    except LookupError:
        logger.debug("NLTK stopwords resource not found; proceeding without stopword removal.")
        bengali_stopwords = set()
    except Exception as e:
        logger.warning(f"Error loading Bengali stopwords: {e}. Proceeding without stopword removal.")
        bengali_stopwords = set()
    
    # Add project-specific domain tokens that are not useful for sentiment
    # (common but sentiment-neutral tokens). Update this set as needed.
    CUSTOM_DOMAIN_STOPWORDS = {
        'ভিডি',  # video
        'আজক',  # today (colloquial/typo variants)
        'আজ',   # today
        'মন',   # mind/feeling (often neutral context words)
        # keep 'খুব' (intensifier) -- don't remove as stopword
        'সবক',
        'ভিডিও',
        'সব',
        'অভিজ্ঞতা',
        'আমি',
        'আম',  # prefix-like token from preprocessing
    }
    # Additional neutral/common tokens identified from feature inspection
    # (added to reduce their influence on class coefficients)
    ADDITIONAL_NEUTRAL_TOKENS = {
        'পেট',
        'ফাট',
        'দিন',
        'নষ্ট',
        'সময়',
        'ভয়ঙ্কর',
        'আছি',
        'সবকিছু',
        # keep 'লাগ' (important sentiment cue)
    }

    bengali_stopwords = bengali_stopwords | ADDITIONAL_NEUTRAL_TOKENS

    bengali_stopwords = bengali_stopwords | CUSTOM_DOMAIN_STOPWORDS

    # Ensure common negation tokens are NOT removed; they are important for sentiment polarity
    NEGATION_TOKENS = {'না', 'নেই', 'নয়'}
    bengali_stopwords = bengali_stopwords - NEGATION_TOKENS

    return [token for token in tokens if token not in bengali_stopwords]


def stem_bangla(tokens: List[str]) -> List[str]:
    """
    Apply Bengali stemming to tokens.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        Stemmed tokens
    """
    stemmed = []
    for token in tokens:
        try:
            stemmed.append(stemmer.stem(token))
        except Exception as e:
            logger.debug(f"Error stemming token '{token}': {e}. Keeping original.")
            stemmed.append(token)
    return stemmed


def preprocess_text(text: str) -> str:
    """
    Complete preprocessing pipeline: normalize → tokenize → stem.
    
    Args:
        text: Raw Bangla text
        
    Returns:
        Preprocessed text (space-separated tokens)
    """
    # Handle NaN/None inputs early
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    # Normalize
    normalized = normalize_bangla_text(text)
    
    # Tokenize
    tokens = tokenize_bangla(normalized)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Apply conservative normalization mappings (token -> mapped token)
    if tokens:
        tokens = [NORMALIZATION_MAP.get(t, t) for t in tokens]
    
    # Stem
    tokens = stem_bangla(tokens)
    
    # Rejoin
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Apply preprocessing to entire DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the text column to preprocess (default: 'text')
        
    Returns:
        DataFrame with 'cleaned_text' column added
    """
    df_copy = df.copy()
    
    # Ensure text column exists
    if text_column not in df_copy.columns:
        logger.warning(f"Column '{text_column}' not found in DataFrame. Available columns: {df_copy.columns.tolist()}")
        return df_copy
    
    # Apply preprocessing with NaN-safe handling
    df_copy['cleaned_text'] = df_copy[text_column].apply(
        lambda x: preprocess_text(x) if pd.notna(x) else ""
    )
    
    return df_copy


def create_tfidf_features(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None, max_features: int = 5000, ngram_range=(1, 1)):
    """
    Create TF-IDF features from preprocessed text.
    
    Args:
        df_train: Training DataFrame with 'cleaned_text' column
        df_test: Test DataFrame (optional)
        max_features: Maximum number of features
        
    Returns:
        Tuple of (vectorizer, train_features, test_features or None)
    """
    # Use min_df=1 to include tokens that may appear infrequently,
    # which is helpful for small but meaningful Bangla tokens.
    # Use whitespace tokenization because text is already preprocessed (tokens separated by spaces)
    # and set lowercase=False to preserve preprocessed casing (already lowered in normalize)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=0.95,
        min_df=1,
        tokenizer=whitespace_tokenizer,
        preprocessor=identity_preprocessor,
        lowercase=False,
        ngram_range=ngram_range,
    )

    # Ensure cleaned_text columns are string and contain no NaN values
    df_train = df_train.copy()
    df_train['cleaned_text'] = df_train['cleaned_text'].fillna('').astype(str)
    if df_test is not None:
        df_test = df_test.copy()
        df_test['cleaned_text'] = df_test['cleaned_text'].fillna('').astype(str)

    X_train = vectorizer.fit_transform(df_train['cleaned_text'].tolist())
    X_test = vectorizer.transform(df_test['cleaned_text'].tolist()) if df_test is not None else None
    
    return vectorizer, X_train, X_test


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "এটি একটি দুর্দান্ত পণ্য এবং আমি খুবই সন্তুষ্ট"
    print(f"Original: {sample_text}")
    print(f"Preprocessed: {preprocess_text(sample_text)}")
