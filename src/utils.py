"""
Utilities and helper functions for the Bangla Sentiment Analysis project.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with columns: text, sentiment, source
    """
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=False)


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the dataset."""
    print(f"Dataset Shape: {df.shape}")
    print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nData Info:\n{df.info()}")
    print(f"\nFirst 5 rows:\n{df.head()}")


if __name__ == "__main__":
    print("Utils module loaded successfully.")
