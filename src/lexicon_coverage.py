"""Scan training data for tokens not covered by the lexicon and produce suggestions.

Usage: python src/lexicon_coverage.py
Writes: data/lexicon_suggestions.json
"""
import sys
sys.path.insert(0, 'src')

import os
import json
from collections import defaultdict, Counter
from typing import Dict

import pandas as pd

import data_preprocessing as dp


def load_train():
    train_path = 'data/processed/train_cleaned.csv'
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
    else:
        df = pd.read_csv('data/processed/train.csv')
        df = dp.preprocess_dataframe(df, text_column='Comment')
    if 'cleaned_text' not in df.columns:
        df = dp.preprocess_dataframe(df, text_column='Comment')
    df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
    return df


def load_lexicon() -> Dict[str, list]:
    path = 'data/lexicon.json'
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def token_matches_lexicon(token: str, lexicon_set: set) -> bool:
    # Conservative matching: check direct containment or substring
    if token in lexicon_set:
        return True
    for lex in lexicon_set:
        if lex in token or token in lex:
            return True
    return False


def main():
    os.makedirs('data', exist_ok=True)
    df = load_train()
    lex = load_lexicon()
    lex_items = set()
    for label, items in lex.items():
        for it in items:
            lex_items.add(it)

    print(f'Loaded lexicon with {len(lex_items)} items')

    token_counts = Counter()
    token_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_tokens_seen = 0
    for _, row in df.iterrows():
        text = row.get('cleaned_text', '') or ''
        label = str(row.get('Sentiment', ''))
        tokens = dp.whitespace_tokenizer(text)
        seen = set()
        for t in tokens:
            if not t:
                continue
            total_tokens_seen += 1
            # skip tokens that are already covered by lexicon
            if token_matches_lexicon(t, lex_items):
                continue
            token_counts[t] += 1
            if t not in seen:
                token_label_counts[t][label] += 1
                seen.add(t)

    print(f'Total tokens seen: {total_tokens_seen}; unique uncovered tokens: {len(token_counts)}')

    # prepare suggestions: top tokens by count
    suggestions = []
    for token, cnt in token_counts.most_common(200):
        label_counts = token_label_counts[token]
        # choose suggested label as the label with highest co-occurrence
        if label_counts:
            suggested_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
        else:
            suggested_label = 'Neutral'
        suggestions.append({
            'token': token,
            'count': int(cnt),
            'suggested_label': suggested_label,
            'label_counts': dict(label_counts),
        })

    out_path = 'data/lexicon_suggestions.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(suggestions)} suggestions to {out_path}')


if __name__ == '__main__':
    main()
