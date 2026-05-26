import sys
sys.path.insert(0, 'src')
import json
import pandas as pd
import data_preprocessing as dp

lex = json.load(open('data/lexicon.json', 'r', encoding='utf-8'))
lex_items = set(x for items in lex.values() for x in items)
print('lex items count:', len(lex_items))

df = pd.read_csv('data/processed/train_cleaned.csv')
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

all_tokens = []
for t in df['cleaned_text']:
    toks = dp.whitespace_tokenizer(t)
    all_tokens.extend([tok for tok in toks if tok])

print('total tokens:', len(all_tokens))
print('unique tokens:', len(set(all_tokens)))

sample = list(set(all_tokens))[:50]
for tok in sample[:30]:
    matched = any((lex in tok or tok in lex) for lex in lex_items)
    if not matched:
        print('UNMATCHED', tok)
    else:
        print('MATCHED', tok)

print('done')
