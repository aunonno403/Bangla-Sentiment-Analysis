from data_preprocessing import normalize_bangla_text, tokenize_bangla, remove_stopwords, stem_bangla

text = 'আজ আমার মন ভালো নেই।'

print('Original:', text)
norm = normalize_bangla_text(text)
print('Normalized:', repr(norm))
tokens = tokenize_bangla(norm)
print('Tokenized:', tokens)
filtered = remove_stopwords(tokens)
print('After stopword removal:', filtered)
stemmed = stem_bangla(filtered)
print('Stemmed:', stemmed)
