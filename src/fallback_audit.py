from prediction import SentimentPredictor
from data_preprocessing import preprocess_text
import os

predictor = SentimentPredictor(model_path='models/best_model.joblib',
                               vectorizer_path='models/vectorizer.joblib',
                               label_encoder_path='models/label_encoder.joblib')

samples = [
    'আজ আমার মন ভালো নেই।',
    'আমি খুব খুশি',
    'আজকে দিনটা ভালো গেল, অনেক আনন্দ',
    'খারাপ লাগছে আজ',
    'তুমি ঠিক আছো?',
    'আমি বই পড়ছি এবং কফি খাচ্ছি',
    'এটা দুর্দান্ত, আমি পছন্দ করলাম',
    'আমি হাসছি',
    'বুঝতে পারছি না',
    'এইটা ঠিক আছে'
]

vocab_size = None
try:
    vocab_size = len(predictor.vectorizer.get_feature_names_out())
except Exception:
    try:
        vocab_size = len(predictor.vectorizer.vocabulary_)
    except Exception:
        vocab_size = 'unknown'

print('Vectorizer vocab size:', vocab_size)

total = 0
fallback_count = 0
nnz_counts = []

for s in samples:
    total += 1
    cleaned = preprocess_text(s)
    features = predictor.vectorizer.transform([cleaned])
    nnz = features.nnz if hasattr(features, 'nnz') else None
    nnz_counts.append(nnz)

    # determine fallback: short or nnz==0 or lexicon substring match
    tokens = cleaned.split()
    used_fallback = False
    # short text and lexicon
    if len(tokens) <= 2:
        for label, kws in predictor._fallback_lexicon.items():
            for kw in kws:
                if kw and kw in cleaned:
                    used_fallback = True
                    break
            if used_fallback:
                break
    if nnz == 0:
        for label, kws in predictor._fallback_lexicon.items():
            for kw in kws:
                if kw and kw in cleaned:
                    used_fallback = True
                    break
            if used_fallback:
                break

    if used_fallback:
        fallback_count += 1

    label, conf = predictor.predict(s)
    print(f"Sample: {s!r}\n Cleaned: {cleaned!r}  nnz={nnz}  fallback={used_fallback} => {label} (conf={conf})\n")

print('Summary:')
print(' total samples:', total)
print(' fallback_count:', fallback_count)
print(' nnz_counts:', nnz_counts)
