from data_preprocessing import preprocess_text
from prediction import SentimentPredictor

predictor = SentimentPredictor(model_path='models/best_model.joblib',
                               vectorizer_path='models/vectorizer.joblib',
                               label_encoder_path='models/label_encoder.joblib')

texts = ['যেমনটা আশা করেছিলাম তেমনটা ছিল না', 'খুব ভালো লাগলো', 'এটা মোটামুটি ছিল','খুব খারাপ লাগলো']

print('=== Samples and Predictions (via SentimentPredictor) ===')
for i, t in enumerate(texts):
    cleaned = preprocess_text(t)
    label, conf = predictor.predict(t)
    row = predictor.vectorizer.transform([cleaned]).getrow(0)
    nz = row.nonzero()[1]
    vals = row.data
    print(f'Index {i}:')
    print('Original:', t)
    print('Cleaned tokens:', cleaned.split() if cleaned else [])
    print('Predicted label:', label, f'(conf={conf})')
    print('TF-IDF nnz count:', row.nnz)
    if row.nnz>0:
        print('TF-IDF features (index:value):', ', '.join([f'{int(idx)}:{float(v):.6f}' for idx, v in zip(nz, vals)]))
    print()

failing = [i for i in range(len(texts)) if predictor.vectorizer.transform([preprocess_text(texts[i])]).getrow(0).nnz==0]
print('Failing rows (no TF-IDF features):', failing)
if failing:
    vocab = getattr(predictor.vectorizer, 'vocabulary_', {})
    for i in failing:
        print(f'--- Details for failing row {i} ---')
        print('Original:', texts[i])
        tokens = preprocess_text(texts[i]).split()
        token_indices = [(token, vocab.get(token)) for token in tokens]
        print('Cleaned tokens list:', tokens)
        print('Token -> index in vocabulary:', token_indices)
        print()
