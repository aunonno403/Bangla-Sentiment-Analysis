from prediction import SentimentPredictor
from data_preprocessing import preprocess_text

text = 'আজ আমার মন ভালো নেই।'

predictor = SentimentPredictor(model_path='models/best_model.joblib',
                               vectorizer_path='models/vectorizer.joblib',
                               label_encoder_path='models/label_encoder.joblib')

clean = preprocess_text(text)
tokens = clean.split()

print('Original:', text)
print('Cleaned :', repr(clean))
print('Tokens  :', tokens)

print('\nFallback lexicon (Happy):', predictor._fallback_lexicon.get('Happy'))
print('Fallback lexicon (Sad)  :', predictor._fallback_lexicon.get('Sad'))

features = predictor.vectorizer.transform([clean])
print('\nVector nnz:', features.nnz)

try:
    names = predictor.vectorizer.get_feature_names_out()
    nz = features.nonzero()[1]
    print('Nonzero features (sample 20):')
    for i in nz[:20]:
        print(' -', names[i])
except Exception as e:
    print('Could not list feature names:', e)

# Lexicon substring matches
matches = []
for label, kws in predictor._fallback_lexicon.items():
    for kw in kws:
        if kw and kw in clean:
            matches.append((label, kw))
print('\nLexicon substring matches:', matches)

pred_label, conf = predictor.predict(text)
print('\nPredict output:', pred_label, conf)

try:
    # raw model prob
    probs = predictor.model.predict_proba(features)
    classes = None
    if predictor.le is not None:
        try:
            classes = predictor.le.inverse_transform(predictor.model.classes_)
        except Exception:
            classes = predictor.model.classes_
    else:
        classes = predictor.model.classes_
    print('\nModel classes:', classes)
    print('Probs:')
    for c, p in zip(classes, probs[0]):
        print(' -', c, p)
except Exception as e:
    print('Could not get predict_proba:', e)

print('\nDone')
