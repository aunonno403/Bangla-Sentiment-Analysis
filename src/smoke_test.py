from prediction import SentimentPredictor

predictor = SentimentPredictor(model_path='models/best_model.joblib',
							   vectorizer_path='models/vectorizer.joblib',
							   label_encoder_path='models/label_encoder.joblib')

texts = ['পণ্যটি ভালো ']
for t in texts:
	label, conf = predictor.predict(t)
	print(t, '->', label, f'(conf={conf})')