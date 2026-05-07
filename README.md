# Bangla Sentiment Analysis

A machine learning project to classify Bangla text into sentiment categories (Positive, Negative, Neutral). Built with scikit-learn and deployed as an interactive Streamlit web application.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline for sentiment analysis:
- **Data**: Combination of public Bangla sentiment datasets + custom-labeled examples
- **Model**: Logistic Regression with TF-IDF vectorization (scikit-learn)
- **Frontend**: Interactive Streamlit application
- **Deployment**: Streamlit Community Cloud (free)

**Status**: 🚧 Under Development (Phase 1: Data Collection)

---

## 📋 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/textbangla-sentiment-analyzer.git
cd textbangla-sentiment-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
textbangla-sentiment-analyzer/
├── data/
│   ├── raw/                    # Original datasets (SemEval, custom labels)
│   └── processed/              # Cleaned, tokenized data ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Model development & evaluation
├── src/
│   ├── data_preprocessing.py  # Bangla text cleaning, tokenization, stemming
│   ├── model_training.py      # Model training pipeline
│   ├── prediction.py          # Inference utilities
│   └── utils.py               # Helper functions
├── app.py                      # Streamlit frontend
├── model.pkl                   # Trained model artifact
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore
```

---

## 🚀 Development Roadmap

- **Phase 1** (Days 1-3): Data collection & setup *(Current)*
- **Phase 2** (Days 4-7): EDA & preprocessing
- **Phase 3** (Days 8-18): Model training & evaluation
- **Phase 4** (Days 19-25): Streamlit frontend & deployment
- **Phase 5** (Days 26-30): Polish & documentation

---

## 📊 Dataset

**Composition**:
- Primary: SemEval-2024 Bangla Sentiment (public benchmark) — ~500 examples
- Supplementary: Custom-labeled Bangla text — ~100-200 examples

**Labels**: Positive, Negative, Neutral

**Sources**:
- Public Benchmark: https://github.com/SemEval-2024/SemEval-2024-Task5-Bangla
- Custom Data: YouTube comments, news site comments, social media

See [data/README.md](data/README.md) for detailed dataset documentation.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **NLP** | NLTK, bengali-stemmer |
| **ML** | scikit-learn |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## 📈 Results

*(To be updated after Phase 3)*

- Model Accuracy: TBD
- Precision / Recall / F1: TBD
- Inference Time: TBD

---

## 🤝 Contributing

This is a personal learning project. For questions or suggestions, feel free to open an issue.

---

## 📝 License

MIT License — See LICENSE file for details

---

## 👤 Author

- **Your Name** — [GitHub](https://github.com/yourusername)

---

## 🔗 Useful Resources

- [NLTK Bangla Support](https://www.nltk.org/)
- [bengali-stemmer PyPI](https://pypi.org/project/bengali-stemmer/)
- [bnlp-toolkit](https://github.com/sagorbrur/bnlp)
- [Streamlit Docs](https://docs.streamlit.io/)
- [scikit-learn Text Classification](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

**Last Updated**: May 7, 2026 | **Phase**: 1 (Setup & Data Collection)
