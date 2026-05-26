"""
Streamlit web application for Bangla Sentiment Analysis.

Run with: streamlit run app.py
"""

import sys
import os
import streamlit as st
import pandas as pd

# Ensure `src` is on sys.path so internal imports like `data_preprocessing` work
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from prediction import SentimentPredictor
from data_preprocessing import preprocess_text


# Set page config
st.set_page_config(
    page_title="Bangla Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar information
with st.sidebar:
    st.title("ℹ️ Information")
    st.write("""
    **Bangla Sentiment Analyzer**
    
    This application classifies Bangla text into sentiment categories:
    - 😀 Happy
    - 😢 Sad
    - 😄 Funny
    - ☠️ Toxic
    - 😐 Neutral
    
    Built with Python, scikit-learn, and Streamlit.
    """)
    
    st.divider()
    
    st.subheader("📊 About the Model")
    st.write("""
    - **Algorithm**: Logistic Regression with TF-IDF
    - **Training Data**: ~600 Bangla text samples
    - **Accuracy**: (To be updated after training)
    - **Language**: Bangla
    """)
    
    st.divider()
    
    st.subheader("🔗 Links")
    st.write("[GitHub Repository](https://github.com/aunonno403/Bangla-Sentiment-Analysis)")
    st.write("[Dataset Source](https://www.kaggle.com/datasets/tahmidmir/largesentimentdata)")
    st.divider()
    st.subheader("🧾 Model Status")
    model_path = 'models/best_model.joblib'
    vec_path = 'models/vectorizer.joblib'
    report_path = 'models/training_report_hybrid.txt'

    def _exists_info(p):
        if os.path.exists(p):
            return f"Exists — {os.path.getsize(p):,} bytes"
        return "Missing"

    st.write(f"Model: {model_path} — {_exists_info(model_path)}")
    st.write(f"Vectorizer: {vec_path} — {_exists_info(vec_path)}")
    st.write(f"Label encoder: {'models/label_encoder.joblib'} — {_exists_info('models/label_encoder.joblib')}")

    if os.path.exists(report_path):
        with st.expander("Training report (click to view)"):
            st.text(open(report_path, 'r', encoding='utf-8').read())
    else:
        st.info("No training report found; run training to generate one.")


# Main content
st.title("💬 Bangla Sentiment Analyzer")
st.write("Enter Bangla text below to analyze its sentiment.")

st.divider()

# Input section
if 'input' not in st.session_state:
    st.session_state['input'] = ''

st.text_area(
    "📝 Enter Bangla text:",
    placeholder="আপনার পাঠ্য এখানে লিখুন...",
    height=150,
    key='input'
)

user_input = st.session_state.get('input', '')

# Prediction button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip():
        st.info("⏳ Analyzing...")
        
        try:
            model_path = 'models/best_model.joblib'
            vec_path = 'models/vectorizer.joblib'
            le_path = 'models/label_encoder.joblib'

            if os.path.exists(model_path) and os.path.exists(vec_path):
                predictor = SentimentPredictor(model_path=model_path,
                                               vectorizer_path=vec_path,
                                               label_encoder_path=le_path)
                label, conf = predictor.predict(user_input)
                conf_str = f"{conf*100:.1f}%" if conf is not None else "N/A"

                # detect whether lexicon fallback likely used
                cleaned = preprocess_text(user_input)
                features = predictor.vectorizer.transform([cleaned])
                used_fallback = False
                tokens = cleaned.split()
                if hasattr(features, 'nnz') and features.nnz == 0:
                    used_fallback = True
                else:
                    # check for exact token match in lexicon
                    for t in tokens:
                        for kws in predictor._fallback_lexicon.values():
                            if t in kws:
                                used_fallback = True
                                break
                        if used_fallback:
                            break

                st.success(f"**Sentiment**: {label}\n\n**Confidence**: {conf_str}")
                if used_fallback:
                    st.info("Note: prediction used lexicon fallback (short/rare input)")
            else:
                st.warning('Model artifacts not found in `models/`. Please run training or use packaged release.')
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some Bangla text first.")

st.divider()

# Example section
st.subheader("📚 Try These Examples")

# Ensure session_state for input
if 'input' not in st.session_state:
    st.session_state['input'] = ''

examples = {
    "দারুণ লাগছে, খুব ভালো!": "Happy",
    "আমার মন খারাপ, কাঁদছি": "Sad",
    "তোমার কথায় হেসে ফেললাম": "Funny",
    "এটা খুবই বাজে": "Toxic",
    "এটা ঠিক আছে, সমস্যা নেই": "Neutral",
}

def _set_example(text):
    # Use a callback to set the widget-backed session state value.
    st.session_state['input'] = text

cols = st.columns(1)
for example_text, label in examples.items():
    st.button(example_text, on_click=_set_example, args=(example_text,), use_container_width=True)

st.divider()

# Footer
st.write("""
---
**Phase**: 1 (Development) | **Last Updated**: May 27, 2026
""")
