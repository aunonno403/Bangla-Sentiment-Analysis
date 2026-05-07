"""
Streamlit web application for Bangla Sentiment Analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from src.prediction import SentimentPredictor
from src.model_training import load_model
import joblib
import os


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
    - 😊 Positive
    - 😐 Neutral
    - 😞 Negative
    
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
    st.write("[GitHub Repository](#)")
    st.write("[Dataset Source](#)")


# Main content
st.title("💬 Bangla Sentiment Analyzer")
st.write("Enter Bangla text below to analyze its sentiment.")

st.divider()

# Input section
user_input = st.text_area(
    "📝 Enter Bangla text:",
    placeholder="আপনার পাঠ্য এখানে লিখুন...",
    height=150
)

# Prediction button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip():
        st.info("⏳ Analyzing...")
        
        try:
            # TODO: Load actual model when ready
            # For now, show placeholder
            st.success("""
            **Sentiment**: Positive 😊
            
            **Confidence**: 92.3%
            
            *(Placeholder - actual model will be used after Phase 3)*
            """)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some Bangla text first.")

st.divider()

# Example section
st.subheader("📚 Try These Examples")

examples = {
    "এটি একটি দুর্দান্ত পণ্য": "positive",
    "আমি খুবই হতাশ": "negative",
    "আজ সুন্দর দিন": "neutral"
}

for example_text, label in examples.items():
    if st.button(example_text, use_container_width=True):
        st.session_state.example = example_text
        st.rerun()

st.divider()

# Footer
st.write("""
---
**Phase**: 1 (Development) | **Last Updated**: May 7, 2026
""")
