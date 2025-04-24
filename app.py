import joblib
import streamlit as st
import pandas as pd
from utils.vectorization import UnifiedTextVectorizer
from utils.preprocessing import TextPreprocessor

# Load your trained pipeline
pipe = joblib.load('production_model/finalized_model.pkl')

# App layout
st.set_page_config(page_title="🧠 Hate Speech Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: crimson;'>💬 Hate Speech Detection App</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Text input
st.markdown("### 📝 Enter a tweet or sentence below:")
user_input = st.text_area("Write something hateful or not...", placeholder="e.g., I can't stand this!", height=150)

# Prediction Button
if st.button("🚀 Predict", use_container_width=True):
    if user_input.strip() != "":
        # Wrap input in DataFrame
        text_df = pd.DataFrame({'tweet': [user_input]})
        
        # Make prediction
        prediction = pipe.predict(text_df)
        prediction_proba = pipe.predict_proba(text_df)

        # Process result
        label = "🟥 Negative (Hate Speech)" if prediction[0] == 1 else "🟩 Positive (Non-Hate)"
        confidence = prediction_proba[0][prediction[0]] * 100

        # Display Results
        st.markdown("### 🔍 Prediction Result:")
        st.success(label)
        st.info(f"📊 Confidence Score: **{confidence:.2f}%**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ali Ayman</p>", unsafe_allow_html=True)
