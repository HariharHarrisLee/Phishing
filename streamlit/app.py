import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths to models
IFOREST_MODEL_PATH = "models/iforest_model.joblib"
LR_MODEL_PATH = "models/lr_model.joblib"

# Load models and embedder
@st.cache_resource
def load_models():
    iforest_model = joblib.load(IFOREST_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return iforest_model, lr_model, embedder

iforest_model, lr_model, embedder = load_models()

st.title("📧 Phishing Email Detector")

# User input
email_text = st.text_area("Paste your email content here:")

if st.button("Analyze"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Generate embedding
        embedding = embedder.encode([email_text])

        # Isolation Forest
        anomaly_score = iforest_model.decision_function(embedding)[0]
        iforest_pred = iforest_model.predict(embedding)[0]  # -1 = anomaly (phishing), 1 = normal (legit)

        # Logistic Regression
        lr_pred = lr_model.predict(embedding)[0]
        lr_prob = lr_model.predict_proba(embedding)[0][1]  # prob phishing

        # Results
        st.subheader("🔎 Results")
        st.write(f"**IsolationForest Score:** {anomaly_score:.4f} → {'Phishing' if iforest_pred == -1 else 'Legitimate'}")
        st.write(f"**Logistic Regression:** {'Phishing' if lr_pred == 1 else 'Legitimate'} (Probability: {lr_prob:.2f})")
