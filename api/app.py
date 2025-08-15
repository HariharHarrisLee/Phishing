# api/app.py
from flask import Flask, request, render_template
import os
import re
from joblib import load
from sentence_transformers import SentenceTransformer

# ----------------------
# Flask app
# ----------------------
app = Flask(__name__)

# ----------------------
# Paths and settings
# ----------------------
ROOT = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
IFOREST_PATH = os.path.join(ROOT, "models", "iforest_model.joblib")
LR_PATH = os.path.join(ROOT, "models", "lr_model.joblib")
THRESHOLD = 0.005  # optional for reference; IF uses predict() directly

# ----------------------
# Load models
# ----------------------
iforest_model = load(IFOREST_PATH)
lr_model = load(LR_PATH)

# Load sentence-transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Preprocessing function
# ----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

# ----------------------
# Prediction functions
# ----------------------
def predict_iforest(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    pred = iforest_model.predict(embedding)[0]  # -1 = anomaly/phishing, 1 = legitimate
    score = iforest_model.decision_function(embedding)[0]
    label = "Phishing" if pred == -1 else "Legitimate"
    return label, score

def predict_lr(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])  # LR expects 2D array
    pred = lr_model.predict(embedding)[0]   # 1 = phishing, 0 = legitimate
    label = "Phishing" if pred == 1 else "Legitimate"
    return label

# ----------------------
# Flask routes
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]

        iforest_label, iforest_score = predict_iforest(email_text)
        lr_label = predict_lr(email_text)

        return render_template(
            "result.html",
            email=email_text,
            iforest_label=iforest_label,
            iforest_score=iforest_score,
            lr_label=lr_label
        )
    return render_template("index.html")

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
