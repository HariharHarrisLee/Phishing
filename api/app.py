from flask import Flask, request, render_template
import os
import re
from joblib import load
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
IFOREST_PATH = os.path.join(ROOT, "models", "iforest_model.joblib")
LR_PATH = os.path.join(ROOT, "models", "lr_model.joblib")
THRESHOLD = 0.005  

#Models
iforest_model = load(IFOREST_PATH)
lr_model = load(LR_PATH)

#Sentence-transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

#Prediction
def predict_iforest(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    pred = iforest_model.predict(embedding)[0] 
    score = iforest_model.decision_function(embedding)[0]
    label = "Phishing" if pred == -1 else "Legitimate"
    return label, score

def predict_lr(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])  
    pred = lr_model.predict(embedding)[0]   
    label = "Phishing" if pred == 1 else "Legitimate"
    return label

#Flask
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
