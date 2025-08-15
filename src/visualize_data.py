# src/visualization/visualize_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import load
from sentence_transformers import SentenceTransformer
import os

# Paths
root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
processed_file = os.path.join(root, "data", "processed", "phishing_email_processed.csv")
emb_file = os.path.join(root, "data", "processed", "embeddings.npy")
model_path = os.path.join(root, "models", "iforest_model.joblib")

# Load processed dataset
df = pd.read_csv(processed_file)

# -----------------------------
# 1. Class distribution
# -----------------------------
plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['Legitimate', 'Phishing'], rotation=0)
plt.title("Class Distribution")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. PCA of embeddings
# -----------------------------
if os.path.exists(emb_file):
    print("Loading cached embeddings for legitimate emails...")
    embeddings = np.load(emb_file)
else:
    print("Generating embeddings for legitimate emails...")
    legit_texts = df[df['label'] == 0]['text_clean'].tolist()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(legit_texts, batch_size=64, show_progress_bar=True)
    np.save(emb_file, embeddings)
    print(f"Embeddings saved to: {emb_file}")

# Reduce dimensionality
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=5, alpha=0.5, color='blue')
plt.title("PCA of Legitimate Email Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Histogram of anomaly scores
# -----------------------------
iforest_model = load(model_path)
scores = iforest_model.decision_function(embeddings)

plt.figure(figsize=(8,6))
plt.hist(scores, bins=50, color='purple', alpha=0.7)
plt.title("IsolationForest Anomaly Scores for Legitimate Emails")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
