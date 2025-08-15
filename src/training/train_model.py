import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from joblib import dump
import numpy as np

# Paths
processed_file = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing\data\processed\phishing_email_processed.csv"
emb_file = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing\data\processed\embeddings.npy"
model_path = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing\models\iforest_model.joblib"

# Load processed dataset
df = pd.read_csv(processed_file)

# Use only legitimate emails for training
legit_df = df[df['label'] == 0]
texts = legit_df['text_clean'].tolist()

# Load sentence-transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate or load embeddings
if os.path.exists(emb_file):
    print("Loading cached embeddings for legitimate emails...")
    embeddings = np.load(emb_file)
else:
    print("Generating embeddings for legitimate emails... (this may take a few minutes)")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(emb_file, embeddings)
    print(f"Embeddings saved to: {emb_file}")

# Train IsolationForest
print("Training IsolationForest...")

# Contamination: proportion of anomalies expected (small since anomalies are phishing emails)
contamination = 0.5

iforest_model = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=contamination,
    random_state=42
)
iforest_model.fit(embeddings)

# Save the trained model
dump(iforest_model, model_path)
print(f"IsolationForest model saved at:\n{model_path}")
