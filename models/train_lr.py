# src/training/train_lr.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

# Paths
root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
processed_file = os.path.join(root, "data", "processed", "phishing_email_processed.csv")
emb_file = os.path.join(root, "data", "processed", "embeddings.npy")
model_path = os.path.join(root, "models", "lr_model.joblib")

# Load data
df = pd.read_csv(processed_file)
embeddings = np.load(emb_file)
labels = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)
lr_model.fit(X_train, y_train)

# Save model
dump(lr_model, model_path)
print(f"Logistic Regression model saved at: {model_path}")

# Evaluate
y_pred = lr_model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
