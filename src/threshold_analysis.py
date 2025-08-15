import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import os

root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
processed_file = os.path.join(root, "data", "processed", "phishing_email_processed.csv")
emb_file = os.path.join(root, "data", "processed", "embeddings.npy")
model_path = os.path.join(root, "models", "iforest_model.joblib")

THRESHOLD = 0.005

df = pd.read_csv(processed_file)
embeddings = np.load(emb_file)
iforest_model = load(model_path)

scores = iforest_model.decision_function(embeddings)
df['anomaly_score'] = scores
df['phishing_flag'] = df['anomaly_score'] > THRESHOLD  

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(df['label'], df['phishing_flag'])
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(df['label'], df['phishing_flag']))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (Threshold={THRESHOLD})')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df[df['label']==0]['anomaly_score'], color='blue', label='Legitimate', bins=50, kde=True, stat="density")
sns.histplot(df[df['label']==1]['anomaly_score'], color='red', label='Phishing', bins=50, kde=True, stat="density")
plt.axvline(THRESHOLD, color='black', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Anomaly Score Distribution: Legitimate vs Phishing')
plt.legend()
plt.tight_layout()
plt.show()
