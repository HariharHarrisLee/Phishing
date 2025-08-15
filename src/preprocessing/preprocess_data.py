import pandas as pd
import os
import re

root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
raw_file = os.path.join(root, "data", "raw", "phishing_email.csv")
processed_file = os.path.join(root, "data", "processed", "phishing_email_processed.csv")

df = pd.read_csv(raw_file)

def clean_text(text):
    text = str(text).lower()               
    text = re.sub(r'\s+', ' ', text)      
    text = re.sub(r'http\S+', '', text)   
    text = re.sub(r'\S+@\S+', '', text)   
    text = text.strip()
    return text

df['text_clean'] = df['text_combined'].apply(clean_text)

df.to_csv(processed_file, index=False)

print(f"Preprocessing complete. Processed file saved at:\n{processed_file}")
print(f"Columns in processed CSV: {df.columns.tolist()}")
print(f"First 3 rows:\n{df.head(3)}")
