import pandas as pd
import os

# Paths
root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"
raw_file = os.path.join(root, "data", "raw", "phishing_email.csv")

# Load CSV
df = pd.read_csv(raw_file)

# 1. Column names
print("Columns in dataset:")
print(df.columns.tolist(), "\n")

# 2. First 3 rows
print("First 3 rows:")
print(df.head(3), "\n")

# 3. Missing values per column
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 4. Class distribution (assuming 'label' column exists)
if 'label' in df.columns:
    print("Class distribution:")
    print(df['label'].value_counts())
else:
    print("No 'label' column found.")
