import os

# Root project folder
root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"

# Folder structure
folders = [
    "data/raw",
    "data/processed",
    "models",
    "src/preprocessing",
    "src/training",
    "src/inference",
    "api"
]

# Create folders
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)

print(f"Folder structure created under: {root}")
