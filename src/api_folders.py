import os

root = r"C:\Users\Harri\OneDrive\Desktop\Phishing_Project\Phishing"

folders = [
    os.path.join(root, "src", "deployment"),
    os.path.join(root, "templates"),
    os.path.join(root, "static")
]

files = [
    os.path.join(root, "src", "deployment", "app.py"),
    os.path.join(root, "src", "deployment", "helpers.py"),
    os.path.join(root, "templates", "index.html"),
    os.path.join(root, "templates", "result.html"),
    os.path.join(root, "static", "style.css"),
    os.path.join(root, "Dockerfile"),
    os.path.join(root, ".dockerignore")
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
    print(f"Created file: {file}")

print("Flask + Docker deployment structure created!")