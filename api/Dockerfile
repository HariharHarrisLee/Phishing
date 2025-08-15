# Base image
FROM python:3.11-slim

# Set working directory to the API folder
WORKDIR /app/api

# Copy requirements first (for caching)
COPY requirements.txt ../

# Install dependencies
RUN pip install --no-cache-dir -r ../requirements.txt

# Copy API code (app.py, templates, static)
COPY api/ ./

# Copy models
COPY models/ ../models/

# Copy processed data needed at runtime (embeddings if used)
COPY data/processed/ ../data/processed/

# Expose Flask port
EXPOSE 5000

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
