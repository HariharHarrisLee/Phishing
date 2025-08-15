FROM python:3.11-slim

WORKDIR /app/api

COPY requirements.txt ../

RUN pip install --no-cache-dir -r ../requirements.txt

COPY api/ ./

COPY models/ ../models/

COPY data/processed/ ../data/processed/

# Expose Flask port
EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

#Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
