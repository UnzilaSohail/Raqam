FROM python:3.12-slim

# opencv-python-headless needs libGL-free runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY raqam ./raqam
COPY selfcheck.py .

# models/ and data/ are runtime volumes; MLP is trained on first boot if absent
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m raqam.train 2>/dev/null || true; exec uvicorn raqam.web.app:app --host 0.0.0.0 --port ${PORT}"]
