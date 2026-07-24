FROM python:3.11-slim

# System dependency: OCR engine used by pytesseract
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (src/, sample_documents/, etc.)
COPY . .

# Most hosts (Render, Railway, Fly.io, Cloud Run) inject the real port via
# $PORT at runtime -- default to 8000 for local `docker run` testing.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
