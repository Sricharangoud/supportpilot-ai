FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/kb ./data/kb

ENV PYTHONPATH=/app
ENV KB_DIR=/app/data/kb
ENV FAISS_INDEX_PATH=/app/data/faiss_index
ENV DATABASE_URL=sqlite:////app/supportpilot.db
ENV FASTEMBED_CACHE_PATH=/app/data/fastembed_cache

# Pre-download embedding model and build FAISS index at image build time
RUN python -c "from app.rag import VectorStore; vs = VectorStore('/app/data/faiss_index'); vs.ingest_directory('/app/data/kb', 500, 50)"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
