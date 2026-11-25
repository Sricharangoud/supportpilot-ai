FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend and knowledge base
COPY backend ./backend
COPY data/kb ./data/kb

ENV PYTHONPATH=/app/backend
ENV KB_DIR=/app/data/kb
ENV FAISS_INDEX_PATH=/app/data/faiss_index
ENV DATABASE_URL=sqlite:////app/backend/supportpilot.db

# Pre-download embedding model and build FAISS index at image build time
RUN python -c "import sys; sys.path.insert(0, '/app/backend'); from app.rag import VectorStore; vs = VectorStore('/app/data/faiss_index'); vs.ingest_directory('/app/data/kb', 500, 50)"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
