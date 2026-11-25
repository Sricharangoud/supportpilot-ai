"""Document chunking, embedding, and FAISS vector store persistence."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np


@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def chunk_text(
    text: str,
    doc_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split text into overlapping token-approximated chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    chunk_id = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        chunks.append(Chunk(doc_name=doc_name, chunk_id=chunk_id, text=" ".join(window)))
        chunk_id += 1
        if start + chunk_size >= len(tokens):
            break
    return chunks


class VectorStore:
    """FAISS IndexFlatIP vector store persisted to disk alongside chunk metadata."""

    def __init__(
        self,
        index_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedder: Optional[Any] = None,
    ):
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self._embedder = embedder
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunks list.")

        vectors = self._encode([c.text for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.chunks = chunks
        self.save()

    def save(self) -> None:
        if self.index is None:
            return
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        faiss_file = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.json"
        if not faiss_file.exists() or not chunks_file.exists():
            return False

        self.index = faiss.read_index(str(faiss_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [Chunk(**item) for item in raw]
        return True

    def ingest_directory(
        self,
        kb_dir: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> tuple[int, int]:
        kb_path = Path(kb_dir)
        files = sorted(kb_path.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No markdown files found in {kb_dir}")

        all_chunks: list[Chunk] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            all_chunks.extend(
                chunk_text(content, doc_name=file_path.name, chunk_size=chunk_size, overlap=overlap)
            )

        self.build_from_chunks(all_chunks)
        return len(files), len(all_chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not self.is_loaded:
            raise RuntimeError("Vector store is not loaded. Run /ingest first.")

        q_vec = self._encode([query])
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results
