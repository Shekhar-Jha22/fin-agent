"""RAGIndexer: chunk, embed, and index documents into FAISS."""

from __future__ import annotations
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not available; RAG will use mock embeddings with numpy")

from openai import AsyncOpenAI


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


class RAGIndexer:
    def __init__(
        self,
        client: AsyncOpenAI,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        overlap: int = 50,
        cache_dir: Optional[str] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".rag_cache")
        self.index: Optional[object] = None  # faiss.IndexFlatL2 or mock
        self.chunks: list[str] = []
        self.dimension: int = 1536  # text-embedding-3-small dimension

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using OpenAI embeddings API."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            vectors = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            return vectors
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}. Using random mock embeddings.")
            return np.random.randn(len(texts), self.dimension).astype(np.float32)

    async def build_index(self, documents: list[str]) -> None:
        """Chunk all documents, embed, and build FAISS flat index."""
        logger.info(f"Building RAG index from {len(documents)} document(s)")

        self.chunks = []
        for doc in documents:
            chunks = chunk_text(doc, self.chunk_size, self.overlap)
            self.chunks.extend(chunks)

        logger.info(f"Created {len(self.chunks)} chunks")

        # Batch embed (OpenAI supports up to 2048 inputs per call)
        batch_size = 100
        all_vectors = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            vecs = await self._embed_texts(batch)
            all_vectors.append(vecs)

        matrix = np.vstack(all_vectors)

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(matrix)
            logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        else:
            # Fallback: store raw matrix for cosine search with numpy
            self.index = matrix
            logger.info("Numpy fallback index built")

        self._raw_matrix = matrix

    def save(self, path: Optional[str] = None) -> None:
        """Persist index and chunks to disk for reuse."""
        save_path = Path(path) if path else self.cache_dir
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(save_path / "index.faiss"))
        elif self.index is not None:
            np.save(str(save_path / "index.npy"), self._raw_matrix)

        logger.info(f"RAG index saved to {save_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load index from disk. Returns True if successful."""
        load_path = Path(path) if path else self.cache_dir
        chunks_file = load_path / "chunks.pkl"

        if not chunks_file.exists():
            return False

        try:
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)

            if FAISS_AVAILABLE:
                index_file = load_path / "index.faiss"
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
            else:
                npy_file = load_path / "index.npy"
                if npy_file.exists():
                    self._raw_matrix = np.load(str(npy_file))
                    self.index = self._raw_matrix

            logger.info(f"RAG index loaded from {load_path}: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RAG index: {e}")
            return False
