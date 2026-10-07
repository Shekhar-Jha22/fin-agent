"""RAGRetriever: similarity search over FAISS index."""

from __future__ import annotations
import logging
import numpy as np
from openai import AsyncOpenAI
from .indexer import RAGIndexer

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class RAGRetriever:
    def __init__(self, indexer: RAGIndexer, top_k: int = 3):
        self.indexer = indexer
        self.client = indexer.client
        self.top_k = top_k
        self.embedding_model = indexer.embedding_model

    async def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=[query],
            )
            return np.array([response.data[0].embedding], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}. Using random vector.")
            return np.random.randn(1, self.indexer.dimension).astype(np.float32)

    async def retrieve(self, query: str) -> list[str]:
        """Return top-k most relevant chunks for the query."""
        if self.indexer.index is None or not self.indexer.chunks:
            logger.warning("RAG index not built; returning empty context")
            return []

        query_vec = await self._embed_query(query)

        if FAISS_AVAILABLE and hasattr(self.indexer.index, "search"):
            # FAISS search
            k = min(self.top_k, len(self.indexer.chunks))
            distances, indices = self.indexer.index.search(query_vec, k)
            results = [self.indexer.chunks[int(i)] for i in indices[0] if i >= 0]
        else:
            # Numpy cosine similarity fallback
            matrix = self.indexer._raw_matrix  # (N, D)
            q = query_vec[0]  # (D,)

            # Normalize
            norm_q = q / (np.linalg.norm(q) + 1e-8)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            norm_matrix = matrix / norms
            similarities = norm_matrix @ norm_q  # (N,)

            top_indices = np.argsort(similarities)[::-1][: self.top_k]
            results = [self.indexer.chunks[int(i)] for i in top_indices]

        logger.info(f"RAG retrieved {len(results)} chunks for query: {query[:60]}...")
        return results

    async def retrieve_with_scores(self, query: str) -> list[tuple[str, float]]:
        """Return top-k chunks with similarity scores."""
        if self.indexer.index is None or not self.indexer.chunks:
            return []

        query_vec = await self._embed_query(query)

        if FAISS_AVAILABLE and hasattr(self.indexer.index, "search"):
            k = min(self.top_k, len(self.indexer.chunks))
            distances, indices = self.indexer.index.search(query_vec, k)
            results = [
                (self.indexer.chunks[int(i)], float(distances[0][j]))
                for j, i in enumerate(indices[0])
                if i >= 0
            ]
        else:
            matrix = self.indexer._raw_matrix
            q = query_vec[0]
            norm_q = q / (np.linalg.norm(q) + 1e-8)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            norm_matrix = matrix / norms
            similarities = norm_matrix @ norm_q
            top_indices = np.argsort(similarities)[::-1][: self.top_k]
            results = [(self.indexer.chunks[int(i)], float(similarities[i])) for i in top_indices]

        return results
