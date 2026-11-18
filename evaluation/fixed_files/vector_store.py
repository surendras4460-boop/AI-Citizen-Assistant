"""Vector database index management and similarity retrieval backed by ChromaDB."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import DEFAULT_SIMILARITY_METRIC, DEFAULT_TOP_K
from src.embedder import SentenceEmbedder
from src.models import Chunk

logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages an isolated ChromaDB vector collection with cosine similarity search.
    
    Attributes:
        name: Name of the vector collection.
        embedder: The embedding engine instance.
        persist_path: Optional filesystem path for disk persistence.
    """

    def __init__(
        self,
        name: str,
        embedder: SentenceEmbedder,
        persist_path: Optional[str] = None
    ):
        self.name = name
        self.embedder = embedder
        self.persist_path = persist_path
        self.chunks: List[Chunk] = []
        self._collection = None

        try:
            import chromadb
            if persist_path:
                client = chromadb.PersistentClient(path=str(persist_path))
            else:
                client = chromadb.Client()

            # Ensure session isolation for runtime uploads by purging old session data
            if not persist_path:
                try:
                    client.delete_collection(name)
                except Exception:
                    pass

            self._collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": DEFAULT_SIMILARITY_METRIC}
            )
        except Exception as e:
            logger.warning("ChromaDB initialization failed (%s); operating in memory.", e)
            self._collection = None

    def add(self, chunks: List[Chunk]) -> None:
        """Add document chunks to the vector collection.
        
        Args:
            chunks: List of Chunk objects to index.
        """
        if not chunks:
            return

        self.chunks.extend(chunks)

        if self._collection is not None:
            docs = [c.text for c in chunks]
            # Deterministic IDs make re-indexing idempotent (no duplicates on app restart).
            ids, seen = [], {}
            for c in chunks:
                base = hashlib.sha1(
                    (c.text + "|" + str(c.meta.get("source", "")) + "|" + str(c.meta.get("page", ""))).encode("utf-8")
                ).hexdigest()[:16]
                n = seen.get(base, 0); seen[base] = n + 1
                ids.append(f"{self.name}_{base}_{n}")
            metas: List[Dict[str, Any]] = []

            for c in chunks:
                safe_meta: Dict[str, Any] = {}
                for k, v in c.meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_meta[k] = v
                    else:
                        safe_meta[k] = str(v)
                metas.append(safe_meta)

            embeddings = self.embedder.embed_texts(docs)
            try:
                self._collection.upsert(
                    ids=ids,
                    documents=docs,
                    embeddings=embeddings,
                    metadatas=metas
                )
            except Exception as e:
                logger.error("Failed to add vectors to ChromaDB collection: %s", e)

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> List[Chunk]:
        """Perform vector similarity search for the given query text.
        
        Args:
            query_text: User question or search phrase.
            top_k: Maximum number of closest chunks to retrieve.
            
        Returns:
            List of matching Chunk objects with cosine similarity scores.
        """
        if not self.chunks:
            return []

        actual_k = min(top_k, len(self.chunks))

        # Query ChromaDB collection
        if self._collection is not None:
            try:
                count = self._collection.count()
                if count > 0:
                    q_emb = self.embedder.embed_query(query_text)
                    res = self._collection.query(
                        query_embeddings=[q_emb],
                        n_results=min(actual_k, count),
                        include=["documents", "metadatas", "distances"]
                    )
                    hits: List[Chunk] = []
                    docs = res.get("documents", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    distances = res.get("distances", [[]])[0]

                    for doc, m, dist in zip(docs, metas, distances):
                        # Cosine distance d in [0, 2] -> similarity = 1 - d
                        if dist is not None:
                            d_val = float(dist)
                            score = max(0.0, min(1.0, 1.0 - d_val))
                        else:
                            score = 0.85
                        m_copy = dict(m)
                        m_copy["score"] = round(score, 3)
                        hits.append(Chunk(text=doc, meta=m_copy))
                    return hits
            except Exception as e:
                logger.warning("ChromaDB query failed (%s); using numpy similarity fallback.", e)

        # Fallback local in-memory cosine similarity
        q_vec = np.array(self.embedder.embed_query(query_text), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        scored_chunks: List[Chunk] = []

        chunk_texts = [c.text for c in self.chunks]
        c_embs = self.embedder.embed_texts(chunk_texts)

        for c, emb in zip(self.chunks, c_embs):
            vec = np.array(emb, dtype=np.float32)
            denom = q_norm * np.linalg.norm(vec)
            sim = float(np.dot(q_vec, vec) / denom) if denom > 0 else 0.0
            meta_copy = dict(c.meta)
            meta_copy["score"] = round(max(0.0, sim), 3)
            scored_chunks.append(Chunk(text=c.text, meta=meta_copy))

        scored_chunks.sort(key=lambda x: x.meta.get("score", 0.0), reverse=True)
        return scored_chunks[:actual_k]


def retrieve(
    indexes: List[Optional[VectorIndex]],
    question: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Chunk]:
    """Retrieve and rank chunks across multiple vector indexes.
    
    Args:
        indexes: Collection of active vector stores (filters out None).
        question: User query string.
        top_k: Overall retrieval depth limit.
        
    Returns:
        Sorted list of candidate Chunk objects by score descending.
    """
    all_hits: List[Chunk] = []
    valid_indexes = [idx for idx in (indexes or []) if idx is not None]

    if not valid_indexes:
        return []

    for idx in valid_indexes:
        try:
            hits = idx.query(question, top_k=top_k)
            all_hits.extend(hits)
        except Exception as e:
            logger.error("Error retrieving from index %s: %s", idx.name, e)

    # Sort descending by score
    all_hits.sort(key=lambda h: h.meta.get("score", 0.0), reverse=True)
    return all_hits[:top_k]
