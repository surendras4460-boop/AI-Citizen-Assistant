"""Local vector embedding engine supporting ONNX runtime and transformer models."""
from __future__ import annotations

import logging
import re
import zlib
from typing import List, Optional

import numpy as np

from src.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """Encodes textual chunks into dense semantic vector representations.
    
    Tries backends in priority order:
    1. SentenceTransformers (PyTorch-based, if installed)
    2. Chroma ONNX DefaultEmbeddingFunction (lightweight, zero PyTorch dependency)
    3. Token Hashing Normalized Vectorizer (deterministic fallback)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._backend = "hash_vectorizer"

        # 1. Try sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._backend = "sentence_transformers"
            logger.info("Initialized SentenceTransformer backend with %s", model_name)
            return
        except Exception:
            pass

        # 2. Try Chroma's DefaultEmbeddingFunction (ONNX)
        try:
            import chromadb.utils.embedding_functions as ef
            self._model = ef.DefaultEmbeddingFunction()
            self._backend = "chroma_onnx"
            logger.info("Initialized Chroma ONNX embedding backend")
            return
        except Exception:
            pass

        logger.warning("Using fallback normalized hash vectorizer")

    @property
    def backend(self) -> str:
        """Returns the active embedding backend."""
        return self._backend

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of input texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of 384-dimensional float vector representations.
        """
        if not texts:
            return []

        if self._backend == "sentence_transformers":
            embs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs.tolist()

        if self._backend == "chroma_onnx":
            try:
                embs = self._model(texts)
                return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]
            except Exception as e:
                logger.error("ONNX embedding error: %s. Falling back to hashing.", e)

        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding vector for a single query string.
        
        Args:
            text: Query string.
            
        Returns:
            384-dimensional float vector.
        """
        res = self.embed_texts([text])
        if not res:
            return [0.0] * EMBEDDING_DIMENSION
        first = res[0]
        if hasattr(first, "tolist"):
            return first.tolist()
        return list(first)

    def _hash_embed(self, text: str, dim: int = EMBEDDING_DIMENSION) -> List[float]:
        """Deterministic, zero-dependency token hashing vectorizer with L2 normalization."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            h = zlib.crc32(tok.encode('utf-8')) % dim  # stable across processes (built-in hash() is salted)
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()
