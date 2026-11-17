"""AI Citizen Assistant - Core Facade Module.
Re-exports modular components from the `src` package to provide a clean,
unified public interface while preserving backward compatibility.
"""
from src.embedder import SentenceEmbedder
from src.llm_client import GeminiClient, generate
from src.models import Chunk, RetrievalResult
from src.pdf_processor import pdf_to_chunks
from src.prompt_templates import qa_prompt, summary_prompt
from src.schemes_loader import clean_text as _clean
from src.schemes_loader import load_schemes_df, schemes_to_chunks
from src.vector_store import VectorIndex, retrieve

__all__ = [
    "SentenceEmbedder",
    "VectorIndex",
    "Chunk",
    "RetrievalResult",
    "pdf_to_chunks",
    "schemes_to_chunks",
    "load_schemes_df",
    "retrieve",
    "qa_prompt",
    "summary_prompt",
    "generate",
    "GeminiClient",
    "_clean",
]
