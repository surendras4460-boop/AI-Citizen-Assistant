"""Configuration and constants for the AI Citizen Assistant application."""
from pathlib import Path
from typing import List

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

# Dataset paths
SCHEMES_CSV_PATH = DATA_DIR / "schemes.csv"
FALLBACK_SCHEMES_CSV_PATH = BASE_DIR / "schemes.csv"

# Vector Database configuration
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_schemes"
CITIZEN_DOCS_COLLECTION = "citizen_documents"
SCHEMES_COLLECTION = "schemes"
DEFAULT_SIMILARITY_METRIC = "cosine"
DEFAULT_TOP_K = 20

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Text chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM configuration (Google Gemini)
GEMINI_MODELS_CASCADE: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]
REQUEST_TIMEOUT_SECONDS = 30
MAX_HISTORY_TURNS = 6
