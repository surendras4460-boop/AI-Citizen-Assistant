"""Data transfer objects and domain models for document processing and retrieval."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents an atomic text segment with associated source metadata.
    
    Attributes:
        text (str): The raw text segment content.
        meta (Dict[str, Any]): Metadata dictionary including:
            - kind (str): 'user_doc' or 'scheme'
            - source (str): Originating file name or scheme name
            - page (Union[str, int]): Page or section identifier
            - score (float): Calculated similarity score in [0.0, 1.0]
    """
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.meta.get("source", "Unknown")

    @property
    def page(self) -> str:
        return str(self.meta.get("page", ""))

    @property
    def score(self) -> float:
        return float(self.meta.get("score", 0.0))

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "document")


@dataclass
class RetrievalResult:
    """Represents the outcome of a vector similarity query across one or more indexes."""
    query: str
    hits: List[Chunk] = field(default_factory=list)
    total_candidates: int = 0
