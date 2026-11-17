"""Utilities for loading and converting government schemes data."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Union

import pandas as pd

from src.config import FALLBACK_SCHEMES_CSV_PATH, SCHEMES_CSV_PATH
from src.models import Chunk

logger = logging.getLogger(__name__)


def clean_text(val: Any) -> str:
    """Helper to clean string values and handle null/NaN values."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def load_schemes_df(csv_path: Union[str, Path] = "schemes.csv") -> pd.DataFrame:
    """Load government schemes dataset from CSV.
    
    Checks specified path, fallback project paths, and root directories.
    
    Args:
        csv_path: Path to schemes.csv.
        
    Returns:
        Pandas DataFrame containing schemes data.
        
    Raises:
        FileNotFoundError: If the CSV file cannot be located.
    """
    candidates = [
        Path(csv_path),
        Path(csv_path).resolve(),
        SCHEMES_CSV_PATH,
        FALLBACK_SCHEMES_CSV_PATH,
        Path.cwd() / "schemes.csv",
        Path.cwd() / "data" / "schemes.csv",
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            logger.info("Loaded schemes dataset from %s", cand)
            return pd.read_csv(cand, encoding="utf-8-sig")

    raise FileNotFoundError(f"Could not locate schemes dataset at: {csv_path}")


def schemes_to_chunks(df: pd.DataFrame) -> List[Chunk]:
    """Convert government schemes DataFrame into searchable Chunk objects.
    
    Generates bilingual chunks (English and Hindi) with complete scheme metadata.
    
    Args:
        df: Schemes DataFrame.
        
    Returns:
        List of Chunk objects.
    """
    chunks: List[Chunk] = []

    for _, r in df.iterrows():
        s_name = clean_text(r.get("scheme_name"))
        s_name_hi = clean_text(r.get("scheme_name_hindi"))
        cat = clean_text(r.get("schemeCategory"))
        level = clean_text(r.get("level"))
        details = clean_text(r.get("details"))
        details_hi = clean_text(r.get("details_hi"))
        benefits = clean_text(r.get("benefits"))
        eligibility = clean_text(r.get("eligibility"))
        application = clean_text(r.get("application"))
        documents = clean_text(r.get("documents"))

        # English scheme chunk
        en_parts = [
            f"Scheme Name: {s_name}",
            f"Category: {cat} (Level: {level})",
            f"Details: {details}" if details else "",
            f"Benefits: {benefits}" if benefits else "",
            f"Eligibility: {eligibility}" if eligibility else "",
            f"Application Process: {application}" if application else "",
            f"Required Documents: {documents}" if documents else ""
        ]
        en_text = "\n".join([p for p in en_parts if p])
        chunks.append(Chunk(
            text=en_text,
            meta={
                "kind": "scheme",
                "source": s_name,
                "page": "Guidelines & Overview",
                "score": 0.0
            }
        ))

        # Hindi scheme chunk
        if s_name_hi or details_hi:
            hi_parts = [
                f"योजना का नाम: {s_name_hi or s_name}",
                f"श्रेणी: {cat} (स्तर: {level})",
                f"विवरण: {details_hi or details}" if (details_hi or details) else "",
                f"लाभ: {benefits}" if benefits else "",
                f"पात्रता: {eligibility}" if eligibility else "",
                f"आवेदन प्रक्रिया: {application}" if application else "",
                f"आवश्यक दस्तावेज: {documents}" if documents else ""
            ]
            hi_text = "\n".join([p for p in hi_parts if p])
            chunks.append(Chunk(
                text=hi_text,
                meta={
                    "kind": "scheme",
                    "source": s_name_hi or s_name,
                    "page": "दिशानिर्देश व विवरण",
                    "score": 0.0
                }
            ))

    logger.info("Converted %d scheme rows into %d searchable chunks", len(df), len(chunks))
    return chunks
