"""PDF parsing, text layer extraction, and semantic character-level chunking."""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

import pypdf

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.models import Chunk

logger = logging.getLogger(__name__)


def pdf_to_chunks(
    file_obj: Any,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> Tuple[List[Chunk], List[int]]:
    """Extract page-wise text and produce overlapping semantic chunks with metadata.
    
    Args:
        file_obj: File-like object, Streamlit UploadedFile, or file path.
        filename: Original name of the parent file.
        chunk_size: Character count target per segment.
        chunk_overlap: Overlapping characters between consecutive segments.
        
    Returns:
        Tuple containing:
            - List of Chunk objects tagged with source filename and page numbers.
            - List of 1-based page numbers skipped due to absence of extractable text.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        split_fn = splitter.split_text
    except Exception:
        def split_fn(text: str) -> List[str]:
            res: List[str] = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                res.append(text[start:end])
                if end == len(text):
                    break
                start += (chunk_size - chunk_overlap)
            return res

    reader = pypdf.PdfReader(file_obj)
    chunks: List[Chunk] = []
    skipped: List[int] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        clean_page_text = raw_text.strip()

        if not clean_page_text:
            logger.info("Page %d of %s has no text layer; skipping.", page_num, filename)
            skipped.append(page_num)
            continue

        pieces = split_fn(clean_page_text)
        for piece in pieces:
            content = piece.strip()
            if content:
                chunks.append(Chunk(
                    text=content,
                    meta={
                        "kind": "user_doc",
                        "source": filename,
                        "page": f"Page {page_num}",
                        "score": 0.0
                    }
                ))

    logger.info("Extracted %d chunks from %s (%d pages skipped)", len(chunks), filename, len(skipped))
    return chunks, skipped
