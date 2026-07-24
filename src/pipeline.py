"""
Orchestrates the decoupled stages: Ingestion -> Chunking -> Embedding ->
Indexing. Retrieval and Generation are handled separately (see
src/retrieval and src/generation) so each stage stays independently
testable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.ingestion.document_parser import parse_pdf, parse_text_file
from src.ingestion.image_processor import process_image_file
from src.vector_store.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXT = {".txt", ".md"}
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_PDF_EXT = {".pdf"}


def parse_any(path: str, image_output_dir: str) -> List[dict]:
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_PDF_EXT:
        return parse_pdf(path, image_output_dir)
    if ext in SUPPORTED_IMAGE_EXT:
        return [process_image_file(path)]
    if ext in SUPPORTED_TEXT_EXT:
        return parse_text_file(path)
    logger.warning("Unsupported file type skipped: %s", path)
    return []


def ingest_directory(
    directory: str,
    store: ChromaManager,
    image_output_dir: str = "./extracted_images",
) -> dict:
    """Ingest every supported file in a directory (recursively) into the store."""
    directory_path = Path(directory)
    files = [
        p
        for p in directory_path.rglob("*")
        if p.is_file() and p.suffix.lower() in (SUPPORTED_TEXT_EXT | SUPPORTED_IMAGE_EXT | SUPPORTED_PDF_EXT)
    ]

    stats = {"files_processed": 0, "files_failed": 0, "chunks_indexed": 0}
    for file_path in files:
        try:
            chunks = parse_any(str(file_path), image_output_dir)
            added = store.add_chunks(chunks)
            stats["chunks_indexed"] += added
            stats["files_processed"] += 1
            logger.info("Ingested %s -> %d chunks", file_path.name, added)
        except Exception as exc:  # noqa: BLE001
            stats["files_failed"] += 1
            logger.error("Failed to ingest %s: %s", file_path, exc)
    return stats
