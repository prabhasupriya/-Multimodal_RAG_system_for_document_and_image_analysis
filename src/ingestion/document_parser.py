"""
Document parsing for PDFs and plain text files.

For PDFs we use:
  * PyMuPDF (fitz)  -> per-page text + embedded raster images
  * pdfplumber      -> table detection/extraction (rows & columns preserved)

Every extracted unit becomes a "chunk" dict with a consistent schema so
downstream embedding/indexing code doesn't need to special-case content
types:

    {
        "document_id": str,
        "page_number": int,
        "content_type": "text" | "image" | "table",
        "text": str,            # text content, table-as-markdown, or OCR text for images
        "image_path": str|None, # populated only for content_type == "image"
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.ingestion.image_processor import run_ocr

logger = logging.getLogger(__name__)

MIN_CHARS_PER_CHUNK = 40
MAX_CHARS_PER_CHUNK = 1000


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Paragraph-aware chunking: keep paragraphs together, split only when a
    paragraph run grows past max_chars. Avoids splitting a chart's caption
    away from the sentence describing it."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHARS_PER_CHUNK] or ([text] if text.strip() else [])


def _table_to_markdown(table: List[List[str]]) -> str:
    rows = [[cell if cell is not None else "" for cell in row] for row in table]
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_pdf(path: str, image_output_dir: str) -> List[dict]:
    """Extract text chunks, tables, and embedded images from a PDF."""
    document_id = Path(path).name
    out_dir = Path(image_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[dict] = []

    # --- text + embedded images via PyMuPDF ---
    doc = fitz.open(path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        text = page.get_text("text")
        for chunk_text in _chunk_text(text):
            chunks.append(
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "content_type": "text",
                    "text": chunk_text,
                    "image_path": None,
                }
            )

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                image_path = out_dir / f"{document_id}_p{page_number}_img{img_index}.{ext}"
                image_path.write_bytes(image_bytes)
                with Image.open(image_path) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    ocr_text = run_ocr(pil_img)
                chunks.append(
                    {
                        "document_id": document_id,
                        "page_number": page_number,
                        "content_type": "image",
                        "text": ocr_text,
                        "image_path": str(image_path.resolve()),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract image xref=%s on page %s: %s", xref, page_number, exc)
    doc.close()

    # --- tables via pdfplumber (structure-preserving) ---
    try:
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                for table in page.extract_tables():
                    md = _table_to_markdown(table)
                    if md:
                        chunks.append(
                            {
                                "document_id": document_id,
                                "page_number": page_number,
                                "content_type": "table",
                                "text": md,
                                "image_path": None,
                            }
                        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Table extraction failed for %s: %s", path, exc)

    return chunks


def parse_text_file(path: str) -> List[dict]:
    document_id = Path(path).name
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [
        {
            "document_id": document_id,
            "page_number": 1,
            "content_type": "text",
            "text": chunk_text,
            "image_path": None,
        }
        for chunk_text in _chunk_text(text)
    ]
