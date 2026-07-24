"""
Standalone image ingestion: OCR + light preprocessing for PNG/JPEG files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Grayscale + autocontrast pass that measurably improves tesseract accuracy."""
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


def run_ocr(image: Image.Image, lang: str = "eng") -> str:
    """Run tesseract OCR on a PIL image and return extracted text (best-effort)."""
    try:
        processed = preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed, lang=lang)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed: %s", exc)
        return ""


def process_image_file(path: str, document_id: Optional[str] = None) -> dict:
    """
    Ingest a single standalone image file.

    Returns a chunk dict compatible with the vector store schema:
        {id, document_id, page_number, content_type, text, image_path}
    """
    path_obj = Path(path)
    document_id = document_id or path_obj.name
    with Image.open(path) as img:
        img = img.convert("RGB")
        ocr_text = run_ocr(img)

    return {
        "document_id": document_id,
        "page_number": 1,
        "content_type": "image",
        "text": ocr_text,
        "image_path": str(path_obj.resolve()),
    }
