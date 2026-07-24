"""
Unit tests for the ingestion pipeline: PDF parsing (text + tables +
embedded images), standalone image OCR, and plain-text chunking.

Run with: pytest tests/test_ingestion.py -v
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.ingestion.document_parser import _chunk_text, _table_to_markdown, parse_pdf, parse_text_file
from src.ingestion.image_processor import process_image_file

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_documents"


@pytest.fixture()
def tmp_image_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_chunk_text_respects_paragraph_boundaries():
    paragraphs = [f"Paragraph number {i} about warehouse robots and logistics." for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_text(text, max_chars=200)
    assert len(chunks) >= 2
    assert all(len(c) <= 250 for c in chunks)  # small overshoot allowed for the paragraph that tips it over


def test_chunk_text_drops_tiny_fragments_but_keeps_something():
    assert _chunk_text("") == []
    assert _chunk_text("hi") == ["hi"]  # single short doc still returned, not silently dropped


def test_table_to_markdown_preserves_structure():
    table = [["A", "B"], ["1", "2"], ["3", "4"]]
    md = _table_to_markdown(table)
    assert "| A | B |" in md
    assert "| 1 | 2 |" in md
    assert md.count("\n") == 3  # header + separator + 2 rows


def test_parse_text_file_produces_text_chunks(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("This is a short note about warehouse robots and safety.")
    chunks = parse_text_file(str(p))
    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "text"
    assert chunks[0]["document_id"] == "note.txt"
    assert chunks[0]["page_number"] == 1
    assert "warehouse" in chunks[0]["text"]


def test_parse_pdf_extracts_text_table_and_image(tmp_image_dir):
    pdf_path = SAMPLES / "pdfs" / "annual_report_2023.pdf"
    assert pdf_path.exists(), "sample PDF missing -- run scripts/generate_sample_documents.py"

    chunks = parse_pdf(str(pdf_path), image_output_dir=tmp_image_dir)
    content_types = {c["content_type"] for c in chunks}

    assert "text" in content_types
    assert "table" in content_types
    assert "image" in content_types

    for c in chunks:
        assert c["document_id"] == "annual_report_2023.pdf"
        assert c["page_number"] >= 1
        if c["content_type"] == "image":
            assert c["image_path"] and Path(c["image_path"]).exists()


def test_process_image_file_runs_ocr_and_sets_metadata():
    img_path = SAMPLES / "images" / "quarterly_sales_chart.png"
    assert img_path.exists(), "sample image missing -- run scripts/generate_sample_documents.py"

    chunk = process_image_file(str(img_path))
    assert chunk["content_type"] == "image"
    assert chunk["document_id"] == "quarterly_sales_chart.png"
    assert chunk["page_number"] == 1
    assert chunk["image_path"] == str(img_path.resolve())
    # OCR should pick up at least part of the chart's title text
    assert "sales" in chunk["text"].lower() or "quarterly" in chunk["text"].lower()
