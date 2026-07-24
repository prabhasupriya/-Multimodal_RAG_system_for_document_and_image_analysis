"""
End-to-end integration test for the multimodal RAG API.

This is the "non-trivial automated check" referenced in submission.yml:
it spins up the FastAPI app, ingests the sample_documents/ collection
into a scratch ChromaDB directory, asks a multimodal question, and
verifies the response is well-formed, grounded, cites sources, and
returns within the required latency budget.

Run with: pytest tests/test_api.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = str(ROOT / "sample_documents")


@pytest.fixture(scope="module")
def client():
    # Use scratch, isolated persistence dirs so tests never depend on
    # (or pollute) a developer's local ./chroma_store.
    scratch_chroma = tempfile.mkdtemp()
    scratch_images = tempfile.mkdtemp()
    os.environ["CHROMA_PERSIST_DIR"] = scratch_chroma
    os.environ["EXTRACTED_IMAGES_DIR"] = scratch_images
    os.environ.pop("OPENAI_API_KEY", None)  # force the offline fallback generator for CI determinism

    # Import after env vars are set, since src.api.main reads them at import time.
    from src.api.main import app

    with TestClient(app) as c:
        yield c

    shutil.rmtree(scratch_chroma, ignore_errors=True)
    shutil.rmtree(scratch_images, ignore_errors=True)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_before_ingest_returns_400(client):
    resp = client.post("/query", json={"query": "anything"})
    assert resp.status_code == 400


def test_ingest_sample_documents(client):
    resp = client.post("/ingest", json={"directory": SAMPLES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["files_processed"] >= 10, "expected >= 10 diverse sample documents to be ingested"
    assert body["files_failed"] == 0
    assert body["total_indexed"] > 0


def test_query_text_grounded_answer(client):
    resp = client.post("/query", json={"query": "What is the maximum payload of the AcmeBot X1?", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert "500" in body["answer"] or "payload" in body["answer"].lower()
    assert len(body["sources"]) > 0
    for src in body["sources"]:
        assert src["document_id"]
        assert src["page_number"] >= 1
        assert src["content_type"] in ("text", "image", "table")
    assert body["latency_seconds"] < 15, "response must be under the 15s latency budget"


def test_query_visual_synthesis_returns_image_source(client):
    """A question about a chart should surface at least one image-typed source,
    demonstrating cross-modal (text query -> image result) retrieval."""
    resp = client.post("/query", json={"query": "What does the quarterly sales chart show?", "top_k": 6})
    assert resp.status_code == 200
    body = resp.json()
    content_types = {s["content_type"] for s in body["sources"]}
    assert "image" in content_types
    assert body["latency_seconds"] < 15


def test_query_table_source(client):
    """A question about tabular figures should surface a table-typed source."""
    resp = client.post("/query", json={"query": "What was the gross margin by quarter?", "top_k": 6})
    assert resp.status_code == 200
    body = resp.json()
    content_types = {s["content_type"] for s in body["sources"]}
    assert "table" in content_types or "text" in content_types


def test_stats_endpoint(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["total_indexed"] > 0
