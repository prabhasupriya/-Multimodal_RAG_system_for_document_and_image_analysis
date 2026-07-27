"""
FastAPI application for the multimodal RAG system.

Endpoints
---------
GET  /health          -> liveness check
POST /ingest           -> ingest a directory of documents *already on the server's disk*
                          (developer/admin use -- e.g. loading sample_documents/)
POST /upload            -> upload document file(s) from your own computer/browser
                          and ingest them (what an end user hitting a deployed
                          URL should use -- no server filesystem access needed)
POST /query             -> ask a question, get a visually-grounded answer + sources
GET  /stats            -> basic index statistics
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Must run before anything below reads os.getenv(...) -- this is what
# actually pulls GROQ_API_KEY / OPENAI_API_KEY / etc. out of a local .env
# file and into the process environment. Without this line, a .env file
# sitting on disk has no effect at all.
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.generation.generator import get_generator
from src.pipeline import SUPPORTED_IMAGE_EXT, SUPPORTED_PDF_EXT, SUPPORTED_TEXT_EXT, ingest_directory, parse_any
from src.retrieval.retriever import Retriever
from src.vector_store.chroma_manager import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EXTRACTED_IMAGES_DIR = os.getenv("EXTRACTED_IMAGES_DIR", "./extracted_images")
UPLOADED_DOCS_DIR = os.getenv("UPLOADED_DOCS_DIR", "./uploaded_documents")
SUPPORTED_UPLOAD_EXT = SUPPORTED_TEXT_EXT | SUPPORTED_IMAGE_EXT | SUPPORTED_PDF_EXT

app = FastAPI(
    title="Multimodal RAG API",
    description="Retrieval-Augmented Generation over text, tables, and images.",
    version="1.0.0",
)

_store = ChromaManager(persist_dir=CHROMA_DIR)
_retriever = Retriever(_store)


class IngestRequest(BaseModel):
    directory: str = Field(..., description="Path to a directory of documents to ingest.")


class IngestResponse(BaseModel):
    files_processed: int
    files_failed: int
    chunks_indexed: int
    total_indexed: int
    failed_files: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: int = Field(5, ge=1, le=20, description="Number of context items to retrieve.")


class Source(BaseModel):
    document_id: str
    page_number: int
    content_type: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_seconds: float


@app.get("/", include_in_schema=False)
def root():
    # There's no UI at the bare root -- send anyone visiting it straight to
    # the interactive Swagger docs instead of a bare, confusing 404.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "indexed_items": _store.count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not os.path.isdir(req.directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")
    stats = ingest_directory(req.directory, _store, image_output_dir=EXTRACTED_IMAGES_DIR)
    return IngestResponse(**stats, total_indexed=_store.count())


@app.post("/upload", response_model=IngestResponse)
async def upload(files: list[UploadFile] = File(..., description="One or more PDF/PNG/JPEG/TXT files.")):
    """
    Upload document file(s) directly from your own computer (via a browser
    file picker in /docs, or any HTTP client) and ingest them immediately.

    This is the endpoint an end user of a *deployed* instance should use --
    unlike /ingest, it needs no access to the server's filesystem at all.
    """
    os.makedirs(UPLOADED_DOCS_DIR, exist_ok=True)

    stats = {"files_processed": 0, "files_failed": 0, "chunks_indexed": 0}
    failed_files: list[str] = []

    for upload_file in files:
        # Strip any path components from the client-supplied filename so a
        # malicious "../../etc/passwd"-style name can't escape the uploads
        # directory (Zip Slip / path traversal).
        safe_name = Path(upload_file.filename or "unnamed").name
        ext = Path(safe_name).suffix.lower()

        if ext not in SUPPORTED_UPLOAD_EXT:
            stats["files_failed"] += 1
            failed_files.append(f"{safe_name} (unsupported file type: {ext or 'none'})")
            continue

        dest_path = Path(UPLOADED_DOCS_DIR) / safe_name
        try:
            contents = await upload_file.read()
            dest_path.write_bytes(contents)

            chunks = parse_any(str(dest_path), EXTRACTED_IMAGES_DIR)
            added = _store.add_chunks(chunks)
            stats["chunks_indexed"] += added
            stats["files_processed"] += 1
            logger.info("Uploaded + ingested %s -> %d chunks", safe_name, added)
        except Exception as exc:  # noqa: BLE001
            stats["files_failed"] += 1
            failed_files.append(f"{safe_name} ({exc})")
            logger.error("Failed to ingest uploaded file %s: %s", safe_name, exc)

    return IngestResponse(**stats, total_indexed=_store.count(), failed_files=failed_files)


@app.get("/stats")
def stats():
    return {"total_indexed": _store.count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()
    if _store.count() == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Call /ingest first.")

    context = _retriever.retrieve(req.query, top_k=req.top_k)
    generator = get_generator()
    answer = generator.generate(req.query, context)

    sources = [
        Source(
            document_id=item["document_id"],
            page_number=item["page_number"],
            content_type=item["content_type"],
            snippet=(item["image_path"] if item["content_type"] == "image" and item.get("image_path") else item["text"][:300]),
        )
        for item in context
    ]

    return QueryResponse(answer=answer, sources=sources, latency_seconds=round(time.time() - start, 3))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
