"""
Unified ChromaDB-backed vector store for text, image, and table chunks.

Design choice ("Hybrid Indexing"): all content types share ONE collection
whose vectors live in the same embedding space (CLIP, or the offline
fallback -- see src/embeddings/model_loader.py). This is what makes a
single text query able to surface a relevant image without a separate
image-query step. Each content type is still distinguishable at query
time via the `content_type` metadata field, so callers can also filter
per-modality if they want (e.g. "only images").
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Iterable, List, Optional

import chromadb
from chromadb.config import Settings

from src.embeddings.model_loader import get_embedder


def _stable_id(chunk: dict) -> str:
    """Idempotent ID so re-ingesting the same document doesn't duplicate it."""
    key = f"{chunk['document_id']}|{chunk['page_number']}|{chunk['content_type']}|{chunk.get('text','')[:200]}|{chunk.get('image_path','')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ChromaManager:
    def __init__(self, persist_dir: str = "./chroma_store", collection_name: str = "multimodal_rag"):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = get_embedder()

    def add_chunks(self, chunks: List[dict]) -> int:
        """Embed and upsert a list of chunk dicts. Returns number of chunks added."""
        if not chunks:
            return 0

        text_chunks = [c for c in chunks if c["content_type"] in ("text", "table")]
        image_chunks = [c for c in chunks if c["content_type"] == "image"]

        ids, embeddings, metadatas, documents = [], [], [], []

        if text_chunks:
            vecs = self.embedder.embed_text([c["text"] for c in text_chunks])
            for c, v in zip(text_chunks, vecs):
                ids.append(_stable_id(c))
                embeddings.append(v.tolist())
                metadatas.append(
                    {
                        "document_id": c["document_id"],
                        "page_number": c["page_number"],
                        "content_type": c["content_type"],
                        "image_path": c.get("image_path") or "",
                    }
                )
                documents.append(c["text"])

        if image_chunks:
            vecs = self.embedder.embed_image([c["image_path"] for c in image_chunks])
            for c, v in zip(image_chunks, vecs):
                ids.append(_stable_id(c))
                embeddings.append(v.tolist())
                metadatas.append(
                    {
                        "document_id": c["document_id"],
                        "page_number": c["page_number"],
                        "content_type": c["content_type"],
                        "image_path": c.get("image_path") or "",
                    }
                )
                # store OCR text (if any) as the document body for readability/debugging
                documents.append(c.get("text") or f"[image: {c.get('image_path')}]")

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        return len(ids)

    def query(self, text_query: str, top_k: int = 8, content_types: Optional[Iterable[str]] = None) -> List[dict]:
        """
        Cross-modal search: embeds the text query with the SAME embedder used
        for images, so text can retrieve text, tables, AND images in one pass.
        """
        query_vec = self.embedder.embed_text([text_query])[0].tolist()
        where = {"content_type": {"$in": list(content_types)}} if content_types else None

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )

        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "id": _id,
                    "text": doc,
                    "document_id": meta.get("document_id"),
                    "page_number": meta.get("page_number"),
                    "content_type": meta.get("content_type"),
                    "image_path": meta.get("image_path") or None,
                    "distance": dist,
                    "score": 1.0 - dist,  # cosine distance -> similarity
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
