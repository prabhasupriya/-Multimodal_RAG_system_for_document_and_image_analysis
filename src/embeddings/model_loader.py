"""
Multimodal embedding model loader.

Primary backend: a CLIP model loaded via `sentence-transformers`
(`clip-ViT-B-32`), which places text and images in the SAME vector
space -- this is what enables true cross-modal search (a text query
retrieving images and vice-versa).

Because CLIP weights must be downloaded from the internet the first
time they are used, this module also ships a deterministic, fully
offline FALLBACK embedder. The fallback is automatically used when the
CLIP backend cannot be loaded (no network access, no GPU, etc.) so
that the rest of the pipeline (ingestion, indexing, retrieval, API,
tests) keeps working in restricted/CI environments.

Both backends expose the exact same interface:

    embed_text(list[str])  -> np.ndarray [n, dim]
    embed_image(list[str | PIL.Image]) -> np.ndarray [n, dim]
    dim -> int

so the rest of the codebase never needs to know which backend is
active.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

EMBED_DIM_FALLBACK = 384


class BaseEmbedder:
    dim: int

    def embed_text(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_image(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vecs / norms


class ClipEmbedder(BaseEmbedder):
    """Real multimodal CLIP embedder (text + image share one space)."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(np.asarray(vecs, dtype=np.float32))

    def embed_image(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        pil_images = [Image.open(im).convert("RGB") if isinstance(im, str) else im for im in images]
        vecs = self.model.encode(pil_images, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(np.asarray(vecs, dtype=np.float32))


class FallbackEmbedder(BaseEmbedder):
    """
    Deterministic, dependency-free embedder used when CLIP cannot be
    downloaded. It is NOT a substitute for a real vision-language
    embedding model, but it keeps the system fully functional offline:

      * Text -> hashed bag-of-words vector (stable, no network calls).
      * Image -> a small perceptual signature (downsampled grayscale
        histogram + average color) PLUS the OCR text extracted from
        the image (if any) is hashed the same way as normal text and
        blended in. This means charts/diagrams with titles, axis
        labels, or legends are still retrievable via keyword overlap,
        which is the dominant real-world signal for OCR-able figures.

    Both text and image vectors are produced in the same `dim`-sized
    space so cosine similarity comparisons between the two modalities
    remain meaningful.
    """

    def __init__(self, dim: int = EMBED_DIM_FALLBACK):
        self.dim = dim

    def _hash_bow(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def embed_text(self, texts: List[str]) -> np.ndarray:
        vecs = np.stack([self._hash_bow(t) for t in texts])
        return self._normalize(vecs)

    def _image_signature(self, image: Image.Image) -> np.ndarray:
        gray = image.convert("L").resize((16, 16))
        arr = np.asarray(gray, dtype=np.float32).flatten() / 255.0
        sig = np.zeros(self.dim, dtype=np.float32)
        n = min(len(arr), self.dim)
        sig[:n] = arr[:n]
        return sig

    def embed_image(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        # local import to avoid a hard dependency at module import time
        from src.ingestion.image_processor import run_ocr

        vecs = []
        for im in images:
            pil_image = Image.open(im).convert("RGB") if isinstance(im, str) else im
            sig = self._image_signature(pil_image)
            sig_norm = np.linalg.norm(sig)
            sig = sig / sig_norm if sig_norm > 0 else sig
            try:
                ocr_text = run_ocr(pil_image)
            except Exception:  # pragma: no cover - OCR binary may be missing
                ocr_text = ""
            if ocr_text.strip():
                text_vec = self._hash_bow(ocr_text)
                text_norm = np.linalg.norm(text_vec)
                text_vec = text_vec / text_norm if text_norm > 0 else text_vec
            else:
                text_vec = np.zeros(self.dim, dtype=np.float32)
            # Weight OCR-derived text more heavily than the raw pixel
            # signature: for a *text query* driving retrieval, keyword
            # overlap with a chart's title/axis labels/legend is a far
            # stronger relevance signal than pixel-level similarity.
            # Each sub-vector is unit-normalized above first, so neither
            # dominates the blend purely due to having more nonzero entries.
            vecs.append(0.75 * text_vec + 0.25 * sig)
        return self._normalize(np.stack(vecs))


_embedder_singleton: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Return a process-wide embedder instance, preferring real CLIP."""
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton

    try:
        _embedder_singleton = ClipEmbedder()
        logger.info("Loaded CLIP embedder (clip-ViT-B-32).")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load CLIP embedder (%s). Falling back to the offline "
            "hashed embedder. Install `sentence-transformers` and ensure "
            "internet access to use real CLIP embeddings.",
            exc,
        )
        _embedder_singleton = FallbackEmbedder()
    return _embedder_singleton
