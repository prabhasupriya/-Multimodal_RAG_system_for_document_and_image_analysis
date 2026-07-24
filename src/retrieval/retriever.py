"""
Cross-modal retriever: queries the unified vector store and fuses results
from different modalities into a single ranked context list.

Fusion strategy
----------------
Chroma already returns one similarity-ranked list across all modalities
(since everything shares one embedding space). On top of that raw
ranking we apply a light re-ranking pass that:

  1. Applies a small modality prior (configurable) so that, for a given
     similarity score, an under-represented modality isn't drowned out
     by a flood of near-duplicate text chunks (a common failure mode:
     10 text chunks all about "revenue" out-ranking the one chart that
     actually shows it).
  2. Deduplicates near-identical hits from the same page/modality.
  3. Guarantees modality diversity: reserves slots so the final context
     passed to the VLM contains images/tables when they exist in the
     corpus, rather than an all-text context.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from src.vector_store.chroma_manager import ChromaManager

DEFAULT_MODALITY_PRIOR = {"image": 1.05, "table": 1.02, "text": 1.0}


class Retriever:
    def __init__(self, store: ChromaManager, modality_prior: dict | None = None):
        self.store = store
        self.modality_prior = modality_prior or DEFAULT_MODALITY_PRIOR

    def retrieve(self, query: str, top_k: int = 8, ensure_modality_diversity: bool = True) -> List[dict]:
        raw_hits = self.store.query(query, top_k=max(top_k * 3, top_k))

        # 1. modality-weighted re-scoring
        for hit in raw_hits:
            prior = self.modality_prior.get(hit["content_type"], 1.0)
            hit["fused_score"] = hit["score"] * prior

        # 2. dedupe identical (document, page, content_type, text) hits
        seen = set()
        deduped = []
        for hit in sorted(raw_hits, key=lambda h: h["fused_score"], reverse=True):
            key = (hit["document_id"], hit["page_number"], hit["content_type"], hit["text"][:120])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)

        if not ensure_modality_diversity:
            return deduped[:top_k]

        # 3. reserve at least one slot per available modality if present
        by_modality: dict[str, list] = defaultdict(list)
        for hit in deduped:
            by_modality[hit["content_type"]].append(hit)

        final: List[dict] = []
        for modality in ("image", "table", "text"):
            if by_modality[modality]:
                final.append(by_modality[modality][0])

        remaining = [h for h in deduped if h not in final]
        for hit in remaining:
            if len(final) >= top_k:
                break
            final.append(hit)

        final.sort(key=lambda h: h["fused_score"], reverse=True)
        return final[:top_k]
