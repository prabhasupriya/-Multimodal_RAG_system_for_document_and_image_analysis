"""
Generation stage: turns retrieved multimodal context into a single,
visually-grounded answer.

Primary backend: any OpenAI-compatible vision-capable chat model.
Supports two providers out of the box, selected automatically from
environment variables (Groq checked first, then OpenAI):

  * Groq  -- given `GROQ_API_KEY` (fast, cheap, e.g. Llama-4-Scout-class
    vision models via https://api.groq.com/openai/v1).
  * OpenAI -- given `OPENAI_API_KEY` (e.g. `gpt-4o`).

Images are base64-encoded and sent alongside text context, per the
(shared) OpenAI-style vision message format.

Offline fallback: if no API key is configured (e.g. in CI, or during
local development without billing set up), a deterministic
`ExtractiveGenerator` synthesizes an answer directly from the
retrieved context so the system's API and retrieval quality can still
be evaluated end-to-end without external calls. It never claims to be
a VLM: it is clearly a template answer, but it does still perform
visual grounding by naming the modality, page and document number of
the item it draws from -- e.g. "As shown in the image on page 3 of
report.pdf ...".
"""
from __future__ import annotations

import base64
import logging
import os

from dotenv import load_dotenv

# Safety net: src/api/main.py also calls this, but generator.py is often
# imported directly (evaluation notebook, scripts, a REPL) without going
# through the API at all -- so this module loads .env for itself too.
# load_dotenv() is idempotent and never overrides variables already set
# explicitly in the environment (e.g. by a shell `export`/`set`).
load_dotenv()
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful multimodal research assistant.
You will be given a user question plus a set of retrieved context items
(text passages, tables rendered as markdown, and images) each tagged with
its source document and page number.

Rules:
1. Answer using ONLY the given context. If the context is insufficient, say so.
2. When your answer relies on an image, EXPLICITLY reference it in prose,
   e.g. "As shown in the chart on page 5 of report.pdf, ...".
3. When your answer relies on a table, reference it similarly, e.g.
   "According to the table on page 2, ...".
4. Be concise and factual. Do not invent numbers or details not present
   in the context.
"""


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMGenerator:
    """
    Real vision-capable generator via any OpenAI-compatible chat completions
    API. Used for both:

      * OpenAI itself (api_key from OPENAI_API_KEY, default base_url, e.g.
        model="gpt-4o"), and
      * Groq (api_key from GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
        e.g. model="meta-llama/llama-4-scout-17b-16e-instruct" or whichever
        vision-capable model Groq currently serves -- Groq's multimodal
        lineup changes fairly often, so check
        https://console.groq.com/docs/models for the current name and set
        GROQ_VLM_MODEL accordingly if the default here is stale).

    Groq's API mirrors OpenAI's chat completions format (including the
    `image_url` content-block format used below), so no other code needs
    to change -- only which client/base_url/model get plugged in here.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI  # lazy import, optional dependency

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, query: str, context: List[dict]) -> str:
        content = [{"type": "text", "text": f"Question: {query}\n\nContext items follow."}]
        for item in context:
            label = f"[{item['content_type'].upper()} | {item['document_id']} | page {item['page_number']}]"
            if item["content_type"] == "image" and item.get("image_path") and Path(item["image_path"]).exists():
                content.append({"type": "text", "text": label})
                b64 = _encode_image_b64(item["image_path"])
                ext = Path(item["image_path"]).suffix.lstrip(".") or "png"
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}
                )
            else:
                content.append({"type": "text", "text": f"{label}\n{item['text']}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=600,
        )
        return response.choices[0].message.content


class ExtractiveGenerator:
    """
    Dependency-free, offline fallback generator.

    Synthesizes a grounded answer purely from retrieved context, with
    explicit references to the modality/page/document of each item used
    -- satisfying the "visually grounded" requirement even without a
    live VLM call.
    """

    def generate(self, query: str, context: List[dict]) -> str:
        if not context:
            return "I couldn't find any relevant information in the ingested documents to answer that."

        sentences = []
        for item in context:
            loc = f"page {item['page_number']} of {item['document_id']}"
            if item["content_type"] == "image":
                snippet = item["text"].strip().replace("\n", " ")
                if snippet:
                    sentences.append(f"As shown in the image on {loc}, the visible text/labels read: \"{snippet[:200]}\".")
                else:
                    sentences.append(f"An image on {loc} appears relevant, though no legible text could be extracted from it.")
            elif item["content_type"] == "table":
                sentences.append(f"According to the table on {loc}:\n{item['text']}")
            else:
                snippet = item["text"].strip().replace("\n", " ")
                sentences.append(f"Per the text on {loc}: {snippet[:300]}")

        header = f"Based on the retrieved context for the question \"{query}\":\n\n"
        return header + "\n\n".join(sentences)


_generator_singleton = None


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OPENAI_VLM_MODEL = "gpt-4o"


def get_generator():
    global _generator_singleton
    if _generator_singleton is not None:
        return _generator_singleton

    # Groq is checked first: if GROQ_API_KEY is set, prefer it (it's what
    # this deployment is configured to use). Falls through to OpenAI, then
    # to the offline extractive generator, if not set / init fails.
    if os.getenv("GROQ_API_KEY"):
        model = os.getenv("GROQ_VLM_MODEL", DEFAULT_GROQ_VLM_MODEL)
        try:
            _generator_singleton = VLMGenerator(
                model=model,
                api_key=os.getenv("GROQ_API_KEY"),
                base_url=GROQ_BASE_URL,
            )
            logger.info("Using VLMGenerator (Groq %s).", model)
            return _generator_singleton
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to initialize Groq VLMGenerator (%s). If this is a "
                "'model not found' error, Groq's vision lineup changes "
                "often -- check https://console.groq.com/docs/models and "
                "set GROQ_VLM_MODEL to a currently-served vision model. "
                "Falling back.",
                exc,
            )

    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("VLM_MODEL", DEFAULT_OPENAI_VLM_MODEL)
        try:
            _generator_singleton = VLMGenerator(model=model, api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Using VLMGenerator (OpenAI %s).", model)
            return _generator_singleton
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize VLMGenerator (%s); falling back to ExtractiveGenerator.", exc)

    logger.info("No OPENAI_API_KEY found; using offline ExtractiveGenerator.")
    _generator_singleton = ExtractiveGenerator()
    return _generator_singleton
