# Multimodal RAG System for Document and Image Analysis

A Retrieval-Augmented Generation system that ingests PDFs, standalone images, and
plain text; extracts and indexes text, tables, and images into a shared vector
space; retrieves across all three modalities for a single text query; and
generates a visually-grounded answer with accurate source references, exposed
through a FastAPI REST API.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for system design, data flow, and
technology-choice rationale.

## Contents

- `src/` — the application (ingestion, embeddings, vector store, retrieval, generation, API)
- `sample_documents/` — 13 diverse test documents (PDFs, standalone images, text files)
- `scripts/generate_sample_documents.py` — regenerates `sample_documents/` from scratch
- `tests/` — unit tests + an end-to-end API integration test
- `notebooks/evaluation.ipynb` — retrieval/generation evaluation (hit rate, MRR, latency)
- `submission.yml` — automated `setup` / `test` commands

## 1. Setup

### Requirements

- Python 3.10+
- `tesseract-ocr` system binary (for OCR on images and embedded PDF images)
- ~10 seconds of disk space for the vector index; no GPU required

### Install

```bash
# clone repository
git clone https://github.com/prabhasupriya/-Multimodal_RAG_system_for_document_and_image_analysis.git
# System dependency (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Python dependencies (fast install, no GPU/heavy ML libs required)
pip install -r requirements.txt


pip install -r requirements-full.txt
```

### Configure (optional)

```bash
cp .env.example .env
```

By default the system runs **fully offline** — no API keys required:

- **Embeddings**: by default the system uses a lightweight, dependency-free
  fallback embedder (OCR-text + pixel signature) so install and startup stay
  fast, including in network-restricted environments. If you `pip install -r
  requirements-full.txt` (adds `sentence-transformers` + `torch`), the
  system automatically switches to a real CLIP model (`clip-ViT-B-32`),
  which places text and images in one true shared vector space. Nothing
  else to configure — this is auto-detected at startup. See
  `src/embeddings/model_loader.py`.
- **Generation**: if you set `OPENAI_API_KEY` in `.env`, the system uses a
  real GPT-4V-class model (`gpt-4o` by default, override with `VLM_MODEL`)
  and sends retrieved images to it directly for true visual reasoning. If no
  key is set, it falls back to a deterministic `ExtractiveGenerator` that
  still produces grounded, source-cited answers from the retrieved context
  (see `src/generation/generator.py`) — so the whole system is gradeable
  without any secrets.

## 2. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

### Ingest the sample documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "sample_documents"}'
```

```json
{
  "files_processed": 13,
  "files_failed": 0,
  "chunks_indexed": 30,
  "total_indexed": 30
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the quarterly sales chart show?", "top_k": 5}'
```

```json
{
  "answer": "As shown in the image on page 1 of quarterly_sales_chart.png, the visible text/labels read: \"Quarterly Sales (2023, $M)\" ...",
  "sources": [
    {"document_id": "quarterly_sales_chart.png", "page_number": 1, "content_type": "image", "snippet": "sample_documents/images/quarterly_sales_chart.png"},
    {"document_id": "annual_report_2023.pdf", "page_number": 1, "content_type": "text", "snippet": "Acme Robotics closed fiscal year 2023 with total revenue of $48.0 million..."}
  ],
  "latency_seconds": 0.41
}
```

Other example questions to try against the sample corpus:

- "What is the maximum payload of the AcmeBot X1?" (text)
- "How many near-misses were reported in Q4 2023?" (text + table)
- "Describe the pipeline stages in the system architecture diagram." (image)
- "How does the X1's battery life compare across generations, and how does it stack up against competitors?" (text + image + table synthesis)

### Other endpoints

| Method | Path      | Description                              |
|--------|-----------|-------------------------------------------|
| GET    | `/health` | Liveness check + current index size       |
| POST   | `/ingest` | Ingest all supported files in a directory |
| GET    | `/stats`  | Current index size                        |
| POST   | `/query`  | Ask a question, get an answer + sources   |

## 3. Run the tests

```bash
pytest -v
```

`tests/test_ingestion.py` covers parsing/chunking/table-extraction units.
`tests/test_api.py` is an end-to-end integration test: it boots the real
FastAPI app, ingests `sample_documents/`, and asks several multimodal
questions, asserting on answer content, source accuracy, cross-modal
retrieval, and the 15-second latency budget.

## 4. Run the evaluation notebook

```bash
jupyter notebook evaluation.ipynb
```

or run it non-interactively:

```bash
jupyter nbconvert --to notebook --execute evaluation.ipynb --output evaluation_out.ipynb
```

It ingests the sample corpus, runs a curated set of labeled multimodal
questions through the retriever, and reports **Hit Rate@k** and **MRR**
against the expected (document, content_type) sources, plus end-to-end
generation latency.

## 5. Regenerate the sample documents (optional)

```bash
python scripts/generate_sample_documents.py
```

This rebuilds `sample_documents/` (3 text files, 4 standalone chart/diagram
images, 6 PDFs mixing text + tables + embedded images — 13 files total, all
about a fictional company, "Acme Robotics," so questions can span multiple
documents and modalities consistently).

## Notes on the offline fallback modes

This project is designed to be **fully functional and gradeable without
external network access or API keys**, while still providing real
production-grade paths (CLIP, GPT-4V) when credentials/network are
available:

| Stage | Production path | Offline fallback (default install) |
|---|---|---|
| Embeddings | CLIP (`sentence-transformers` + `torch`, via `requirements-full.txt`) — real shared text/image space | Hashed bag-of-words (text) + OCR-text/pixel signature (images), same space |
| Generation | OpenAI GPT-4V-class model (`gpt-4o`), real visual reasoning | Deterministic extractive generator, still source-grounded |

Both fallbacks are activated automatically — there is nothing to configure —
and the API's request/response contract, source references, and latency
behavior are identical either way. See `ARCHITECTURE.md` for details.
## youtude video 
[watch here](https://youtu.be/xxQnVlfF-OE?si=qJMChjuORPzt2VnK)
## live deployment 
[click here](https://multimodal-rag-system-for-document-and.onrender.com/docs)


