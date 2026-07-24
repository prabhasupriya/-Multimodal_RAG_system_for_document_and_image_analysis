"""
Builds evaluation.ipynb programmatically (kept in scripts/ so the notebook
itself can be regenerated deterministically). Run:

    python scripts/build_evaluation_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Multimodal RAG — Evaluation

This notebook ingests `sample_documents/` and evaluates **retrieval**
(Hit Rate@k, Mean Reciprocal Rank) and **generation** (latency, grounding)
against a small, curated set of labeled multimodal questions spanning text,
tables, and images.

It works fully offline (see `README.md` — embeddings/generation
automatically fall back to deterministic offline backends if CLIP/OpenAI
aren't reachable), so these results are reproducible with zero setup."""
))

cells.append(nbf.v4.new_code_cell(
"""import sys, time
sys.path.insert(0, "..") if not __import__("pathlib").Path("src").exists() else None

from src.vector_store.chroma_manager import ChromaManager
from src.pipeline import ingest_directory
from src.retrieval.retriever import Retriever
from src.generation.generator import get_generator

import shutil, tempfile
persist_dir = tempfile.mkdtemp()
image_dir = tempfile.mkdtemp()

store = ChromaManager(persist_dir=persist_dir)
stats = ingest_directory("sample_documents", store, image_output_dir=image_dir)
retriever = Retriever(store)
generator = get_generator()

print("Ingestion stats:", stats)
print("Generator backend:", type(generator).__name__)"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Labeled evaluation set

Each question is labeled with the set of **acceptable source documents**
(and, where relevant, the content type that should ground the answer) based
on how `scripts/generate_sample_documents.py` constructed the corpus. A
retrieval "hit" means at least one of the top-k results comes from an
acceptable `(document_id, content_type)` pair."""
))

cells.append(nbf.v4.new_code_cell(
"""EVAL_SET = [
    {
        "query": "What is the maximum payload of the AcmeBot X1?",
        "expected": [("product_faq.txt", "text"), ("product_spec_x1.pdf", "table"),
                     ("product_spec_x1.pdf", "text"), ("competitor_comparison.pdf", "table")],
    },
    {
        "query": "How long does the AcmeBot X1 battery last on a full charge?",
        "expected": [("product_faq.txt", "text"), ("product_spec_x1.pdf", "table"),
                     ("battery_life_trend.png", "image")],
    },
    {
        "query": "What does the quarterly sales chart show?",
        "expected": [("quarterly_sales_chart.png", "image"), ("annual_report_2023.pdf", "text"),
                     ("annual_report_2023.pdf", "table")],
    },
    {
        "query": "How many near-misses were reported each quarter of 2023?",
        "expected": [("safety_incident_summary.pdf", "table"), ("safety_incident_summary.pdf", "text")],
    },
    {
        "query": "Describe the pipeline stages shown in the system architecture diagram.",
        "expected": [("system_architecture_diagram.png", "image"),
                     ("system_architecture_whitepaper.pdf", "text"),
                     ("system_architecture_whitepaper.pdf", "table")],
    },
    {
        "query": "How many AcmeBot X1 units are deployed in North America vs Europe?",
        "expected": [("regional_deployment_chart.png", "image"), ("competitor_comparison.pdf", "text")],
    },
    {
        "query": "What safety clearance must autonomous robots maintain from pedestrians?",
        "expected": [("safety_policy.txt", "text")],
    },
    {
        "query": "How does the X1 compare to competitors on battery life and payload?",
        "expected": [("competitor_comparison.pdf", "table"), ("competitor_comparison.pdf", "text")],
    },
]
len(EVAL_SET)"""
))

cells.append(nbf.v4.new_markdown_cell("## Retrieval metrics: Hit Rate@k and MRR"))

cells.append(nbf.v4.new_code_cell(
"""def is_hit(item, expected):
    return (item["document_id"], item["content_type"]) in expected

def evaluate_retrieval(eval_set, top_k=5):
    rows = []
    for case in eval_set:
        hits = retriever.retrieve(case["query"], top_k=top_k)
        expected = set(case["expected"])
        hit_at_k = any(is_hit(h, expected) for h in hits)
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if is_hit(h, expected):
                rr = 1.0 / rank
                break
        rows.append({
            "query": case["query"],
            "hit@k": hit_at_k,
            "reciprocal_rank": rr,
            "top_content_types": [h["content_type"] for h in hits],
        })
    return rows

results = evaluate_retrieval(EVAL_SET, top_k=5)
hit_rate = sum(r["hit@k"] for r in results) / len(results)
mrr = sum(r["reciprocal_rank"] for r in results) / len(results)
print(f"Hit Rate@5: {hit_rate:.2%}")
print(f"MRR@5:      {mrr:.3f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
pd.DataFrame(results)[["query", "hit@k", "reciprocal_rank", "top_content_types"]]"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## End-to-end generation: latency + visual grounding check

For each query we also run full generation and check:
1. Latency is under the 15s budget required by the task spec.
2. At least one source is returned.
3. If any retrieved source is an image, the answer text should reference
   it (a lightweight proxy for "visually grounded" — looks for words like
   "image", "chart", "diagram", "figure", "shown")."""
))

cells.append(nbf.v4.new_code_cell(
"""GROUNDING_WORDS = ("image", "chart", "diagram", "figure", "shown", "visible")

gen_rows = []
for case in EVAL_SET:
    t0 = time.time()
    ctx = retriever.retrieve(case["query"], top_k=5)
    answer = generator.generate(case["query"], ctx)
    latency = time.time() - t0

    has_image_source = any(c["content_type"] == "image" for c in ctx)
    grounded = any(w in answer.lower() for w in GROUNDING_WORDS) if has_image_source else None

    gen_rows.append({
        "query": case["query"],
        "latency_s": round(latency, 3),
        "under_15s": latency < 15,
        "n_sources": len(ctx),
        "has_image_source": has_image_source,
        "visually_grounded_if_image": grounded,
    })

df = pd.DataFrame(gen_rows)
df"""
))

cells.append(nbf.v4.new_code_cell(
"""print(f"All under 15s budget: {df['under_15s'].all()}")
print(f"Mean latency: {df['latency_s'].mean():.3f}s  (max: {df['latency_s'].max():.3f}s)")
img_cases = df[df["has_image_source"]]
if len(img_cases):
    print(f"Visually grounded when an image source is present: "
          f"{img_cases['visually_grounded_if_image'].mean():.0%} of {len(img_cases)} cases")"""
))

cells.append(nbf.v4.new_markdown_cell("## Example: full answer with sources, for manual inspection"))

cells.append(nbf.v4.new_code_cell(
"""case = EVAL_SET[2]  # the quarterly sales chart question
ctx = retriever.retrieve(case["query"], top_k=5)
answer = generator.generate(case["query"], ctx)

print("Q:", case["query"])
print()
print("ANSWER:")
print(answer)
print()
print("SOURCES:")
for c in ctx:
    print(f"  - [{c['content_type']}] {c['document_id']} (page {c['page_number']}, score={c['score']:.3f})")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Cleanup

Removes the scratch ChromaDB / extracted-image directories created for this
evaluation run (the main `sample_documents/` corpus is untouched)."""
))

cells.append(nbf.v4.new_code_cell(
"""shutil.rmtree(persist_dir, ignore_errors=True)
shutil.rmtree(image_dir, ignore_errors=True)
print("Cleaned up scratch directories.")"""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("evaluation.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote evaluation.ipynb")
