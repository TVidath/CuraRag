"""
End-to-End MedQuAD RAG Pipeline CLI Script

Executes full RAG query workflow:
1. Multi-Query Pre-Retrieval Expansion (gemma4:latest)
2. Dense Vector Candidate Search (all-MiniLM-L6-v2)
3. Candidate Child Chunk Deduplication
4. Cross-Encoder Reranking (bge-reranker-base)
5. Parent Document Context Lookup
6. Gemma 4 Grounded Answer Synthesis (ChatOllama)

Usage:
  python scripts/run_rag_pipeline.py "What are the symptoms of Glaucoma?"
"""

import os
import sys
from typing import Optional

# Ensure Windows terminal outputs UTF-8 characters without UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import MedQuADRAGPipeline


def run_pipeline(query_text: Optional[str] = None):
    print("=" * 80)
    print("      MEDQUAD END-TO-END RAG PIPELINE (RETRIEVAL + GEMMA 4 SYNTHESIS)       ")
    print("=" * 80)

    if not query_text:
        query_text = "What are the symptoms of Glaucoma?"

    print(f"\n[INPUT QUERY]: {repr(query_text)}")
    print("\nExecuting RAG Pipeline (Pre-Retrieval -> Vector Search -> Rerank -> Gemma 4)...")
    print("-" * 80)

    pipeline = MedQuADRAGPipeline()
    result = pipeline.query(user_query=query_text)

    print("\n1. MULTI-QUERY VARIATIONS (gemma4:latest):")
    print("-" * 60)
    print(f"   Original Query : {result['query']}")
    for idx, var in enumerate(result.get("generated_query_variations", []), 1):
        print(f"   Variation {idx}    : {repr(var)}")

    print("\n2. RETRIEVED PARENT DOCUMENT CONTEXTS:")
    print("-" * 60)
    print(f"   Total Parent Documents Retrieved: {result['retrieved_parent_count']}")
    for idx, doc in enumerate(result.get("parent_documents", []), 1):
        print(f"   [{idx}] Focus: {doc.get('focus')} | Source: {doc.get('source')}")
        print(f"       Question: {repr(doc.get('question'))}")

    print("\n3. GEMMA 4 GROUNDED CLINICAL RESPONSE:")
    print("=" * 80)
    print(result.get("answer", "No answer generated."))
    print("=" * 80)

    print(f"\nExecution Completed in {result['execution_time_seconds']}s")
    print("=" * 80)


if __name__ == "__main__":
    user_query = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(user_query)
