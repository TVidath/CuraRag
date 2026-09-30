"""
MedQuAD Pre-Retrieval Multi-Query Search CLI Script

Demonstrates:
1. Generating 2 query variations using local LLM gemma4:latest in Ollama via LangChain.
2. Executing vector similarity search across all 3 queries.
3. Deduplicating results to return the TOP-3 best similarity score child chunks.
4. Assembling full Parent Document clinical answers for downstream LLM generation.
"""

import os
import sys
from typing import Optional

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import ParentChildRetriever


def run_multi_query_search(query_text: Optional[str] = None):
    print("=" * 80)
    print("    MEDQUAD TWO-STAGE MULTI-QUERY SEARCH + CROSS-ENCODER RERANKING          ")
    print("=" * 80)

    if not query_text:
        query_text = "What are the symptoms of Glaucoma?"

    print(f"\n1. INITIAL USER QUERY: {repr(query_text)}")
    print("\n2. EXECUTING TWO-STAGE RETRIEVAL (MULTI-QUERY BI-ENCODER + CROSS-ENCODER RERANKING)...")
    
    retriever = ParentChildRetriever()
    payload = retriever.retrieve(user_query=query_text, top_k_candidates=10, top_n_final=3)

    print("\n3. MULTI-QUERY & BI-ENCODER POOL SUMMARY:")
    print("-" * 60)
    print(f"   Original User Query : {payload['user_query']}")
    for idx, var_q in enumerate(payload['generated_queries'], 1):
        print(f"   LLM Variation {idx}     : {repr(var_q)}")
    print(f"   Total Queries Evaluated : {payload['total_queries_searched']}")
    print(f"   Bi-Encoder Pool Size    : {payload['biencoder_candidate_pool_size']} candidate chunks")
    print("-" * 60)

    print("\n4. TOP-3 CROSS-ENCODER RERANKED CHILD CHUNKS (BAAI/bge-reranker-base):")
    print("=" * 80)
    for rank, chunk in enumerate(payload["top_child_chunks"], 1):
        meta = chunk["metadata"]
        print(f"\n[RANK {rank}] Rerank Score: {chunk['rerank_score']:.4f} (Bi-Encoder Distance: {chunk['similarity_score']:.4f})")
        print(f"  Child ID      : {chunk['child_id']}")
        print(f"  Parent ID     : {chunk['parent_id']}")
        print(f"  Matched Query : {repr(chunk['matched_query'])}")
        print(f"  Source | Focus: {meta['source']} | {meta['focus']}")
        print(f"  Question      : {repr(meta['question'])}")
        print(f"  Matched Text  : {repr(chunk['chunk_text'][:160])}...")
        print("-" * 80)

    print("\n5. ASSEMBLED PARENT DOCUMENT CONTEXTS FOR GENERATOR:")
    print("=" * 80)
    for idx, parent in enumerate(payload["parent_documents"], 1):
        print(f"\n[PARENT {idx}] ID: {parent['parent_id']} | Focus: {parent['focus']}")
        print(f"  Question: {repr(parent['question'])}")
        print(f"  Full Answer ({len(parent['answer'])} chars): {repr(parent['answer'][:250])}...")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("         TWO-STAGE MULTI-QUERY RERANKED RETRIEVAL DEMO COMPLETED           ")
    print("=" * 80)


if __name__ == "__main__":
    query_input = sys.argv[1] if len(sys.argv) > 1 else "What are the symptoms of Glaucoma?"
    run_multi_query_search(query_text=query_input)
