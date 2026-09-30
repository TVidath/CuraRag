"""
MedQuAD Dense Index Smoke Test Script (Nearest-Neighbor Vector Search)

Demonstrates nearest-neighbor similarity search over the built dense vector index
for a user-entered medical query.
"""

import os
import sys
from typing import Optional

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vectorstore.vector_index import DenseVectorIndex


def smoke_test_search(query_text: Optional[str] = None, top_k: int = 3):
    print("=" * 80)
    print("           MEDQUAD DENSE VECTOR INDEX SEARCH (SMOKE TEST)                    ")
    print("=" * 80)

    if not query_text:
        query_text = "What are the symptoms and causes of Glaucoma?"

    print(f"\nQUERY INPUT: {repr(query_text)}")
    print(f"FETCHING TOP-{top_k} NEAREST NEIGHBORS VIA COSINE SIMILARITY...")
    print("-" * 80)

    index_mgr = DenseVectorIndex()
    results = index_mgr.query_nearest_neighbors(query_text, top_k=top_k)

    for rank, item in enumerate(results, 1):
        meta = item["metadata"]
        print(f"\n[RANK {rank}] Similarity Score: {item['similarity_score']:.4f}")
        print(f"  Child ID   : {item['child_id']}")
        print(f"  Parent ID  : {item['parent_id']}")
        print(f"  Source     : {meta['source']} | Focus: {meta['focus']}")
        print(f"  Intent     : {meta['question_type']}")
        print(f"  Question   : {repr(meta['question'])}")
        print(f"  Matched Text: {repr(item['chunk_text'][:160])}...")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("                    SMOKE TEST COMPLETED SUCCESSFULLY                       ")
    print("=" * 80)


if __name__ == "__main__":
    query_input = sys.argv[1] if len(sys.argv) > 1 else "What are the symptoms of Glaucoma?"
    smoke_test_search(query_text=query_input)
