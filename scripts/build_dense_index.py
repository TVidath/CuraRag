"""
MedQuAD Dense Embedding Index Builder Script

Reads child chunks from data/medquad_children.jsonl, computes dense vector embeddings
using HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2), and indexes them
into local persistent ChromaDB (or cloud Pinecone if PINECONE_API_KEY is configured).
"""

import json
import os
import sys
from typing import Optional

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import CHILDREN_DATASET_PATH, CHROMA_PERSIST_DIR, PINECONE_API_KEY
from src.data.schema import ChildChunk
from src.vectorstore.vector_index import DenseVectorIndex


def rebuild_index(batch_size: int = 500, max_chunks: Optional[int] = None):
    print("=" * 80)
    print("                 MEDQUAD DENSE EMBEDDING INDEX BUILDER                       ")
    print("=" * 80)

    if not os.path.exists(CHILDREN_DATASET_PATH):
        print(f"Error: Child dataset artifact not found at '{CHILDREN_DATASET_PATH}'. Please run scripts/run_chunking_pipeline.py first.")
        sys.exit(1)

    print(f"\n1. READING CHILD CHUNKS FROM ARTIFACT: '{CHILDREN_DATASET_PATH}'...")
    child_chunks = []
    with open(CHILDREN_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                child_chunks.append(ChildChunk.from_dict(json.loads(line)))

    if max_chunks:
        print(f"   - Capping index build to first {max_chunks:,} child chunks for quick verification...")
        child_chunks = child_chunks[:max_chunks]
    else:
        print(f"   - Loaded {len(child_chunks):,} total child chunks for indexing.")

    print("\n2. INITIALIZING EMBEDDING MODEL & VECTOR INDEX BACKEND...")
    index_mgr = DenseVectorIndex()

    print("\n3. GENERATING DENSE EMBEDDINGS & BUILDING INDEX...")
    report = index_mgr.build_index(child_chunks, batch_size=batch_size)

    print("\n4. INDEX BUILD SUMMARY REPORT:")
    print("-" * 60)
    print(f"   Backend Storage          : {report['backend']}")
    print(f"   Embedding Model          : {report['embedding_model']}")
    print(f"   Vector Dimensions        : {report['dimensions']}")
    print(f"   Total Chunks Indexed     : {report['indexed_count']:,}")
    if not bool(PINECONE_API_KEY.strip()):
        print(f"   Persisted Local Path     : '{CHROMA_PERSIST_DIR}'")
    print("-" * 60)

    print("\n" + "=" * 80)
    print("               DENSE EMBEDDING INDEX REBUILT SUCCESSFULLY                   ")
    print("=" * 80)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    rebuild_index(max_chunks=limit)
