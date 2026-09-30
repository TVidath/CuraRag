"""
Parent-Child Multi-Query Vector Retriever Module with Cross-Encoder Reranking

Combines:
1. Multi-Query Generation (3 queries total: user query + 2 LLM variations via gemma4:latest).
2. Dense Vector Index Search across all 3 queries (Bi-Encoder candidate pool generation).
3. Candidate Child Chunk Deduplication.
4. Cross-Encoder Reranking (BAAI/bge-reranker-base deep joint attention scoring).
5. Complete Parent Document Context Fetching from data/medquad_parents.jsonl.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional

from config.settings import (
    PARENTS_DATASET_PATH,
    TOP_K_CANDIDATE_RESULTS,
    TOP_N_RERANKED_RESULTS
)
from src.data.schema import ParentDocument
from src.vectorstore.vector_index import DenseVectorIndex
from src.retrieval.multi_query import MultiQueryGenerator
from src.retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ParentChildRetriever:
    """
    Parent-Child Retriever executing multi-query vector search, cross-encoder reranking, and parent answer assembly.
    """

    def __init__(
        self,
        parents_path: str = str(PARENTS_DATASET_PATH),
        vector_index: Optional[DenseVectorIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.parents_path = parents_path
        self.vector_index = vector_index or DenseVectorIndex()
        self.query_generator = MultiQueryGenerator()
        self.reranker = reranker or CrossEncoderReranker()
        self.parents_store: Dict[str, ParentDocument] = {}
        self._load_parents_store()

    def _load_parents_store(self):
        """Loads parent document records into memory dictionary keyed by parent_id."""
        if not os.path.exists(self.parents_path):
            logging.warning(f"Parent dataset file not found at '{self.parents_path}'. Parent document lookup will be unavailable.")
            return

        logging.info(f"Loading parent documents from '{self.parents_path}'...")
        count = 0
        with open(self.parents_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = ParentDocument.from_dict(json.loads(line))
                    self.parents_store[doc.parent_id] = doc
                    count += 1
        logging.info(f"Loaded {count:,} parent documents into retriever memory.")

    def retrieve(
        self,
        user_query: str,
        top_k_candidates: int = TOP_K_CANDIDATE_RESULTS,
        top_n_final: int = TOP_N_RERANKED_RESULTS
    ) -> Dict[str, Any]:
        """
        Executes Two-Stage Reranked Multi-Query Retrieval:
        1. Multi-Query Generation: 3 query variations (user query + 2 LLM variations).
        2. Bi-Encoder Vector Search: Collects dense candidates for all 3 queries.
        3. Deduplication: Keeps best candidate hit per child_id (Top-K pool, default 10).
        4. Cross-Encoder Reranking: Rescores candidate pool against initial user query.
        5. Selects top N reranked child chunks (default Top 3).
        6. Parent Assembly: Fetches complete Parent Document contexts for top reranked chunks.
        """
        # Step 1: Generate 3 query variations
        all_queries = self.query_generator.generate_queries(user_query)
        logging.info(f"Evaluating {len(all_queries)} queries for bi-encoder retrieval: {all_queries}")

        # Step 2: Query vector index for each query variation
        candidates_by_child_id: Dict[str, Dict[str, Any]] = {}

        for q in all_queries:
            raw_hits = self.vector_index.query_nearest_neighbors(q, top_k=top_k_candidates)
            for hit in raw_hits:
                cid = hit["child_id"]
                score = hit["similarity_score"]
                
                # Deduplicate: Keep hit with best (lowest distance / highest similarity) score
                if cid not in candidates_by_child_id:
                    candidates_by_child_id[cid] = hit
                    candidates_by_child_id[cid]["matched_query"] = q
                else:
                    if score < candidates_by_child_id[cid]["similarity_score"]:
                        candidates_by_child_id[cid] = hit
                        candidates_by_child_id[cid]["matched_query"] = q

        # Step 3: Extract deduplicated candidate pool (Top K)
        deduplicated_candidates = sorted(candidates_by_child_id.values(), key=lambda x: x["similarity_score"])
        candidate_pool = deduplicated_candidates[:top_k_candidates]
        logging.info(f"Deduplicated bi-encoder candidate pool size: {len(candidate_pool)} chunks")

        # Step 4: Perform Cross-Encoder Reranking over candidate pool using original user_query
        top_child_chunks = self.reranker.rerank(
            query=user_query,
            candidate_chunks=candidate_pool,
            top_n=top_n_final
        )

        # Step 5: Fetch complete Parent Document contexts for top reranked child chunks
        parent_documents: List[Dict[str, Any]] = []
        seen_parent_ids = set()

        for chunk in top_child_chunks:
            pid = chunk["parent_id"]
            if pid not in seen_parent_ids and pid in self.parents_store:
                seen_parent_ids.add(pid)
                p_doc = self.parents_store[pid]
                parent_documents.append(p_doc.to_dict())

        return {
            "user_query": user_query,
            "generated_queries": all_queries[1:],  # The 2 LLM generated variations
            "total_queries_searched": len(all_queries),
            "biencoder_candidate_pool_size": len(candidate_pool),
            "top_child_chunks": top_child_chunks,   # Final reranked top chunks
            "parent_documents": parent_documents
        }

