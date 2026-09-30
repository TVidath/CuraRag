"""
Cross-Encoder Reranker Module for MedQuAD RAG Application.

Uses CrossEncoder models (e.g., BAAI/bge-reranker-base or cross-encoder/ms-marco-MiniLM-L-6-v2)
to score full (query, chunk_text) pairs and re-rank initial bi-encoder candidates.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Fix Windows PyTorch DLL loading (c10.dll / libiomp5md.dll)
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    import site
    for site_pkg in site.getsitepackages():
        torch_lib = os.path.join(site_pkg, "torch", "lib")
        if os.path.exists(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass

from sentence_transformers import CrossEncoder

from config.settings import (
    CROSS_ENCODER_MODEL_NAME,
    TOP_N_RERANKED_RESULTS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CrossEncoderReranker:
    """
    Reranks candidate child chunks using a Cross-Encoder model.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME):
        self.model_name = model_name
        logging.info(f"Initializing Cross-Encoder Reranker with model '{self.model_name}'...")
        self.model = CrossEncoder(self.model_name)
        logging.info("Cross-Encoder Reranker initialized successfully.")

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_n: int = TOP_N_RERANKED_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Calculates cross-encoder relevance scores for (query, chunk_text) pairs,
        attaches 'rerank_score' to each chunk, and returns top_n reranked chunks.
        """
        if not candidate_chunks:
            return []

        logging.info(f"Reranking {len(candidate_chunks)} candidate chunks for query: '{query}'...")

        # Build (query, chunk_text) sentence pairs for joint cross-attention scoring
        pairs = [(query, chunk["chunk_text"]) for chunk in candidate_chunks]

        # Compute cross-encoder relevance scores
        scores = self.model.predict(pairs)
        
        # Ensure scores is a list of floats (handling single element scalar array case)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (float, int)):
            scores = [float(scores)]

        # Attach rerank score and create updated chunk records
        scored_chunks: List[Dict[str, Any]] = []
        for chunk, score in zip(candidate_chunks, scores):
            updated_chunk = dict(chunk)
            updated_chunk["rerank_score"] = round(float(score), 4)
            scored_chunks.append(updated_chunk)

        # Sort candidate chunks descending by rerank_score (higher = more relevant)
        reranked = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)
        top_reranked = reranked[:top_n]

        logging.info(f"Reranking complete. Extracted top {len(top_reranked)} chunks.")
        return top_reranked
