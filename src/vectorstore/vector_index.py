"""
Dense Embedding Indexing Module for MedQuAD RAG System.

Generates dense vector embeddings for Child Chunks using HuggingFaceEmbeddings
(sentence-transformers/all-MiniLM-L6-v2, 384 dimensions, L2 normalized) and stores them in a vector index.
Supports both persistent local ChromaDB index and cloud Pinecone index.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple

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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Safe Pinecone import handling legacy vs modern SDK
HAS_PINECONE = False
try:
    import pinecone
    if hasattr(pinecone, "Pinecone"):
        from pinecone import Pinecone, ServerlessSpec
        HAS_PINECONE = True
except Exception:
    HAS_PINECONE = False

from config.settings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    NORMALIZE_EMBEDDINGS,
    CHROMA_PERSIST_DIR,
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT
)
from src.data.schema import ChildChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Constructs the exact HuggingFaceEmbeddings instance requested:
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )


class DenseVectorIndex:
    """
    Manages building, persisting, and querying the Dense Embedding Vector Index.
    """

    def __init__(self, persist_directory: Optional[str] = str(CHROMA_PERSIST_DIR), collection_name: str = "medquad_child_chunks"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = get_embedding_function()
        self.vectorstore: Optional[Chroma] = None
        self.use_pinecone = bool(PINECONE_API_KEY.strip()) and HAS_PINECONE

        if self.use_pinecone:
            logging.info(f"Pinecone API Key detected. Initializing Pinecone Index '{PINECONE_INDEX_NAME}'...")
            self._init_pinecone()
        else:
            logging.info(f"Initializing Local Persistent ChromaDB at '{self.persist_directory}'...")
            self._init_chroma()

    def _init_chroma(self):
        """Initializes persistent local ChromaDB instance."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def _init_pinecone(self):
        """Initializes cloud Pinecone index if API key is provided."""
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indices = [idx.name for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME not in existing_indices:
            logging.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (384 dimensions, cosine distance)...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
            )

        from langchain_community.vectorstores import Pinecone as LangChainPinecone
        self.vectorstore = LangChainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )

    def build_index(self, child_chunks: List[ChildChunk], batch_size: int = 500) -> Dict[str, Any]:
        """
        Generates embeddings for child chunks and indexes them with metadata.
        
        Payload stored per vector:
        - id: child_id (e.g. mq_5487404c1a6e_c0)
        - text: chunk_text
        - metadata: {parent_id, question, question_type, focus, source, start_char, end_char, token_count}
        """
        if not child_chunks:
            logging.warning("No child chunks provided for indexing.")
            return {"indexed_count": 0}

        total = len(child_chunks)
        logging.info(f"Starting batch dense indexing for {total:,} child chunks (batch size: {batch_size})...")

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for c in child_chunks:
            texts.append(c.chunk_text)
            ids.append(c.chunk_id)
            metadatas.append({
                "child_id": c.chunk_id,
                "parent_id": c.parent_id,
                "question": c.question,
                "question_type": c.question_type,
                "focus": c.focus,
                "source": c.source,
                "token_count": c.token_count,
                "start_char": c.start_char,
                "end_char": c.end_char
            })

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            self.vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logging.info(f"Indexed batch {i//batch_size + 1} / {(total + batch_size - 1)//batch_size} ({len(batch_texts)} chunks)")

        logging.info(f"Dense vector indexing complete! Total indexed: {total:,} child chunks.")
        return {
            "indexed_count": total,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSION,
            "backend": "Pinecone" if self.use_pinecone else "ChromaDB Local"
        }

    def query_nearest_neighbors(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs nearest-neighbor similarity search for a query string.
        Returns matched child chunks with cosine similarity scores and metadata.
        """
        if not self.vectorstore:
            raise ValueError("Vector store index is not initialized.")

        results = self.vectorstore.similarity_search_with_score(query_text, k=top_k)
        
        output: List[Dict[str, Any]] = []
        for doc, score in results:
            output.append({
                "child_id": doc.metadata.get("child_id", doc.metadata.get("id")),
                "parent_id": doc.metadata.get("parent_id"),
                "similarity_score": round(float(score), 4),
                "chunk_text": doc.page_content,
                "metadata": {
                    "question": doc.metadata.get("question"),
                    "question_type": doc.metadata.get("question_type"),
                    "focus": doc.metadata.get("focus"),
                    "source": doc.metadata.get("source"),
                    "token_count": doc.metadata.get("token_count")
                }
            })
        return output
