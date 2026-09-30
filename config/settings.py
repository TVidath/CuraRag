"""
Global Configuration Settings for MedQuAD RAG Application.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows PyTorch DLL loading (c10.dll / libiomp5md.dll)
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import ctypes
        import site
        for site_pkg in site.getsitepackages():
            torch_lib = os.path.join(site_pkg, "torch", "lib")
            if os.path.exists(torch_lib):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
        import torch  # Pre-initialize PyTorch DLLs on Windows
    except Exception:
        pass

# Load environment variables from .env file if present
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"
DOCS_DIR = BASE_DIR / "docs"

# Dataset Artifact Paths
RAW_DATASET_PATH = DATA_DIR / "medquad.csv"
SAMPLE_DATASET_PATH = SAMPLES_DIR / "medquad_sample_200.csv"
CANONICAL_DATASET_PATH = DATA_DIR / "medquad_canonical.jsonl"
PARENTS_DATASET_PATH = DATA_DIR / "medquad_parents.jsonl"
CHILDREN_DATASET_PATH = DATA_DIR / "medquad_children.jsonl"

# Processing & Cleaner Default Settings
RANDOM_SEED = 42
DEFAULT_FOCUS_AREA = "General Health"
DEFAULT_SOURCE = "MedQuAD"

# Dense Embedding Model Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
NORMALIZE_EMBEDDINGS = True

# Vector Store Storage Configurations
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medquad-dense-index")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Ollama & Pre-Retrieval Multi-Query Settings
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma4:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NUM_GENERATED_QUERIES = 2  # Total queries evaluated = 1 (user) + 2 (LLM variations) = 3
TOP_K_DEDUPLICATED_RESULTS = 10 # Candidate pool size before reranking

# Cross-Encoder Reranker Configurations
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-base")
TOP_K_CANDIDATE_RESULTS = 10   # Initial bi-encoder candidate pool size
TOP_N_RERANKED_RESULTS = 3     # Final cross-encoder reranked top results count

# Output Guardrails Configurations
LLAMA_JUDGE_MODEL_NAME = os.getenv("LLAMA_JUDGE_MODEL_NAME", OLLAMA_MODEL_NAME)
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))


