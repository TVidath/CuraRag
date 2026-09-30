# 🏥 CuraRAG Advanced Two-Stage Parent-Child RAG Application

An end-to-end medical Retrieval-Augmented Generation (RAG) system built on the **MedQuAD medical dataset**. It combines **Pre-Retrieval LLM Multi-Query Expansion**, **Bi-Encoder Dense Vector Search**, **Cross-Encoder Reranking**, and **Parent-Child Document Context Assembly** for high-precision clinical document retrieval.

---

## 🌟 Key Features Implemented

- **Data Processing & Normalization**: Automated cleaning pipeline that drops invalid records, deduplicates Q&A pairs, imputes metadata, and generates deterministic document IDs.
- **Parent-Child Chunking**: Full medical answers are preserved as `ParentDocument` records while being split into smaller overlapping `ChildChunk` passages (600 chars, 120 char overlap) for precise vector matching.
- **Dense Embedding Vector Indexing**: Indexed using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2 normalized) with persistent local **ChromaDB** storage and cloud **Pinecone** support.
- **Pre-Retrieval Multi-Query Expansion**: Uses local LLM (`gemma4:latest` via Ollama and LangChain) to generate 2 alternative clinical query variations per prompt to improve retrieval recall.
- **Two-Stage Cross-Encoder Reranking**: Re-scores bi-encoder candidate pools using `BAAI/bge-reranker-base` joint cross-attention to deliver accurate top-ranked chunks.
- **Parent Context Assembly**: Automatically maps top-ranked child chunks back to their complete Parent Document context for downstream LLM answer generation.

---

## 🏗️ System Architecture

```text
                               ┌─────────────────────────────┐
                               │  User Input Medical Query   │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ Multi-Query Generator (LLM) │
                               └──────────────┬──────────────┘
                                              │ (3 Query Variations)
                                              ▼
                               ┌─────────────────────────────┐
                               │  Bi-Encoder Vector Search   │
                               │   (ChromaDB / MiniLM-L6-v2) │
                               └──────────────┬──────────────┘
                                              │ (10 Candidate Chunks)
                                              ▼
                               ┌─────────────────────────────┐
                               │    Cross-Encoder Reranker   │
                               │   (BAAI/bge-reranker-base)  │
                               └──────────────┬──────────────┘
                                              │ (Top 3 Reranked Chunks)
                                              ▼
                               ┌─────────────────────────────┐
                               │ Parent Document Assembly    │
                               │ (medquad_parents.jsonl)     │
                               └─────────────────────────────┘
```

---

## 📁 Repository Structure

```text
CuraRAG/
├── config/
│   └── settings.py                  # Global configurations (Model names, paths, top-k parameters)
├── data/
│   ├── medquad.csv                  # Raw MedQuAD CSV dataset
│   ├── medquad_canonical.jsonl      # Cleaned canonical JSONL dataset
│   ├── medquad_parents.jsonl        # Parent documents store
│   ├── medquad_children.jsonl       # Child chunks store
│   └── chroma_db/                   # Persistent ChromaDB vector database directory
├── scripts/
│   ├── run_cleaning_pipeline.py     # Executes data cleaning & canonical formatting
│   ├── run_chunking_pipeline.py     # Executes Parent-Child chunking engine
│   ├── build_dense_index.py         # Generates vector embeddings & builds ChromaDB index
│   ├── test_index_search.py         # Smoke test script for dense vector search
│   └── run_multi_query_search.py    # End-to-End CLI demo (Multi-Query + Cross-Encoder)
├── src/
│   ├── data/
│   │   ├── schema.py                # Dataclass schemas (CanonicalDocument, ParentDocument, ChildChunk)
│   │   ├── cleaner.py               # Data cleaner & deduplication functions
│   │   └── chunker.py               # Recursive text splitter & parent-child chunking
│   ├── vectorstore/
│   │   └── vector_index.py          # Dense vector indexing engine (ChromaDB / Pinecone)
│   └── retrieval/
│       ├── multi_query.py           # Pre-retrieval LLM query expansion (gemma4:latest)
│       ├── reranker.py              # Cross-Encoder reranker (BAAI/bge-reranker-base)
│       └── retriever.py             # ParentChildRetriever orchestrator
├── tests/
│   ├── test_cleaner.py              # Unittests for data cleaner
│   ├── test_chunker.py              # Unittests for parent-child chunker
│   ├── test_vectorstore.py          # Unittests for vector index search
│   ├── test_multi_query.py          # Unittests for multi-query generator
│   └── test_reranker.py             # Unittests for Cross-Encoder reranker
└── requirements.txt                 # Project dependencies
```

---

## ⚡ Quickstart & Usage

### 1. Installation

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Data Processing & Indexing Pipelines

Execute the pipeline scripts in sequence:

```bash
# Step 1: Clean raw dataset
python scripts/run_cleaning_pipeline.py

# Step 2: Generate Parent and Child datasets
python scripts/run_chunking_pipeline.py

# Step 3: Compute dense embeddings and build ChromaDB vector index
python scripts/build_dense_index.py
```

### 3. Run Two-Stage Retrieval Demo

Execute the full two-stage RAG retrieval pipeline:

```bash
python scripts/run_multi_query_search.py "What are the symptoms of Glaucoma?"
```

---

## 🧪 Running Unit Tests

Run the test suite using Python `unittest`:

```bash
python tests/test_cleaner.py
python tests/test_chunker.py
python tests/test_vectorstore.py
python tests/test_reranker.py
```

---

## ⚙️ Configuration Reference

Key settings in [`config/settings.py`](file:///c:/Users/k16na/Desktop/Medrag/config/settings.py):

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/all-MiniLM-L6-v2"` | Bi-encoder dense embedding model |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector dimensions |
| `CROSS_ENCODER_MODEL_NAME` | `"BAAI/bge-reranker-base"` | Cross-encoder reranking transformer |
| `OLLAMA_MODEL_NAME` | `"gemma4:latest"` | Local LLM for multi-query generation |
| `TOP_K_CANDIDATE_RESULTS` | `10` | Bi-encoder candidate over-fetching pool size |
| `TOP_N_RERANKED_RESULTS` | `3` | Final reranked top chunks output count |
