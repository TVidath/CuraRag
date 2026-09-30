# MedQuAD Parent-Child Chunking Design Document

## 1. Problem Statement

In Retrieval-Augmented Generation (RAG), there is a fundamental tension between **retrieval accuracy** and **generation completeness**:

* **Small Chunks (e.g., 100-150 tokens)** produce sharp, dense vector embeddings. They excel at matching granular user queries. However, when passed to an LLM, small chunks lack surrounding context, causing the generator to produce incomplete or hallucinated answers.
* **Large Documents (e.g., 800-4,000 tokens)** contain complete clinical context. However, embedding an entire 2,000-word document dilutes fine-grained semantic details, leading to poor vector similarity scores during retrieval.

In MedQuAD, answer lengths vary wildly:
- **33.2%** of answers are $\le 150$ tokens.
- **42.9%** are between 151 and 400 tokens.
- **20.5%** are between 401 and 1,000 tokens.
- **3.4%** exceed 1,000 tokens (up to 6,750 tokens!).

**Goal**: Decouple the **search payload** (small child chunks) from the **llm context payload** (full parent document), ensuring optimal vector matching without sacrificing clinical context completeness.

---

## 2. Parent Definition

A **Parent Document** represents the complete, canonical medical Q&A record:
* **Payload**: `document_id`, `question`, `answer`, `source`, `focus`, `question_type`, `char_length`, `word_count`.
* **Role**: Stored in a document storage dictionary or key-value store. It is **never** embedded directly into the primary vector index for top-k matching. Instead, when a child chunk hits top-k vector similarity, its Parent Document is retrieved and injected into the LLM context prompt.
* **Maximum Size Cap**: For ultra-long documents (>2,500 tokens / 10,000 characters, <1% of dataset), parent answers are truncated or section-capped during context assembly to protect LLM context windows.

---

## 3. Child Definition

A **Child Chunk** is a small, focused passage extracted from the parent answer text:
* **Payload**: `chunk_id` (`{parent_id}_c{index}`), `parent_id`, `chunk_text`, `start_char`, `end_char`, `token_count`, plus inherited metadata (`source`, `focus`, `question_type`).
* **Role**: Embedded using a dense vector embedding model and indexed in the vector database for top-k similarity search.
* **Relationship**: Every child contains a reference to `parent_id`. Multiple child chunks map to 1 parent document.

---

## 4. Chunking Algorithm

We employ a **Recursive Sentence-Aware Parent-Child Chunking Algorithm**:

```
                              Canonical Document
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      Token Count <= 150?                           Token Count > 150?
               │                                             │
               ▼                                             ▼
       [Single Child Chunk]                       [Recursive Sentence Split]
   1 Child = 1 Parent text.                   Separators: ["\n\n", "\n", ". ", " "]
   Child vectors = Parent vector.             Target: 150 tokens, Overlap: 30 tokens
                                              Discard chunks < 25 tokens.
```

### Step-by-step Execution:
1. **Input**: A `CanonicalDocument` object.
2. **Evaluate Document Size**:
   - If total estimated answer tokens $\le 150$ tokens: Create exactly **1 child chunk** containing the full answer text.
   - If total answer tokens $> 150$ tokens: Recursively split `answer` text using separators `["\n\n", "\n", ". ", " "]`.
3. **Enforce Overlap & Boundaries**: Move a sliding window of target size **150 tokens** with **30 tokens overlap**, snapping boundaries to sentence endings.
4. **Filter Noise**: Discard any trailing child chunk smaller than **25 tokens** (e.g. lone copyright strings).
5. **Assign Metadata**: Attach `parent_id` (`document_id`), `chunk_id`, `focus`, `source`, and `question_type` to every child chunk.

---

## 5. Parameters & Empirical Justification

| Parameter | Recommended Value | Empirical Justification from MedQuAD Dataset |
| :--- | :--- | :--- |
| **Child Chunk Size** | **150 tokens** (~600 chars / ~100 words) | **33.2%** of MedQuAD documents are naturally $\le 150$ tokens. 150 tokens fits 2–4 medical sentences, creating dense, un-diluted vector embeddings. |
| **Child Overlap** | **30 tokens** (~120 chars / ~20 words) | Provides 15–20% overlap across chunk boundaries, ensuring clinical facts spanning sentence boundaries are captured. |
| **Minimum Child Size** | **25 tokens** (~100 chars / ~15 words) | Discards noisy trailing web headers or copyright lines (e.g. *"Learn more at NIH.gov"*). |
| **Parent Max Cap** | **2,500 tokens** (~10,000 chars) | Caps the top 0.5% extreme outlier answers (>10,000 chars) to prevent prompt overflow. |
| **Separators** | `["\n\n", "\n", ". ", " ", ""]` | Respects paragraph breaks and medical sentence boundaries. Never breaks mid-word or mid-number. |

---

## 6. Real MedQuAD Example

### Raw Canonical Document:
* **ID**: `mq_7c78d0aedf1e`
* **Source**: `NIHSeniorHealth`
* **Focus**: `Urinary Tract Infections`
* **Question**: `How to prevent Urinary Tract Infections ?`
* **Raw Answer Length**: 2,678 characters (~460 words / ~670 tokens)

---

### Parent Document Payload:
```json
{
  "document_id": "mq_7c78d0aedf1e",
  "question": "How to prevent Urinary Tract Infections ?",
  "answer": "Changing some of these daily habits may help prevent UTIs. - Wipe from front to back after using the toilet. Women should wipe from front to back to keep bacteria from getting into the urinary tract... [Full 2,678 char medical answer text] ...",
  "source": "NIHSeniorHealth",
  "focus": "Urinary Tract Infections",
  "question_type": "Prevention & Risk Reduction"
}
```

---

### Extracted Child Chunks (Demonstration):

#### Child Chunk 1 (`mq_7c78d0aedf1e_c0`):
```json
{
  "chunk_id": "mq_7c78d0aedf1e_c0",
  "parent_id": "mq_7c78d0aedf1e",
  "chunk_text": "Changing some of these daily habits may help prevent UTIs. - Wipe from front to back after using the toilet. Women should wipe from front to back to keep bacteria from getting into the urinary tract. - Drink plenty of liquids, especially water.",
  "token_count": 48,
  "start_char": 0,
  "end_char": 245
}
```

#### Child Chunk 2 (`mq_7c78d0aedf1e_c1`) [With 30-token overlap]:
```json
{
  "chunk_id": "mq_7c78d0aedf1e_c1",
  "parent_id": "mq_7c78d0aedf1e",
  "chunk_text": "- Drink plenty of liquids, especially water. Drinking plenty of water helps flush bacteria out of your urinary tract. - Urinate often and when you feel the need. Try to urinate at least every 3 to 4 hours.",
  "token_count": 44,
  "start_char": 198,
  "end_char": 412
}
```

---

## 7. Trade-offs

| Advantage | Disadvantage | Mitigation |
| :--- | :--- | :--- |
| **High Retrieval Precision**: Small 150-token child vectors match specific queries accurately. | **Storage Overhead**: Must maintain both a Vector Store (children) and a Key-Value Store (parents). | Store parent text in lightweight JSONL / SQLite or in-memory dictionary. |
| **Zero LLM Context Loss**: LLM receives the complete parent medical answer. | **Prompt Token Consumption**: Injecting full parent answers consumes more LLM context window tokens. | Apply Parent Max Cap (2,500 tokens) and deduplicate parents when multiple children hit the same parent. |
| **Eliminates Snippet Truncation**: No broken sentences or incomplete list items in LLM generation. | **Index Complexity**: Requires mapping `child -> parent_id`. | Encapsulate mapping cleanly in the retrieval engine. |

---

## 8. Expected Effect on Retrieval

1. **Hit Rate @ K Improvement**: Small child chunks produce higher cosine similarity scores for specific search queries compared to embedding multi-paragraph documents.
2. **Elimination of Hallucinations**: Passing full parent answers ensures the generator receives complete clinical context (all bullet points and dosage caveats), eliminating context-fragmentation hallucinations.
3. **No Overhead for Short Answers**: Because **33.2%** of MedQuAD records are $\le 150$ tokens, 1 out of every 3 documents produces a 1-to-1 parent-child mapping, minimizing storage and computation overhead.
