"""
MedQuAD Parent-Child Chunking Pipeline Runner Script

Executes the Parent-Child Chunking Engine over the canonical MedQuAD dataset,
prints summary metrics, displays 2-3 detailed data flow examples, and exports
the processed Parent and Child datasets to local artifacts for downstream vector indexing.
"""

import json
import os
import sys

# Add root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import CANONICAL_DATASET_PATH, PARENTS_DATASET_PATH, CHILDREN_DATASET_PATH
from src.data.schema import CanonicalDocument, ParentDocument, ChildChunk
from src.data.chunker import chunk_canonical_dataset


def run_pipeline():
    print("=" * 80)
    print("                 MEDQUAD PARENT-CHILD CHUNKING PIPELINE                     ")
    print("=" * 80)

    if not os.path.exists(CANONICAL_DATASET_PATH):
        print(f"Error: Canonical dataset file not found at '{CANONICAL_DATASET_PATH}'. Please run scripts/run_cleaning_pipeline.py first.")
        sys.exit(1)

    print(f"\n1. LOADING CANONICAL DATASET: '{CANONICAL_DATASET_PATH}'...")
    canonical_docs = []
    with open(CANONICAL_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                canonical_docs.append(CanonicalDocument.from_dict(json.loads(line)))

    print(f"   - Loaded {len(canonical_docs):,} canonical documents.")

    print("\n2. EXECUTING PARENT-CHILD CHUNKING ENGINE...")
    parents, children, report = chunk_canonical_dataset(canonical_docs)

    print("\n3. CHUNKING PIPELINE REPORT & METRICS:")
    print("-" * 60)
    print(f"   Total Parent Documents       : {report.total_parents:,}")
    print(f"   Total Child Chunks           : {report.total_children:,}")
    print(f"   Single-Child Parents (<=150t): {report.single_child_parents:,} ({report.single_child_parents / report.total_parents * 100:.2f}%)")
    print(f"   Multi-Child Parents (>150t)  : {report.multi_child_parents:,} ({report.multi_child_parents / report.total_parents * 100:.2f}%)")
    print(f"   Average Children per Parent  : {report.avg_children_per_parent}")
    print(f"   Average Child Token Count    : {report.avg_child_token_count} tokens")
    print(f"   Average Child Char Length    : {report.avg_child_char_length} chars")
    print("-" * 60)

    print("\n4. REAL DATA FLOW EXAMPLES (MedQuAD -> Canonical -> Parent -> Children):")
    print("=" * 80)

    # Example 1: Single Child Parent (Short document <= 150 tokens)
    short_parents = [p for p in parents if len(p.child_ids) == 1]
    if short_parents:
        p_short = short_parents[0]
        child_short = [c for c in children if c.parent_id == p_short.parent_id][0]
        print("\n[EXAMPLE 1: SHORT DOCUMENT <= 150 TOKENS (1 Parent -> 1 Child)]")
        print(f"  Parent ID    : {p_short.parent_id}")
        print(f"  Question     : {repr(p_short.question)}")
        print(f"  Source/Focus : {p_short.source} | {p_short.focus}")
        print(f"  Parent Answer: {repr(p_short.answer[:120])}...")
        print(f"  Child Chunks : {len(p_short.child_ids)} child chunk")
        print(f"    - Child ID : {child_short.chunk_id}")
        print(f"    - Snippet  : {repr(child_short.chunk_text[:120])}...")
        print(f"    - Tokens   : {child_short.token_count} tokens | Metadata: {child_short.source} / {child_short.question_type}")
        print("-" * 80)

    # Example 2: Multi Child Parent (Long document > 150 tokens)
    multi_parents = [p for p in parents if len(p.child_ids) >= 3]
    if multi_parents:
        p_multi = multi_parents[0]
        c_multi = [c for c in children if c.parent_id == p_multi.parent_id]
        print("\n[EXAMPLE 2: LONG DOCUMENT > 150 TOKENS (1 Parent -> Multiple Children)]")
        print(f"  Parent ID    : {p_multi.parent_id}")
        print(f"  Question     : {repr(p_multi.question)}")
        print(f"  Source/Focus : {p_multi.source} | {p_multi.focus}")
        print(f"  Parent Answer: {repr(p_multi.answer[:150])}...")
        print(f"  Child Chunks : {len(p_multi.child_ids)} child chunks generated")
        for idx, child in enumerate(c_multi[:3], 1):
            print(f"    [{idx}] Child ID: {child.chunk_id}")
            print(f"        Snippet : {repr(child.chunk_text[:120])}...")
            print(f"        Range   : chars {child.start_char}-{child.end_char} | {child.token_count} tokens")
        print("-" * 80)

    print("\n5. SAVING PROCESSED ARTIFACTS TO DISK...")

    # Save Parents Dataset
    with open(PARENTS_DATASET_PATH, "w", encoding="utf-8") as f:
        for parent in parents:
            f.write(json.dumps(parent.to_dict(), ensure_ascii=False) + "\n")
    print(f"   - Saved Parent Documents: '{PARENTS_DATASET_PATH}' ({os.path.getsize(PARENTS_DATASET_PATH)/(1024*1024):.2f} MB)")

    # Save Children Dataset
    with open(CHILDREN_DATASET_PATH, "w", encoding="utf-8") as f:
        for child in children:
            f.write(json.dumps(child.to_dict(), ensure_ascii=False) + "\n")
    print(f"   - Saved Child Chunks:    '{CHILDREN_DATASET_PATH}' ({os.path.getsize(CHILDREN_DATASET_PATH)/(1024*1024):.2f} MB)")

    print("\n" + "=" * 80)
    print("                PARENT-CHILD CHUNKING COMPLETED SUCCESSFULLY                ")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
