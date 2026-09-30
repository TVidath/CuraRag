"""
MedQuAD Preprocessing & Cleaning Pipeline Runner Script

Executes the cleaning, validation, normalization, and stable ID assignment pipeline
over the raw MedQuAD dataset, prints a summary report, displays before/after
transformation examples, and saves the canonical dataset to disk.
"""

import json
import os
import sys
import pandas as pd

# Add root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import RAW_DATASET_PATH, CANONICAL_DATASET_PATH, CANONICAL_CSV_PATH
from src.data.cleaner import clean_dataset, clean_text, normalize_focus, normalize_source, generate_stable_doc_id


def run_pipeline():
    print("=" * 80)
    print("                 MEDQUAD DATA CLEANING & NORMALIZATION PIPELINE              ")
    print("=" * 80)

    if not os.path.exists(RAW_DATASET_PATH):
        print(f"Error: Raw dataset file not found at '{RAW_DATASET_PATH}'. Please run scripts/download_dataset.py first.")
        sys.exit(1)

    print(f"\n1. LOADING RAW DATASET: '{RAW_DATASET_PATH}'...")
    df_raw = pd.read_csv(RAW_DATASET_PATH)
    print(f"   - Loaded {len(df_raw):,} raw records.")

    print("\n2. EXECUTING DATA CLEANING & NORMALIZATION PIPELINE...")
    canonical_docs, report = clean_dataset(df_raw)

    print("\n3. PIPELINE METRICS & CLEANING REPORT:")
    print("-" * 60)
    print(f"   Raw Records Processed        : {report.raw_record_count:,}")
    print(f"   Missing Questions Dropped    : {report.missing_question_dropped:,}")
    print(f"   Missing Answers Dropped      : {report.missing_answer_dropped:,}")
    print(f"   Empty Strings Dropped        : {report.empty_string_dropped:,}")
    print(f"   Duplicate QA Pairs Dropped   : {report.duplicate_qa_dropped:,}")
    print(f"   Focus Imputed ('General')    : {report.focus_imputed_count:,}")
    print(f"   Source Imputed ('MedQuAD')   : {report.source_imputed_count:,}")
    print("-" * 60)
    print(f"   FINAL CANONICAL DOCUMENTS    : {report.canonical_record_count:,}")
    print(f"   Retained Quality Percentage  : {report.canonical_record_count / report.raw_record_count * 100:.2f}%")
    print("-" * 60)

    print("\n4. BEFORE vs. AFTER TRANSFORMATION EXAMPLES:")
    print("=" * 80)

    # Pick 3 specific representative examples for demonstration
    sample_indices = [0, 50, 100]
    for idx in sample_indices:
        raw_row = df_raw.iloc[idx]
        can_doc = canonical_docs[idx]

        print(f"\n[EXAMPLE {idx+1}] Stable Document ID: {can_doc.document_id}")
        print("  BEFORE (Raw CSV Record):")
        print(f"    - Question  : {repr(raw_row['question'])}")
        print(f"    - Answer    : {repr(str(raw_row['answer'])[:120])}...")
        print(f"    - Source    : {repr(raw_row['source'])}")
        print(f"    - Focus     : {repr(raw_row['focus_area'])}")

        print("  AFTER (Canonical Document):")
        print(f"    - Question  : {repr(can_doc.question)}")
        print(f"    - Answer    : {repr(can_doc.answer[:120])}...")
        print(f"    - Source    : {repr(can_doc.source)}")
        print(f"    - Focus     : {repr(can_doc.focus)}")
        print(f"    - Type      : {repr(can_doc.question_type)}")
        print(f"    - Stats     : {can_doc.char_length} chars | {can_doc.word_count} words")
        print("-" * 80)

    # Demonstration of Missing Focus Imputation
    null_focus_sample = df_raw[df_raw['focus_area'].isnull()]
    if len(null_focus_sample) > 0:
        raw_null_row = null_focus_sample.iloc[0]
        # Find corresponding canonical doc
        q_clean = clean_text(str(raw_null_row['question']))
        matching_docs = [d for d in canonical_docs if d.question == q_clean]
        if matching_docs:
            matched = matching_docs[0]
            print("\n[EXAMPLE - MISSING FOCUS IMPUTATION]")
            print(f"  BEFORE -> Focus Area: {repr(raw_null_row['focus_area'])}")
            print(f"  AFTER  -> Focus Area: {repr(matched.focus)} (Imputed Default)")
            print(f"  Doc ID -> {matched.document_id}")
            print("-" * 80)

    print(f"\n5. SAVING CANONICAL DATASET TO DISK...")
    # Export to JSONL format
    with open(CANONICAL_DATASET_PATH, "w", encoding="utf-8") as f:
        for doc in canonical_docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    print(f"   - Saved JSONL canonical dataset: '{CANONICAL_DATASET_PATH}' ({os.path.getsize(CANONICAL_DATASET_PATH)/(1024*1024):.2f} MB)")

    # Export to CSV format for audit
    df_canonical = pd.DataFrame([doc.to_dict() for doc in canonical_docs])
    df_canonical.to_csv(CANONICAL_CSV_PATH, index=False, encoding="utf-8")
    print(f"   - Saved CSV canonical dataset:   '{CANONICAL_CSV_PATH}' ({os.path.getsize(CANONICAL_CSV_PATH)/(1024*1024):.2f} MB)")

    print("\n" + "=" * 80)
    print("                    DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY            ")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
