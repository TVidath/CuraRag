"""
MedQuAD Dataset Inspection Pipeline

This script performs exploratory data analysis (EDA) on the MedQuAD dataset.
It analyzes structural properties, text statistics, duplicate records,
question intent distributions, and exports a 200-row sample for manual audit.
"""

import os
import re
import pandas as pd
import numpy as np


def categorize_question_type(question: str) -> str:
    """
    Categorizes medical question intent based on common question phrasing patterns.
    """
    if not isinstance(question, str):
        return "Unknown"
    
    q_lower = question.lower().strip()
    
    if re.search(r"what is|what are|definition|information about", q_lower):
        return "Definition / General Info"
    elif re.search(r"cause|why|risk factor|how do you get", q_lower):
        return "Causes & Risk Factors"
    elif re.search(r"symptom|sign|look like|feel like", q_lower):
        return "Symptoms & Signs"
    elif re.search(r"treat|cure|medication|drug|therapy|manage|surgery", q_lower):
        return "Treatment & Management"
    elif re.search(r"diagnos|test|screen|exam|detect", q_lower):
        return "Diagnosis & Testing"
    elif re.search(r"prevent|avoid|reduce risk|lifestyle", q_lower):
        return "Prevention & Risk Reduction"
    elif re.search(r"complication|outcome|prognosis|stage|severe", q_lower):
        return "Complications & Prognosis"
    elif re.search(r"genetic|inherit|hereditary|gene|mutation", q_lower):
        return "Genetics & Inheritance"
    else:
        return "Other / Specific Query"


def inspect_medquad(csv_path: str = os.path.join("data", "medquad.csv"), sample_output_path: str = os.path.join("samples", "medquad_sample_200.csv")):
    """
    Loads and inspects the MedQuAD dataset, printing statistical summaries
    and exporting a 200-row sample.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at '{csv_path}'. Please run scripts/download_dataset.py first.")

    print("=" * 80)
    print("                      MEDQUAD DATASET INSPECTION REPORT                     ")
    print("=" * 80)

    # 1. Load Dataset
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"\n1. DATASET OVERVIEW")
    print(f"   - File Path: {csv_path}")
    print(f"   - Total Rows (Records): {total_rows:,}")
    print(f"   - Column Names: {list(df.columns)}")

    # 2. Missing Values Analysis
    print(f"\n2. MISSING VALUES ANALYSIS")
    missing_df = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / total_rows * 100).round(3)
    })
    print(missing_df.to_string())

    # 3. Duplicate Records Analysis
    print(f"\n3. DUPLICATE ANALYSIS")
    dup_questions_count = df['question'].duplicated().sum()
    unique_questions_count = df['question'].nunique()
    exact_duplicates_count = df.duplicated().sum()
    dup_qa_count = df.duplicated(subset=['question', 'answer']).sum()

    print(f"   - Unique Questions: {unique_questions_count:,}")
    print(f"   - Duplicate Question Texts: {dup_questions_count:,} ({dup_questions_count / total_rows * 100:.2f}%)")
    print(f"   - Duplicate (Question, Answer) Pairs: {dup_qa_count:,} ({dup_qa_count / total_rows * 100:.2f}%)")
    print(f"   - Exact Duplicate Rows (All columns): {exact_duplicates_count:,}")

    # 4. Question Type Intent Distribution
    df['question_type'] = df['question'].apply(categorize_question_type)
    print(f"\n4. QUESTION TYPE / INTENT DISTRIBUTION")
    q_type_counts = df['question_type'].value_counts()
    q_type_pct = (q_type_counts / total_rows * 100).round(2)
    q_type_df = pd.DataFrame({'Count': q_type_counts, 'Percentage (%)': q_type_pct})
    print(q_type_df.to_string())

    # 5. Focus Area Distribution
    print(f"\n5. FOCUS AREA DISTRIBUTION")
    unique_focus = df['focus_area'].nunique(dropna=True)
    print(f"   - Total Unique Focus Areas / Conditions: {unique_focus:,}")
    print(f"   - Top 15 Most Frequent Focus Areas:")
    top_focus = df['focus_area'].value_counts().head(15)
    top_focus_df = pd.DataFrame({'Count': top_focus, 'Percentage (%)': (top_focus / total_rows * 100).round(2)})
    print(top_focus_df.to_string())

    # 6. Source Provider Distribution
    print(f"\n6. SOURCE PROVIDER DISTRIBUTION")
    source_counts = df['source'].value_counts()
    source_pct = (source_counts / total_rows * 100).round(2)
    source_df = pd.DataFrame({'Count': source_counts, 'Percentage (%)': source_pct})
    print(source_df.to_string())

    # 7. Text Length Statistics (Questions & Answers)
    df['q_char_len'] = df['question'].astype(str).str.len()
    df['q_word_len'] = df['question'].astype(str).str.split().str.len()
    
    valid_answers = df['answer'].dropna().astype(str)
    df['a_char_len'] = valid_answers.str.len()
    df['a_word_len'] = valid_answers.str.split().str.len()

    print(f"\n7. QUESTION LENGTH STATISTICS")
    q_stats = pd.DataFrame({
        'Char Length': df['q_char_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
        'Word Count': df['q_word_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    }).round(1)
    print(q_stats.to_string())

    print(f"\n8. ANSWER LENGTH STATISTICS")
    a_stats = pd.DataFrame({
        'Char Length': df['a_char_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
        'Word Count': df['a_word_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    }).round(1)
    print(a_stats.to_string())

    # 8. Representative Examples
    print(f"\n9. REPRESENTATIVE EXAMPLES (10 DIVERSE SAMPLES)")
    print("-" * 80)
    # Pick 10 diverse examples across different sources
    sample_10 = df.groupby('source', group_keys=False).apply(lambda x: x.sample(1, random_state=42)).head(10)
    for idx, (_, row) in enumerate(sample_10.iterrows(), 1):
        answer_preview = str(row['answer'])[:150].replace('\n', ' ') + "..." if pd.notnull(row['answer']) else "[MISSING ANSWER]"
        print(f"[{idx}] Source: {row['source']} | Focus: {row['focus_area']}")
        print(f"    Question : {row['question']}")
        print(f"    Answer   : {answer_preview}")
        print("-" * 80)

    # 9. Random 200-row sample for manual audit
    os.makedirs(os.path.dirname(sample_output_path), exist_ok=True)
    sample_200 = df.sample(n=200, random_state=42)
    export_cols = ['question', 'answer', 'source', 'focus_area', 'question_type']
    sample_200[export_cols].to_csv(sample_output_path, index=False)
    print(f"\n10. SAMPLE EXPORT")
    print(f"    - Saved 200 randomly sampled rows to '{sample_output_path}' with seed=42.")
    print("=" * 80)
    print("                               END OF REPORT                                ")
    print("=" * 80)


if __name__ == "__main__":
    inspect_medquad()
