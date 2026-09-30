"""
Unit tests for MedQuAD data cleaning and normalization pipeline using standard library unittest.
"""

import unittest
import pandas as pd
from src.data.cleaner import (
    clean_text,
    normalize_question_type,
    normalize_focus,
    normalize_source,
    generate_stable_doc_id,
    clean_dataset
)


class TestMedQuADCleaner(unittest.TestCase):

    def test_clean_text_html_and_unicode(self):
        raw = "<p>What is &quot;Glaucoma&quot; &amp; how to treat it?&#39;</p>"
        expected = 'What is "Glaucoma" & how to treat it?\''
        self.assertEqual(clean_text(raw), expected)

    def test_clean_text_medical_symbol_preservation(self):
        # Medical text with dosages, inequality symbols, acronyms, and dash
        raw = "Patient dosage: >50 mg/dL of H2O2 (Type-1 Diabetes) — status: OK."
        cleaned = clean_text(raw)
        self.assertIn(">50 mg/dL", cleaned)
        self.assertIn("H2O2", cleaned)
        self.assertIn("Type-1", cleaned)
        self.assertNotIn("—", cleaned)  # em-dash converted to standard hyphen
        self.assertIn("-", cleaned)

    def test_clean_text_whitespace_normalization(self):
        raw = "  Glaucoma    is a group   of diseases.  \n\n\n\n  Symptoms include...  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Glaucoma is a group of diseases.\n\nSymptoms include...")

    def test_missing_question_and_answer_dropped(self):
        df_raw = pd.DataFrame([
            {"question": None, "answer": "Valid answer", "source": "CDC", "focus_area": "Condition A"},
            {"question": "Valid question?", "answer": None, "source": "CDC", "focus_area": "Condition A"},
            {"question": "   ", "answer": "Valid answer", "source": "CDC", "focus_area": "Condition A"},
            {"question": "Valid question?", "answer": "   ", "source": "CDC", "focus_area": "Condition A"},
            {"question": "What is Glaucoma?", "answer": "Eye disease", "source": "CDC", "focus_area": "Glaucoma"},
        ])
        canonical_docs, report = clean_dataset(df_raw)
        self.assertEqual(len(canonical_docs), 1)
        self.assertEqual(report.missing_question_dropped, 1)
        self.assertEqual(report.missing_answer_dropped, 1)
        self.assertEqual(report.empty_string_dropped, 2)
        self.assertEqual(report.canonical_record_count, 1)
        self.assertEqual(canonical_docs[0].question, "What is Glaucoma?")

    def test_metadata_imputation(self):
        df_raw = pd.DataFrame([
            {"question": "What is Diabetes?", "answer": "High blood sugar", "source": None, "focus_area": None}
        ])
        canonical_docs, report = clean_dataset(df_raw)
        self.assertEqual(len(canonical_docs), 1)
        doc = canonical_docs[0]
        self.assertEqual(doc.focus, "General Health")
        self.assertEqual(doc.source, "MedQuAD")
        self.assertEqual(report.focus_imputed_count, 1)
        self.assertEqual(report.source_imputed_count, 1)

    def test_duplicate_qa_removal(self):
        df_raw = pd.DataFrame([
            {"question": "What is Asthma?", "answer": "Airway inflammation", "source": "CDC", "focus_area": "Asthma"},
            {"question": "What is Asthma?", "answer": "Airway inflammation", "source": "NIH", "focus_area": "Asthma"},
        ])
        canonical_docs, report = clean_dataset(df_raw)
        self.assertEqual(len(canonical_docs), 1)
        self.assertEqual(report.duplicate_qa_dropped, 1)

    def test_stable_doc_id_determinism(self):
        id1 = generate_stable_doc_id("What is Glaucoma?", "Eye disease", "CDC")
        id2 = generate_stable_doc_id("What is Glaucoma?", "Eye disease", "CDC")
        id3 = generate_stable_doc_id("What is Diabetes?", "Eye disease", "CDC")
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertTrue(id1.startswith("mq_"))


if __name__ == "__main__":
    unittest.main()
