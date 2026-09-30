"""
Unit tests for MedQuAD Parent-Child Chunking Engine using standard library unittest.
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schema import CanonicalDocument, ParentDocument, ChildChunk
from src.data.cleaner import generate_stable_doc_id
from src.data.chunker import (
    split_text_recursive,
    create_parent_and_children,
    chunk_canonical_dataset
)


class TestMedQuADChunker(unittest.TestCase):

    def setUp(self):
        # Sample short document (<= 150 tokens)
        self.doc_short = CanonicalDocument(
            document_id=generate_stable_doc_id("What is Glaucoma?", "Eye disease causing vision loss.", "CDC"),
            question="What is Glaucoma?",
            answer="Eye disease causing vision loss.",
            question_type="Definition / General Info",
            focus="Glaucoma",
            source="CDC",
            char_length=32,
            word_count=5
        )

        # Sample long document (> 150 tokens / > 600 chars)
        long_answer = (
            "Glaucoma is a group of eye conditions that damage the optic nerve. "
            "The health of the optic nerve is vital for good vision. This damage is often caused by an abnormally high pressure in your eye. "
            "Glaucoma is one of the leading causes of blindness for people over the age of 60. "
            "It can occur at any age but is more common in older adults. "
            "Many forms of glaucoma have no warning signs. The effect is so gradual that you may not notice a change in vision until the condition is at an advanced stage. "
            "\n\n"
            "Symptoms of glaucoma include severe headache, eye pain, nausea and vomiting, blurred vision, and halo around lights. "
            "If you experience any of these symptoms, seek emergency medical care immediately. "
            "\n\n"
            "Treatment options include prescription eye drops, oral medications, laser treatment, or surgery. "
            "Regular eye exams are essential for early detection."
        )
        self.doc_long = CanonicalDocument(
            document_id=generate_stable_doc_id("What are symptoms and treatments for Glaucoma?", long_answer, "NIHSeniorHealth"),
            question="What are symptoms and treatments for Glaucoma?",
            answer=long_answer,
            question_type="Symptoms & Signs",
            focus="Glaucoma",
            source="NIHSeniorHealth",
            char_length=len(long_answer),
            word_count=len(long_answer.split())
        )

    def test_single_child_for_short_document(self):
        parent, children = create_parent_and_children(self.doc_short)
        self.assertEqual(len(children), 1)
        self.assertEqual(parent.child_ids, [children[0].chunk_id])
        self.assertEqual(children[0].chunk_text, self.doc_short.answer)

    def test_every_child_has_valid_parent_id(self):
        canonical_docs = [self.doc_short, self.doc_long]
        parents, children, report = chunk_canonical_dataset(canonical_docs)
        parent_id_set = {p.parent_id for p in parents}

        for child in children:
            self.assertIn(child.parent_id, parent_id_set)

    def test_every_parent_has_at_least_one_child(self):
        canonical_docs = [self.doc_short, self.doc_long]
        parents, children, report = chunk_canonical_dataset(canonical_docs)

        for parent in parents:
            self.assertGreaterEqual(len(parent.child_ids), 1)

    def test_metadata_preservation_in_children(self):
        parent, children = create_parent_and_children(self.doc_long)
        for child in children:
            self.assertEqual(child.question, self.doc_long.question)
            self.assertEqual(child.question_type, self.doc_long.question_type)
            self.assertEqual(child.focus, self.doc_long.focus)
            self.assertEqual(child.source, self.doc_long.source)

    def test_no_empty_chunks_produced(self):
        canonical_docs = [self.doc_short, self.doc_long]
        parents, children, report = chunk_canonical_dataset(canonical_docs)

        for child in children:
            self.assertTrue(bool(child.chunk_text.strip()))
            self.assertGreater(child.token_count, 0)

    def test_chunk_ids_are_unique(self):
        canonical_docs = [self.doc_short, self.doc_long]
        parents, children, report = chunk_canonical_dataset(canonical_docs)
        child_id_set = {c.chunk_id for c in children}

        self.assertEqual(len(child_id_set), len(children))

    def test_parent_ids_are_stable(self):
        parent1, _ = create_parent_and_children(self.doc_short)
        parent2, _ = create_parent_and_children(self.doc_short)
        self.assertEqual(parent1.parent_id, parent2.parent_id)

    def test_parent_child_reconstruction(self):
        canonical_docs = [self.doc_short, self.doc_long]
        parents, children, report = chunk_canonical_dataset(canonical_docs)

        parent_map = {p.parent_id: p for p in parents}
        child_map = {c.chunk_id: c for c in children}

        # Pick a target parent and reconstruct all of its child chunks
        target_parent = parents[1]
        reconstructed_children = [child_map[cid] for cid in target_parent.child_ids]

        self.assertEqual(len(reconstructed_children), len(target_parent.child_ids))
        for child in reconstructed_children:
            self.assertEqual(child.parent_id, target_parent.parent_id)
            # Verify child chunk text stems from parent answer
            words = child.chunk_text.split()
            # Ensure key words from child exist in parent answer
            self.assertTrue(any(w in target_parent.answer for w in words[:5]))


if __name__ == "__main__":
    unittest.main()
