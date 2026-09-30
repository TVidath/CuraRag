"""
Unit tests for MedQuAD Cross-Encoder Reranker using standard library unittest.
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker()

    def test_reranker_returns_top_n(self):
        query = "What are the symptoms of Glaucoma?"
        candidate_chunks = [
            {
                "child_id": "c1",
                "parent_id": "p1",
                "similarity_score": 0.45,
                "chunk_text": "Glaucoma symptoms include severe headache, eye pain, blurred vision, and halo around lights."
            },
            {
                "child_id": "c2",
                "parent_id": "p2",
                "similarity_score": 0.50,
                "chunk_text": "Asthma is a chronic respiratory condition causing wheezing and shortness of breath."
            },
            {
                "child_id": "c3",
                "parent_id": "p1",
                "similarity_score": 0.48,
                "chunk_text": "Early signs of eye pressure elevation may lead to optic nerve damage in glaucoma."
            }
        ]

        reranked = self.reranker.rerank(query, candidate_chunks, top_n=2)
        
        self.assertEqual(len(reranked), 2)
        self.assertIn("rerank_score", reranked[0])
        # Verify glaucoma chunk (c1) is ranked higher than asthma chunk (c2)
        self.assertEqual(reranked[0]["child_id"], "c1")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])

    def test_reranker_handles_empty_candidates(self):
        reranked = self.reranker.rerank("Glaucoma", [], top_n=3)
        self.assertEqual(reranked, [])


if __name__ == "__main__":
    unittest.main()
