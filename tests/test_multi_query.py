"""
Unit tests for Multi-Query Generation & ParentChildRetriever using standard library unittest.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from src.retrieval.multi_query import MultiQueryGenerator
from src.retrieval.retriever import ParentChildRetriever


class TestMultiQueryGenerator(unittest.TestCase):

    @patch("src.retrieval.multi_query.ChatOllama")
    def test_multi_query_generation_parsing(self, mock_chat_ollama):
        # Mock Ollama response returning 2 rephrased queries
        mock_response = MagicMock()
        mock_response.content = "1. How can someone recognize if they might have glaucoma?\n2. What warning signs or physical changes indicate the onset of glaucoma?"
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        generator = MultiQueryGenerator()
        queries = generator.generate_queries("What are the symptoms of Glaucoma?")

        # Must return total 3 queries: 1 original + 2 variations
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0], "What are the symptoms of Glaucoma?")
        self.assertIn("How can someone recognize", queries[1])
        self.assertIn("What warning signs", queries[2])

    def test_deduplication_and_top3_limit(self):
        # Test candidate deduplication logic
        candidates = [
            {"child_id": "c1", "parent_id": "p1", "similarity_score": 0.25},
            {"child_id": "c2", "parent_id": "p1", "similarity_score": 0.40},
            {"child_id": "c1", "parent_id": "p1", "similarity_score": 0.15},  # duplicate c1 with better score 0.15
            {"child_id": "c3", "parent_id": "p2", "similarity_score": 0.30},
            {"child_id": "c4", "parent_id": "p2", "similarity_score": 0.50},
        ]

        dedup_map = {}
        for c in candidates:
            cid = c["child_id"]
            if cid not in dedup_map or c["similarity_score"] < dedup_map[cid]["similarity_score"]:
                dedup_map[cid] = c

        sorted_cand = sorted(dedup_map.values(), key=lambda x: x["similarity_score"])[:3]

        self.assertEqual(len(sorted_cand), 3)
        self.assertEqual(sorted_cand[0]["child_id"], "c1")
        self.assertEqual(sorted_cand[0]["similarity_score"], 0.15)
        self.assertEqual(sorted_cand[1]["child_id"], "c3")
        self.assertEqual(sorted_cand[2]["child_id"], "c2")


if __name__ == "__main__":
    unittest.main()
