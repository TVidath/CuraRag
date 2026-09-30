"""
Unit tests for Gemma 4 RAG Response Generator and End-to-End Pipeline.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.generation.generator import RAGGenerator, format_parent_context
from src.rag_pipeline import MedQuADRAGPipeline


class TestRAGGenerator(unittest.TestCase):

    def test_format_parent_context(self):
        parent_docs = [
            {
                "parent_id": "mq_p1",
                "focus": "Glaucoma",
                "source": "GARD",
                "question": "What are the symptoms of Glaucoma?",
                "answer": "Glaucoma often has no early symptoms. Over time, peripheral vision gradually decreases.",
                "child_ids": ["c1", "c2"],
                "word_count": 14
            },
            {
                "parent_id": "mq_p2",
                "focus": "Cataracts",
                "source": "MedlinePlus",
                "question": "How are Cataracts treated?",
                "answer": "Cataracts are treated with surgery to remove the cloudy lens.",
                "child_ids": ["c3"],
                "word_count": 10
            }
        ]

        context_str = format_parent_context(parent_docs)

        # Check that high-value fields are present
        self.assertIn("[Source Document 1]", context_str)
        self.assertIn("Medical Topic: Glaucoma", context_str)
        self.assertIn("Source: GARD", context_str)
        self.assertIn("Document Question: What are the symptoms of Glaucoma?", context_str)
        self.assertIn("peripheral vision gradually decreases", context_str)

        self.assertIn("[Source Document 2]", context_str)
        self.assertIn("Medical Topic: Cataracts", context_str)

        # Ensure internal database fields are excluded
        self.assertNotIn("mq_p1", context_str)
        self.assertNotIn("child_ids", context_str)
        self.assertNotIn("word_count", context_str)

    def test_format_parent_context_empty(self):
        empty_str = format_parent_context([])
        self.assertEqual(empty_str, "No relevant parent document contexts retrieved.")

    @patch("src.generation.generator.ChatOllama")
    def test_generator_generate(self, mock_chat_ollama):
        mock_response = MagicMock()
        mock_response.content = "Glaucoma causes loss of peripheral vision over time as noted in [Source Document 1]."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        generator = RAGGenerator()
        result = generator.generate(
            user_query="What happens in Glaucoma?",
            parent_documents=[{
                "focus": "Glaucoma",
                "source": "GARD",
                "question": "What are symptoms?",
                "answer": "Loss of peripheral vision."
            }]
        )

        self.assertEqual(result["user_query"], "What happens in Glaucoma?")
        self.assertIn("Glaucoma causes loss of peripheral vision", result["answer"])
        self.assertEqual(result["num_parent_documents"], 1)

    @patch("src.rag_pipeline.ParentChildRetriever")
    @patch("src.rag_pipeline.RAGGenerator")
    def test_rag_pipeline_end_to_end(self, mock_generator_cls, mock_retriever_cls):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = {
            "user_query": "What is Diabetes?",
            "generated_queries": ["Symptoms of diabetes", "Type 2 diabetes signs"],
            "top_child_chunks": [{"child_id": "c1", "parent_id": "p1"}],
            "parent_documents": [{"parent_id": "p1", "focus": "Diabetes", "answer": "Diabetes is high blood sugar."}]
        }

        mock_generator = MagicMock()
        mock_generator.generate.return_value = {
            "answer": "Diabetes is a chronic condition characterized by high blood sugar levels [Source Document 1].",
            "formatted_context": "Formatted context text",
            "model_used": "gemma4:latest"
        }


        pipeline = MedQuADRAGPipeline(retriever=mock_retriever, generator=mock_generator)
        output = pipeline.query("What is Diabetes?")

        self.assertEqual(output["query"], "What is Diabetes?")
        self.assertIn("high blood sugar", output["answer"])
        self.assertEqual(output["retrieved_parent_count"], 1)
        self.assertEqual(len(output["generated_query_variations"]), 2)


if __name__ == "__main__":
    unittest.main()
