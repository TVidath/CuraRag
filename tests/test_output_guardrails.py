"""
Unit tests for 3-Layer Output Guardrails System.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.guardrails.output_guardrails import (
    DetoxifyToxicityFilter,
    CitationVerificationFilter,
    Llama31RagasFaithfulnessChecker,
    OutputGuardrailManager,
    OutputGuardrailResult
)


class TestOutputGuardrails(unittest.TestCase):

    def setUp(self):
        self.citation_filter = CitationVerificationFilter()
        self.sample_parent_docs = [
            {
                "focus": "Glaucoma",
                "source": "MedQuAD",
                "question": "What is glaucoma?",
                "answer": "Glaucoma is a group of eye conditions that damage the optic nerve, often caused by high pressure in the eye."
            },
            {
                "focus": "Glaucoma Treatment",
                "source": "MedQuAD",
                "question": "What are treatments for glaucoma?",
                "answer": "Treatments for glaucoma include prescription eye drops, oral medications, laser therapy, or surgery."
            }
        ]

    def test_citation_verification_filter_keeps_cited_lines(self):
        answer_text = (
            "Glaucoma damages the optic nerve [Source Document 1].\n"
            "Treatment includes eye drops and laser therapy [Source Document 2]."
        )
        sanitized, stripped_count, is_clean = self.citation_filter.filter(answer_text, num_documents=2)

        self.assertTrue(is_clean)
        self.assertEqual(stripped_count, 0)
        self.assertIn("[Source Document 1]", sanitized)
        self.assertIn("[Source Document 2]", sanitized)

    def test_citation_verification_filter_strips_uncited_lines(self):
        answer_text = (
            "Glaucoma damages the optic nerve [Source Document 1].\n"
            "This line has no citation and should be stripped out.\n"
            "Treatment includes prescription eye drops [Source Document 2]."
        )
        sanitized, stripped_count, is_clean = self.citation_filter.filter(answer_text, num_documents=2)

        self.assertFalse(is_clean)
        self.assertEqual(stripped_count, 1)
        self.assertIn("Glaucoma damages the optic nerve [Source Document 1].", sanitized)
        self.assertIn("Treatment includes prescription eye drops [Source Document 2].", sanitized)
        self.assertNotIn("This line has no citation and should be stripped out.", sanitized)

    def test_detoxify_toxicity_filter_fallback_clean(self):
        filter_inst = DetoxifyToxicityFilter()
        is_clean, score, reason = filter_inst.check("Glaucoma treatment includes eye drops.")

        self.assertTrue(is_clean)
        self.assertLess(score, 0.5)
        self.assertIsNone(reason)

    def test_detoxify_toxicity_filter_flagged(self):
        filter_inst = DetoxifyToxicityFilter()
        is_clean, score, reason = filter_inst.check("I hate you, go kill yourself you idiot.")

        self.assertFalse(is_clean)
        self.assertGreaterEqual(score, 0.5)
        self.assertIsNotNone(reason)


    @patch("src.guardrails.output_guardrails.ChatOllama")
    def test_llama31_faithfulness_checker_passes_threshold(self, mock_chat_ollama):
        mock_response = MagicMock()
        mock_response.content = "FAITHFULNESS_SCORE: 0.95\nEXPLANATION: All statements strictly match context."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        checker = Llama31RagasFaithfulnessChecker()
        answer = "Glaucoma damages the optic nerve [Source Document 1]."
        is_faithful, score, reason = checker.check("What is glaucoma?", answer, self.sample_parent_docs)

        self.assertTrue(is_faithful)
        self.assertEqual(score, 0.95)
        self.assertIsNone(reason)

    @patch("src.guardrails.output_guardrails.ChatOllama")
    def test_llama31_faithfulness_checker_fails_threshold(self, mock_chat_ollama):
        mock_response = MagicMock()
        mock_response.content = "FAITHFULNESS_SCORE: 0.30\nEXPLANATION: Answer makes unsupported medical claims."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        checker = Llama31RagasFaithfulnessChecker()
        answer = "Glaucoma can be instantly cured by drinking herbal tea [Source Document 1]."
        is_faithful, score, reason = checker.check("What cures glaucoma?", answer, self.sample_parent_docs)

        self.assertFalse(is_faithful)
        self.assertEqual(score, 0.30)
        self.assertIn("below threshold 0.5", reason)

    @patch("src.guardrails.output_guardrails.ChatOllama")
    def test_output_guardrail_manager_end_to_end_success(self, mock_chat_ollama):
        mock_response = MagicMock()
        mock_response.content = "FAITHFULNESS_SCORE: 0.90\nEXPLANATION: High quality."
        mock_chat_ollama.return_value.invoke.return_value = mock_response

        manager = OutputGuardrailManager()
        raw_answer = (
            "Glaucoma damages the optic nerve [Source Document 1].\n"
            "Eye drops help lower eye pressure [Source Document 2]."
        )

        result: OutputGuardrailResult = manager.validate(
            user_query="What is glaucoma?",
            raw_answer=raw_answer,
            parent_documents=self.sample_parent_docs
        )

        self.assertTrue(result.is_allowed)
        self.assertTrue(result.citation_check_passed)
        self.assertEqual(result.stripped_lines_count, 0)
        self.assertGreaterEqual(result.faithfulness_score, 0.5)


if __name__ == "__main__":
    unittest.main()
