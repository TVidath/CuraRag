"""
End-to-End Parent-Child Multi-Query RAG Pipeline with Input and Output Guardrails for MedQuAD.

Connects:
1. Multi-Layer Input Guardrails (Prompt Injection, Llama Guard, Presidio PII Redaction, Topic Moderation)
2. Multi-Query Pre-Retrieval Generation (gemma4:latest)
3. Dense Vector Candidate Search (all-MiniLM-L6-v2 + Chroma/Pinecone)
4. Candidate Child Chunk Deduplication
5. Cross-Encoder Reranking (bge-reranker-base)
6. Parent Document Lookup & Context Assembly
7. Gemma 4 Grounded Response Synthesis (ChatOllama)
8. Multi-Layer Output Guardrails (Detoxify Toxicity, Citation Verification Line Stripping, Llama 3.1 & Ragas Faithfulness Check)
"""

import time
import logging
from typing import Dict, Any, Optional

from config.settings import TOP_K_CANDIDATE_RESULTS, TOP_N_RERANKED_RESULTS
from src.guardrails.input_guardrails import InputGuardrailManager, GuardrailResult
from src.guardrails.output_guardrails import OutputGuardrailManager, OutputGuardrailResult
from src.retrieval.retriever import ParentChildRetriever
from src.generation.generator import RAGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MedQuADRAGPipeline:
    """
    Complete Medical RAG Pipeline orchestrating Guardrails, Retrieval, and Synthesis stages.
    """

    def __init__(
        self,
        retriever: Optional[ParentChildRetriever] = None,
        generator: Optional[RAGGenerator] = None,
        guardrail_manager: Optional[InputGuardrailManager] = None,
        output_guardrail_manager: Optional[OutputGuardrailManager] = None
    ):
        self.retriever = retriever or ParentChildRetriever()
        self.generator = generator or RAGGenerator()
        self.guardrail_manager = guardrail_manager or InputGuardrailManager()
        self.output_guardrail_manager = output_guardrail_manager or OutputGuardrailManager()

    def query(
        self,
        user_query: str,
        top_k_candidates: int = TOP_K_CANDIDATE_RESULTS,
        top_n_final: int = TOP_N_RERANKED_RESULTS
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query execution from input guardrails to final synthesized answer.
        """
        start_time = time.time()
        logging.info(f"Executing MedQuAD RAG pipeline for query: '{user_query}'")

        # Step 0: Input Guardrails Inspection & Sanitization
        guardrail_result: GuardrailResult = self.guardrail_manager.validate_and_sanitize(user_query)

        if not guardrail_result.is_allowed:
            elapsed = round(time.time() - start_time, 3)
            logging.warning(f"Query blocked by Input Guardrail [{guardrail_result.flagged_layer}]: {guardrail_result.rejection_reason}")
            return {
                "query": user_query,
                "sanitized_query": guardrail_result.sanitized_query,
                "answer": f"Request Blocked by Input Safety Guardrails [{guardrail_result.flagged_layer}]: {guardrail_result.rejection_reason}",
                "guardrail_passed": False,
                "input_guardrail_passed": False,
                "output_guardrail_passed": False,
                "flagged_layer": guardrail_result.flagged_layer,
                "rejection_reason": guardrail_result.rejection_reason,
                "generated_query_variations": [],
                "retrieved_parent_count": 0,
                "top_child_chunks": [],
                "parent_documents": [],
                "execution_time_seconds": elapsed
            }

        sanitized_query = guardrail_result.sanitized_query

        # Step 1-5: Execute Retrieval Stage
        retrieval_result = self.retriever.retrieve(
            user_query=sanitized_query,
            top_k_candidates=top_k_candidates,
            top_n_final=top_n_final
        )

        parent_documents = retrieval_result.get("parent_documents", [])
        top_child_chunks = retrieval_result.get("top_child_chunks", [])
        generated_queries = retrieval_result.get("generated_queries", [])

        # Step 6: Execute Gemma 4 Answer Generation Stage
        generation_result = self.generator.generate(
            user_query=sanitized_query,
            parent_documents=parent_documents
        )

        raw_answer = generation_result.get("answer", "")

        # Step 7: Execute Output Guardrails (Detoxify, Citation Verification Filter, Llama 3.1 & Ragas Faithfulness Check)
        output_guardrail_result: OutputGuardrailResult = self.output_guardrail_manager.validate(
            user_query=sanitized_query,
            raw_answer=raw_answer,
            parent_documents=parent_documents
        )

        elapsed_time = round(time.time() - start_time, 3)
        logging.info(f"Pipeline execution completed in {elapsed_time}s")

        final_answer = output_guardrail_result.sanitized_answer if output_guardrail_result.is_allowed else (
            f"Response Blocked by Output Safety/Faithfulness Guardrail [{output_guardrail_result.flagged_layer}]: {output_guardrail_result.rejection_reason}"
        )

        return {
            "query": user_query,
            "sanitized_query": sanitized_query,
            "raw_answer": raw_answer,
            "answer": final_answer,
            "guardrail_passed": output_guardrail_result.is_allowed,
            "input_guardrail_passed": True,
            "output_guardrail_passed": output_guardrail_result.is_allowed,
            "output_guardrail_details": {
                "toxicity_score": output_guardrail_result.toxicity_score,
                "faithfulness_score": output_guardrail_result.faithfulness_score,
                "citation_check_passed": output_guardrail_result.citation_check_passed,
                "stripped_lines_count": output_guardrail_result.stripped_lines_count,
                "flagged_layer": output_guardrail_result.flagged_layer,
                "rejection_reason": output_guardrail_result.rejection_reason
            },
            "generated_query_variations": generated_queries,
            "retrieved_parent_count": len(parent_documents),
            "top_child_chunks": top_child_chunks,
            "parent_documents": parent_documents,
            "formatted_context": generation_result.get("formatted_context"),
            "model_used": generation_result.get("model_used"),
            "execution_time_seconds": elapsed_time
        }

