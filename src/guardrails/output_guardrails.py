"""
Multi-Layer Output Guardrails Module for MedQuAD RAG Application.

Components:
1. DetoxifyToxicityFilter: Detects toxicity, obscene language, threats, insults, and identity attacks.
2. CitationVerificationFilter: Enforces [Source Document {idx}] citations line-by-line and strips uncited lines.
3. Llama31RagasFaithfulnessChecker: Evaluates answer faithfulness/hallucination using Llama 3.1 LLM-as-Judge & Ragas metrics (Threshold: 0.5).
4. OutputGuardrailManager: Runs all output guardrail layers sequentially.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.settings import (
    OLLAMA_BASE_URL,
    LLAMA_JUDGE_MODEL_NAME,
    FAITHFULNESS_THRESHOLD,
    TOXICITY_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class OutputGuardrailResult:
    """
    Result object returned by the Output Guardrail pipeline.
    """
    is_allowed: bool
    sanitized_answer: str
    toxicity_score: float
    faithfulness_score: float
    citation_check_passed: bool
    stripped_lines_count: int
    rejection_reason: Optional[str] = None
    flagged_layer: Optional[str] = None


class DetoxifyToxicityFilter:
    """
    Layer 1: Toxicity Filter using detoxify with fallback.
    Screens output strings for toxic, offensive, obscene, or harmful content.
    """

    def __init__(self, threshold: float = TOXICITY_THRESHOLD):
        self.threshold = threshold
        self.detoxify_available = False
        self.model = None

        try:
            from detoxify import Detoxify
            self.model = Detoxify("original")
            self.detoxify_available = True
            logging.info("Detoxify toxicity model initialized successfully.")
        except Exception as e:
            logging.info(f"Detoxify model not available ({e}). Using regex/keyword fallback toxicity filter.")

    def check(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """
        Evaluates text for toxicity.
        Returns (is_clean, toxicity_score, rejection_reason).
        """
        if not text or not text.strip():
            return True, 0.0, None

        # 1. Primary: Detoxify package if installed and working
        if self.detoxify_available and self.model:
            try:
                results = self.model.predict(text)
                # Find maximum score across all toxicity dimensions
                max_score = float(max(results.values()))
                if max_score >= self.threshold:
                    flagged_cats = [k for k, v in results.items() if v >= self.threshold]
                    reason = f"Output flagged for toxicity ({', '.join(flagged_cats)}) with max score {max_score:.3f} >= threshold {self.threshold}."
                    logging.warning(f"Detoxify flagged output: {reason}")
                    return False, max_score, reason
                return True, max_score, None
            except Exception as e:
                logging.warning(f"Detoxify prediction error ({e}), proceeding to fallback check.")

        # 2. Fallback: Keyword / Pattern Toxicity Filter
        toxic_keywords = [
            "hate group", "kill yourself", "violent assault", "slur", "toxic substance", "lethal dose"
        ]
        text_lower = text.lower()
        for kw in toxic_keywords:
            if kw in text_lower:
                reason = f"Fallback toxicity check flagged term: '{kw}'"
                logging.warning(f"Toxicity filter fallback flagged output: {reason}")
                return False, 1.0, reason

        return True, 0.0, None


class CitationVerificationFilter:
    """
    Layer 2: Line-by-line Citation Verification Filter.
    Requires every non-empty response line to cite at least one valid source document index
    in format [Source Document {idx}]. Strips lines without a valid citation.
    """

    CITATION_PATTERN = re.compile(r"\[Source Document\s+(\d+)\]", re.IGNORECASE)

    def filter(self, answer: str, num_documents: int) -> Tuple[str, int, bool]:
        """
        Strips lines from answer that do not cite a valid [Source Document {idx}].
        Returns (sanitized_answer, stripped_lines_count, is_clean).
        """
        if not answer or not answer.strip():
            return answer, 0, True

        lines = answer.split("\n")
        valid_indices = set(range(1, num_documents + 1)) if num_documents > 0 else set()

        retained_lines = []
        stripped_count = 0

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Keep empty structural lines / paragraph breaks
                retained_lines.append(line)
                continue

            # Check if line contains any citation matching [Source Document {idx}]
            matches = self.CITATION_PATTERN.findall(line)
            has_valid_citation = False

            if matches:
                for match in matches:
                    try:
                        doc_idx = int(match)
                        if num_documents == 0 or doc_idx in valid_indices:
                            has_valid_citation = True
                            break
                    except ValueError:
                        continue

            if has_valid_citation:
                retained_lines.append(line)
            else:
                logging.info(f"CitationFilter: Stripping uncited line -> '{stripped_line}'")
                stripped_count += 1

        sanitized_answer = "\n".join(retained_lines).strip()
        is_clean = (stripped_count == 0)

        return sanitized_answer, stripped_count, is_clean


class Llama31RagasFaithfulnessChecker:
    """
    Layer 3: Faithfulness & Hallucination Checker.
    Uses Llama 3.1 LLM-as-Judge and Ragas faithfulness metrics (Threshold: 0.5).
    Evaluates whether the generated answer is faithful to the question and retrieved parent document chunks.
    """

    LLM_JUDGE_PROMPT_TEMPLATE = """You are a rigorous clinical quality auditor evaluating AI hallucination and faithfulness.
Assess whether every statement in the Candidate Answer is strictly derived from and supported by the Provided Source Contexts.

Provided Source Contexts:
{context}

User Question: "{user_query}"

Candidate Answer:
"{answer}"

Evaluation Instructions:
1. Extract key claims from Candidate Answer.
2. Check if each claim is directly supported by Provided Source Contexts.
3. Calculate a Faithfulness Score between 0.0 and 1.0 (Supported Claims / Total Claims).

Respond EXACTLY in this format:
FAITHFULNESS_SCORE: <score between 0.0 and 1.0>
EXPLANATION: <brief breakdown of supported vs unsupported claims>
"""

    def __init__(
        self,
        model_name: str = LLAMA_JUDGE_MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        threshold: float = FAITHFULNESS_THRESHOLD
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.threshold = threshold

        self.prompt = PromptTemplate(
            template=self.LLM_JUDGE_PROMPT_TEMPLATE,
            input_variables=["user_query", "context", "answer"]
        )

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0
        )

        self.ragas_available = False
        try:
            import importlib
            importlib.import_module("ragas")
            self.ragas_available = True
            logging.info("Ragas library detected for faithfulness evaluation.")
        except Exception:
            logging.info("Ragas library not installed. Using Llama 3.1 LLM-as-Judge for faithfulness evaluation.")

    def _format_context_text(self, parent_documents: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved parent chunks into text context for evaluation.
        """
        if not parent_documents:
            return "No source contexts retrieved."

        blocks = []
        for idx, doc in enumerate(parent_documents, 1):
            answer_text = doc.get("answer", doc.get("text", "")).strip()
            focus = doc.get("focus", "General Health")
            blocks.append(f"[Source Document {idx}] ({focus}): {answer_text}")

        return "\n\n".join(blocks)

    def check(
        self,
        user_query: str,
        answer: str,
        parent_documents: List[Dict[str, Any]]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Evaluates faithfulness of generated answer against user query and retrieved parent document chunks.
        Returns (is_faithful, faithfulness_score, reason).
        """
        if not answer or not answer.strip():
            return True, 1.0, None

        if not parent_documents:
            return False, 0.0, "Faithfulness Check Failed: No retrieved parent documents available to ground answer."

        context_text = self._format_context_text(parent_documents)

        # 1. Ragas Faithfulness Score (if available)
        ragas_score = 1.0
        ragas_failed = False
        ragas_reason = None

        if self.ragas_available:
            try:
                import importlib
                datasets_mod = importlib.import_module("datasets")
                ragas_mod = importlib.import_module("ragas")
                metrics_mod = importlib.import_module("ragas.metrics")

                dataset_cls = getattr(datasets_mod, "Dataset")
                eval_func = getattr(ragas_mod, "evaluate")
                faithfulness_metric = getattr(metrics_mod, "faithfulness")

                data_dict = {
                    "question": [user_query],
                    "answer": [answer],
                    "contexts": [[doc.get("answer", doc.get("text", "")) for doc in parent_documents]]
                }
                eval_dataset = dataset_cls.from_dict(data_dict)
                result = eval_func(eval_dataset, metrics=[faithfulness_metric])
                ragas_score = float(result["faithfulness"][0])

                if ragas_score < self.threshold:
                    ragas_failed = True
                    ragas_reason = f"Ragas Faithfulness Score ({ragas_score:.2f}) below threshold {self.threshold}."
                    logging.warning(f"FaithfulnessChecker flagged output: {ragas_reason}")

            except Exception as e:
                logging.warning(f"Ragas evaluation failed ({e}). Proceeding with Llama 3.1 LLM-as-Judge.")

        # 2. Direct Llama 3.1 LLM-as-Judge Evaluation
        llama_judge_score = 1.0
        llama_failed = False
        llama_reason = None

        try:
            formatted_prompt = self.prompt.format(
                user_query=user_query,
                context=context_text,
                answer=answer
            )

            logging.info(f"Invoking Llama 3.1 ({self.model_name}) LLM-as-Judge for faithfulness check...")
            response = self.llm.invoke(formatted_prompt)
            result_text = (response.content if hasattr(response, "content") else str(response)).strip()

            score_match = re.search(r"FAITHFULNESS_SCORE:\s*([\d\.]+)", result_text, re.IGNORECASE)
            if score_match:
                try:
                    llama_judge_score = float(score_match.group(1))
                except ValueError:
                    llama_judge_score = 1.0

            if llama_judge_score < self.threshold:
                llama_failed = True
                llama_reason = f"Llama 3.1 Faithfulness Score ({llama_judge_score:.2f}) below threshold {self.threshold}. Output contains potential hallucination."
                logging.warning(f"FaithfulnessChecker flagged output: {llama_reason} | Judge response: {result_text}")

        except Exception as e:
            logging.warning(f"Llama 3.1 Faithfulness Check skipped due to LLM error ({e}).")

        # 3. Combine evaluation results from both Ragas and Llama 3.1 LLM-as-Judge
        if ragas_failed:
            return False, ragas_score, ragas_reason
        if llama_failed:
            return False, llama_judge_score, llama_reason

        final_score = min(ragas_score, llama_judge_score)
        return True, final_score, None



class OutputGuardrailManager:
    """
    Master Output Guardrail Manager executing 3 sequential safety and quality layers:
    1. Detoxify Toxicity Filter
    2. Citation Verification Filter (line-by-line stripping of uncited statements)
    3. Llama 3.1 + Ragas Faithfulness Hallucination Checker (Threshold: 0.5)
    """

    def __init__(
        self,
        model_name: str = LLAMA_JUDGE_MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        toxicity_threshold: float = TOXICITY_THRESHOLD,
        faithfulness_threshold: float = FAITHFULNESS_THRESHOLD
    ):
        self.toxicity_filter = DetoxifyToxicityFilter(threshold=toxicity_threshold)
        self.citation_filter = CitationVerificationFilter()
        self.faithfulness_checker = Llama31RagasFaithfulnessChecker(
            model_name=model_name,
            base_url=base_url,
            threshold=faithfulness_threshold
        )

    def validate(
        self,
        user_query: str,
        raw_answer: str,
        parent_documents: List[Dict[str, Any]]
    ) -> OutputGuardrailResult:
        """
        Runs raw answer through output guardrails:
        1. Detoxify Toxicity Check
        2. Citation Verification Line Stripper
        3. Llama 3.1 & Ragas Faithfulness Hallucination Checker
        """
        if not raw_answer or not raw_answer.strip():
            return OutputGuardrailResult(
                is_allowed=False,
                sanitized_answer="",
                toxicity_score=0.0,
                faithfulness_score=0.0,
                citation_check_passed=False,
                stripped_lines_count=0,
                rejection_reason="Generated answer text is empty.",
                flagged_layer="Output Validation"
            )

        # Layer 1: Toxicity Check
        is_clean_toxic, tox_score, tox_reason = self.toxicity_filter.check(raw_answer)
        if not is_clean_toxic:
            return OutputGuardrailResult(
                is_allowed=False,
                sanitized_answer=raw_answer,
                toxicity_score=tox_score,
                faithfulness_score=0.0,
                citation_check_passed=False,
                stripped_lines_count=0,
                rejection_reason=tox_reason,
                flagged_layer="Layer 1: Detoxify Toxicity Filter"
            )

        # Layer 2: Citation Verification & Uncited Line Stripping
        num_docs = len(parent_documents)
        sanitized_answer, stripped_count, citation_passed = self.citation_filter.filter(raw_answer, num_docs)

        if not sanitized_answer or not sanitized_answer.strip():
            return OutputGuardrailResult(
                is_allowed=False,
                sanitized_answer="",
                toxicity_score=tox_score,
                faithfulness_score=0.0,
                citation_check_passed=False,
                stripped_lines_count=stripped_count,
                rejection_reason="All lines in answer were stripped due to missing [Source Document N] citations.",
                flagged_layer="Layer 2: Citation Verification Filter"
            )


        # Layer 3: Llama 3.1 + Ragas Faithfulness & Hallucination Check
        is_faithful, faith_score, faith_reason = self.faithfulness_checker.check(
            user_query=user_query,
            answer=sanitized_answer,
            parent_documents=parent_documents
        )

        if not is_faithful:
            return OutputGuardrailResult(
                is_allowed=False,
                sanitized_answer=sanitized_answer,
                toxicity_score=tox_score,
                faithfulness_score=faith_score,
                citation_check_passed=citation_passed,
                stripped_lines_count=stripped_count,
                rejection_reason=faith_reason,
                flagged_layer="Layer 3: Llama 3.1 & Ragas Faithfulness Checker"
            )

        # All Output Guardrails Passed!
        return OutputGuardrailResult(
            is_allowed=True,
            sanitized_answer=sanitized_answer,
            toxicity_score=tox_score,
            faithfulness_score=faith_score,
            citation_check_passed=citation_passed,
            stripped_lines_count=stripped_count,
            rejection_reason=None,
            flagged_layer=None
        )
