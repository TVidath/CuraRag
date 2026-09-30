"""
RAG Synthesis Generator Module

Synthesizes grounded medical answers from user queries and retrieved Parent Document contexts
using local Gemma 4 LLM via LangChain ChatOllama.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.settings import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


RAG_SYNTHESIS_PROMPT_TEMPLATE = """You are an expert, highly knowledgeable medical AI assistant.
Answer the user's medical question thoroughly, clearly, and accurately using the provided clinical source documents.

Guidelines:
- Provide a well-structured response based strictly on the facts in the source documents.
- EVERY statement line MUST cite the corresponding source context ID using the exact format [Source Document {{idx}}] (e.g., [Source Document 1], [Source Document 2]).
- Summarize the key medical insights clearly.

Source Documents:
{context}

User Question: "{user_query}"

Clinical Answer:
"""




def format_parent_context(parent_documents: List[Dict[str, Any]]) -> str:
    """
    Formats a list of Parent Document dictionaries into clean, high-signal text blocks for the LLM prompt.
    Strips internal database fields (parent_id, child_ids, char_length, word_count) to preserve context window.
    """
    if not parent_documents:
        return "No relevant parent document contexts retrieved."

    formatted_blocks = []
    for idx, doc in enumerate(parent_documents, 1):
        focus = doc.get("focus", "General Health")
        source = doc.get("source", "MedQuAD")
        question = doc.get("question", "N/A")
        answer = doc.get("answer", "").strip()

        block = (
            f"[Source Document {idx}]\n"
            f"Medical Topic: {focus}\n"
            f"Source: {source}\n"
            f"Document Question: {question}\n"
            f"Content:\n{answer}"
        )
        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)


class RAGGenerator:
    """
    Gemma 4 RAG Synthesis Generator using local Ollama instance.
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.2
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        self.prompt = PromptTemplate(
            template=RAG_SYNTHESIS_PROMPT_TEMPLATE,
            input_variables=["user_query", "context"]
        )

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

    def generate(
        self,
        user_query: str,
        parent_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes a grounded answer for the user query using the retrieved parent documents.
        """
        formatted_context = format_parent_context(parent_documents)

        if not parent_documents:
            logging.warning("Generating answer with empty parent context pool.")

        try:
            formatted_prompt = self.prompt.format(
                user_query=user_query,
                context=formatted_context
            )

            logging.info(f"Invoking Gemma 4 ({self.model_name}) for RAG synthesis...")
            response = self.llm.invoke(formatted_prompt)
            answer_text = response.content if hasattr(response, "content") else str(response)

            return {
                "user_query": user_query,
                "answer": answer_text.strip(),
                "formatted_context": formatted_context,
                "num_parent_documents": len(parent_documents),
                "model_used": self.model_name
            }

        except Exception as e:
            logging.error(f"Gemma 4 generation failed with exception: {e}")
            return {
                "user_query": user_query,
                "answer": f"Generation unavailable due to Ollama connection/model error: {e}",
                "formatted_context": formatted_context,
                "num_parent_documents": len(parent_documents),
                "model_used": self.model_name,
                "error": str(e)
            }
