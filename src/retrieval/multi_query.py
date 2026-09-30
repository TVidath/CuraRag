"""
Multi-Query Generator Module (Pre-Retrieval Stage)

Uses LangChain's ChatOllama integration with local LLM (gemma4:latest) and PromptTemplate
to generate 2 alternative, clinically relevant queries from an initial user query.
"""

import re
import logging
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.settings import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, NUM_GENERATED_QUERIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


MULTI_QUERY_PROMPT_TEMPLATE = """You are an expert medical AI assistant.
Given a user query about a health condition or medical topic, generate exactly {num_queries} alternative versions of the question.
Each rephrased question should approach the intent from different perspectives (e.g. layman symptom focus vs. formal clinical indicators).

Rules:
- Output ONLY the rephrased questions, one per line.
- Do NOT include numbering, bullet points, headers, or conversational introductions/conclusions.
- Keep each variation concise, precise, and medically accurate.

User Query: "{user_query}"

Alternative Questions:
"""


class MultiQueryGenerator:
    """
    Generates alternative query variations using local LLM via LangChain.
    """

    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, base_url: str = OLLAMA_BASE_URL, num_queries: int = NUM_GENERATED_QUERIES):
        self.model_name = model_name
        self.base_url = base_url
        self.num_queries = num_queries
        
        self.prompt = PromptTemplate(
            template=MULTI_QUERY_PROMPT_TEMPLATE,
            input_variables=["user_query", "num_queries"]
        )
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.3
        )

    def generate_queries(self, user_query: str) -> List[str]:
        """
        Generates alternative queries and returns a list containing:
        [original_user_query, variation_1, variation_2] (Total 3 queries)
        """
        if not user_query or not user_query.strip():
            return [user_query]

        clean_user_query = user_query.strip()
        queries = [clean_user_query]

        try:
            formatted_prompt = self.prompt.format(
                user_query=clean_user_query,
                num_queries=self.num_queries
            )
            
            response = self.llm.invoke(formatted_prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)

            # Parse line by line and clean numbering / bullet formatting
            for line in raw_text.strip().split("\n"):
                cleaned_line = line.strip()
                # Remove leading numbers, dashes, quotes, or bullets
                cleaned_line = re.sub(r"^[\d\.\-\*\>\"\']+\s*", "", cleaned_line).strip()
                cleaned_line = cleaned_line.strip('"').strip("'")
                
                if cleaned_line and cleaned_line.lower() != clean_user_query.lower() and cleaned_line not in queries:
                    queries.append(cleaned_line)
                    if len(queries) >= self.num_queries + 1:
                        break

            logging.info(f"Generated {len(queries)-1} LLM query variations for: '{clean_user_query}'")

        except Exception as e:
            logging.warning(f"Ollama Multi-Query generation encountered an issue ({e}). Falling back to original user query.")

        # Ensure we always return original query if parsing yielded fewer variations
        return queries
