# src/rdf_rag.py
"""
End-to-end RAG using the thesaurus graph + vector index.
"""
from __future__ import annotations

import logging

from rag_utils import chat_client
from rdf_graph import ThesaurusGraph
from rdf_query import HybridThesaurusRetriever
from settings import THESAURUS_RDF

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert assistant for IPBES terminology.
You are given a set of IPBES thesaurus concepts with their definitions,
hierarchical context, sibling and related concepts.

Answer the user's question using ONLY the provided concept context.
When you mention a term, use its exact prefLabel.
If a concept has a parent, feel free to explain its place in the hierarchy.
If the context is insufficient, say so explicitly.
"""


class ThesaurusRAG:
    def __init__(self, rdf_path=THESAURUS_RDF, model_name: str = "llama3.1"):
        self.graph = ThesaurusGraph(rdf_path)
        self.retriever = HybridThesaurusRetriever(self.graph, n_results=5)
        self.model_name = model_name

    def _call_llm(self, prompt: str) -> str:
        response = chat_client().chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]

    def answer(self, query: str, return_context: bool = False):
        retrieved = self.retriever.retrieve(query)
        context = self.retriever.format_context(retrieved)

        prompt = f"""User question:
{query}

Retrieved IPBES thesaurus context:
{context}

Answer:"""

        answer = self._call_llm(prompt)
        if return_context:
            return answer, context, retrieved
        return answer