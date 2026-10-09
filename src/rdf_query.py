# src/rdf_query.py
"""
Hybrid retrieval: vector search over concept profiles + graph expansion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.rag_utils import embed_query
from src.rdf_graph import ThesaurusGraph
from src.rdf_index import get_client, COLLECTION_NAME

logger = logging.getLogger(__name__)


@dataclass
class RetrievedConcept:
    concept_id: str
    uri: str
    pref_label: str
    text: str
    metadata: dict
    score: float
    # Graph-expanded context
    ancestors: list[str] = field(default_factory=list)   # pref labels
    descendants: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


class HybridThesaurusRetriever:
    def __init__(self, graph: ThesaurusGraph, top_k: int = 5, n_results: int = 5):
        self.graph = graph
        self.top_k = top_k
        self.n_results = n_results
        self.client = get_client()
        # No embedding function: queries are embedded by embed_query and
        # passed as vectors, as in pdf_rag.py.
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
        )

    # --------------------------------------------------------------
    def _vector_search(self, query: str) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[list(embed_query(query))],
            n_results=self.n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            hits.append(
                {
                    "concept_id": cid,
                    "uri": meta.get("uri", ""),
                    "pref_label": meta.get("pref_label", ""),
                    "text": doc,
                    "metadata": meta,
                    "score": 1.0 - dist,  # cosine → similarity
                }
            )
        return hits

    # --------------------------------------------------------------
    def _expand_via_graph(self, uri: str) -> dict:
        return {
            "ancestors": [self.graph.pref_label_of(u) for u in self.graph.ancestors(uri)],
            "descendants": [self.graph.pref_label_of(u) for u in self.graph.descendants(uri)],
            "siblings": [self.graph.pref_label_of(u) for u in self.graph.siblings(uri)],
            "related": [self.graph.pref_label_of(u) for u in self.graph.get_concept(uri).related],
        }

    # --------------------------------------------------------------
    def retrieve(self, query: str) -> list[RetrievedConcept]:
        hits = self._vector_search(query)
        results: list[RetrievedConcept] = []
        for h in hits:
            expansion = self._expand_via_graph(h["uri"]) if h["uri"] else {}
            results.append(
                RetrievedConcept(
                    concept_id=h["concept_id"],
                    uri=h["uri"],
                    pref_label=h["pref_label"],
                    text=h["text"],
                    metadata=h["metadata"],
                    score=h["score"],
                    ancestors=expansion.get("ancestors", []),
                    descendants=expansion.get("descendants", []),
                    siblings=expansion.get("siblings", []),
                    related=expansion.get("related", []),
                )
            )
        return results

    # --------------------------------------------------------------
    def format_context(self, retrieved: list[RetrievedConcept]) -> str:
        """Format retrieved concepts as a prompt-ready context block."""
        blocks: list[str] = []
        for r in retrieved:
            lines = [
                f"### {r.pref_label}  (relevance {r.score:.2f})",
                r.text,
            ]
            if r.ancestors:
                lines.append(f"Falls under: {', '.join(r.ancestors[:5])}")
            if r.siblings:
                lines.append(f"Sibling concepts: {', '.join(r.siblings[:8])}")
            if r.related:
                lines.append(f"Related: {', '.join(r.related[:5])}")
            blocks.append("\n".join(lines))
        return "\n\n---\n\n".join(blocks)