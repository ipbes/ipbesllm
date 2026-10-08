# src/thesaurus_helper.py
"""
Thesaurus-aware helpers that improve TTL RAG retrieval and generation.

Three public methods used by ttl_rag.py:
    - expand_query(query)            -> str
    - score_chunk(query, chunk_text) -> float in [0, 1]
    - build_glossary(query, chunks)  -> str (markdown)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdf_graph import ThesaurusGraph, Concept

logger = logging.getLogger(__name__)


# Resolve relative to the project root, regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RDF_PATH = _PROJECT_ROOT / "data" / "rdf" / "ipbes-thesaurus.rdf"


@dataclass(frozen=True)
class ConceptMention:
    uri: str
    label: str
    pref_label: str
    definition: str


class ThesaurusHelper:
    """Fast lookup + expansion using the IPBES thesaurus graph."""

    def __init__(self, rdf_path: str | Path = DEFAULT_RDF_PATH):
        rdf_path = Path(rdf_path)
        if not rdf_path.exists():
            raise FileNotFoundError(f"Thesaurus RDF not found: {rdf_path}")

        self.graph = ThesaurusGraph(rdf_path)
        self._lexicon: dict[str, str] = {}
        self._concept_cache: dict[str, Concept] = {}
        self._build_lexicon()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_lexicon(self) -> None:
        for uri in self.graph.all_concept_uris():
            c = self._get_concept(uri)
            if not c.pref_label:
                continue
            # Skip top-level grouping nodes whose label is a URI
            if c.pref_label.startswith("http"):
                continue

            self._lexicon[self._normalize(c.pref_label)] = uri
            for alt in c.alt_labels:
                if alt and not alt.startswith("http"):
                    self._lexicon.setdefault(self._normalize(alt), uri)
            for hidden in c.hidden_labels:
                if hidden and not hidden.startswith("http"):
                    self._lexicon.setdefault(self._normalize(hidden), uri)

        self._sorted_labels = sorted(self._lexicon.keys(), key=len, reverse=True)
        logger.info(
            "Thesaurus lexicon: %d labels across %d concepts",
            len(self._lexicon), len(self.graph.all_concept_uris()),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _get_concept(self, uri: str) -> Concept:
        if uri not in self._concept_cache:
            self._concept_cache[uri] = self.graph.get_concept(uri)
        return self._concept_cache[uri]

    # ------------------------------------------------------------------
    # (1) Query expansion
    # ------------------------------------------------------------------
    @lru_cache(maxsize=1024)
    def find_mentions(self, text: str) -> tuple[ConceptMention, ...]:
        """Detect explicit thesaurus terms in `text`."""
        if not text:
            return ()
        text_norm = self._normalize(text)
        found: dict[str, ConceptMention] = {}

        for label in self._sorted_labels:
            if len(label) < 5:            # avoid 4-char false positives
                continue
            pattern = r"\b" + re.escape(label) + r"\b"
            if not re.search(pattern, text_norm):
                continue
            uri = self._lexicon[label]
            if uri in found:
                continue
            c = self._get_concept(uri)
            found[uri] = ConceptMention(
                uri=uri,
                label=label,
                pref_label=c.pref_label or label,
                definition=c.definitions[0] if c.definitions else "",
            )
        return tuple(found.values())

    def expand_query(self, query: str, max_extra_terms: int = 6) -> str:
        """
        Return the query augmented with synonyms of any detected concepts.
        Original query is always preserved as the prefix so the embedding
        stays anchored to the user's intent.
        """
        mentions = self.find_mentions(query)
        if not mentions:
            return query

        query_norm = self._normalize(query)
        extras: list[str] = []

        for m in mentions:
            c = self._get_concept(m.uri)
            # Synonyms first (safest signal)
            for alt in c.alt_labels[:2]:
                alt_norm = self._normalize(alt)
                if alt_norm and alt_norm not in query_norm and alt not in extras:
                    extras.append(alt)
            # Then one related concept
            for rel_uri in c.related[:1]:
                rel_label = self.graph.pref_label_of(rel_uri)
                rel_norm = self._normalize(rel_label)
                if rel_label and rel_norm not in query_norm and rel_label not in extras:
                    extras.append(rel_label)
            if len(extras) >= max_extra_terms:
                break

        if not extras:
            return query

        expanded = query + " " + " ".join(extras[:max_extra_terms])
        logger.debug("Expanded query: %r -> %r", query, expanded)
        return expanded

    # ------------------------------------------------------------------
    # (2) Re-ranking
    # ------------------------------------------------------------------
    def _neighborhood_terms(self, uri: str) -> set[str]:
        c = self._get_concept(uri)
        terms: set[str] = set()
        for u in c.broader + c.narrower + c.related:
            terms.add(self._normalize(self.graph.pref_label_of(u)))
        terms.discard("")
        return terms

    def score_chunk(self, query: str, chunk_text: str) -> float:
        """
        Bonus score in [0, 1] reflecting how many thesaurus-neighborhood
        terms appear in the chunk. Saturates at 4 hits.
        """
        mentions = self.find_mentions(query)
        if not mentions or not chunk_text:
            return 0.0

        preferred: set[str] = set()
        for m in mentions:
            preferred.add(self._normalize(m.pref_label))
            preferred.update(self._neighborhood_terms(m.uri))
        preferred.discard("")

        if not preferred:
            return 0.0

        chunk_norm = self._normalize(chunk_text)
        hits = sum(1 for term in preferred if term in chunk_norm)
        return min(hits / 4.0, 1.0)

    # ------------------------------------------------------------------
    # (3) Glossary injection
    # ------------------------------------------------------------------
    def build_glossary(
        self,
        query: str,
        chunks: list[str],
        max_entries: int = 6,
    ) -> str:
        """
        Markdown glossary of thesaurus terms found in query or chunks.
        Empty string if none found.
        """
        seen: dict[str, ConceptMention] = {}

        for m in self.find_mentions(query):
            seen[m.uri] = m

        for ch in chunks:
            if len(seen) >= max_entries:
                break
            for m in self.find_mentions(ch):
                if m.uri not in seen:
                    seen[m.uri] = m
                if len(seen) >= max_entries:
                    break

        entries = [m for m in seen.values() if m.definition]
        if not entries:
            return ""

        lines = ["## IPBES Thesaurus Definitions", ""]
        for m in entries[:max_entries]:
            lines.append(f"**{m.pref_label}** — {m.definition}")
            lines.append("")
        return "\n".join(lines).strip()