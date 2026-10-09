# src/thesaurus_helper.py
"""
Thesaurus-aware helpers that improve TTL and PDF RAG retrieval and generation.

Public API used by ttl_rag.py and pdf_rag.py:
    get_thesaurus()                          -> shared ThesaurusHelper | None
    th.find_mentions(text)                   -> tuple[ConceptMention, ...]
    th.inferred_links(uri)                   -> dict[str, list[str]]
    th.expand_query(query)                   -> str
    th.score_chunk(query, chunk_text)        -> float in [0, 1]
    th.rerank_order(query, docs, dists, a)   -> list[int]  (best first)
    th.build_glossary(query, chunks)         -> str (markdown)
    th.definitions_block(query)              -> str (markdown)

Performance notes
-----------------
Finding thesaurus terms in a text used to compile and run one regex per label
(thousands) for every text. It now looks up the text's word n-grams in an
index to get a handful of candidate labels, and only runs the exact same
word-boundary regex on those. The results are identical to the old
implementation; only the cost changed (roughly linear in the text length).
Per-query work (mentions, neighbourhood terms, scoring terms) is
computed once per query rather than once per chunk.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdf_graph import ThesaurusGraph, Concept
from settings import THESAURUS_ENRICHMENT, THESAURUS_RDF

logger = logging.getLogger(__name__)


DEFAULT_RDF_PATH = THESAURUS_RDF

_TOKEN_RE = re.compile(r"\w+")

# score_chunk(): a neighbourhood term counts as present when it starts at a
# word boundary ("use" no longer matches inside "because", while "ecosystem"
# still matches "ecosystems"). Set to False for the old plain-substring test.
SCORE_REQUIRE_WORD_START = True

# Inferred links (see inferred_links): at most this many per kind, and
# concepts named in more definitions than this are too generic to count as a
# link ("Nature", "System", ...).
MAX_INFERRED_LINKS = 8
GENERIC_DEFINITION_COUNT = 30

# Endings removed from a concept's last word so "pollution" also finds
# "pollutant" and "polluter". The first matching ending wins; if the stem
# would be shorter than 4 letters the word is used whole ("cities").
_SUFFIXES = ("ations", "ation", "ions", "ion", "ants", "ant", "ies", "es", "s")


def _stem_pattern(label: str) -> re.Pattern | None:
    """Regex for a label's words at word starts, any ending on the last word."""
    words = _TOKEN_RE.findall((label or "").lower())
    if not words or len(label) < 5:
        return None
    last = words[-1]
    for suffix in _SUFFIXES:
        if last.endswith(suffix):
            if len(last) - len(suffix) >= 4:
                last = last[: -len(suffix)]
            break
    parts = [re.escape(w) for w in words[:-1]] + [re.escape(last)]
    return re.compile(r"\b" + r"\W+".join(parts) + r"\w*")


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
        if THESAURUS_ENRICHMENT.exists():
            self.graph.graph.parse(str(THESAURUS_ENRICHMENT), format="turtle")
            logger.info("Loaded thesaurus enrichment %s", THESAURUS_ENRICHMENT)
        self._lexicon: dict[str, str] = {}
        self._acronyms: set[str] = set()
        self._concept_cache: dict[str, Concept] = {}
        self._neighborhood_cache: dict[str, frozenset[str]] = {}
        self._inferred_cache: dict[str, dict[str, list[str]]] = {}
        self._def_index = None
        self._patterns: dict[str, re.Pattern] = {}
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

            self._register_label(c.pref_label, uri)
            for alt in c.alt_labels:
                if alt and not alt.startswith("http"):
                    self._register_label(alt, uri)
            for hidden in c.hidden_labels:
                if hidden and not hidden.startswith("http"):
                    self._register_label(hidden, uri)

        self._sorted_labels = sorted(self._lexicon.keys(), key=len, reverse=True)
        self._build_scan_index()
        logger.info(
            "Thesaurus lexicon: %d labels (%d acronyms) across %d concepts",
            len(self._lexicon),
            len(self._acronyms),
            len(self.graph.all_concept_uris()),
        )

    def _register_label(self, label: str, uri: str) -> None:
        """
        Add a label to the lexicon, and tag it as an acronym if it looks
        like one. Acronyms bypass the minimum-length filter in
        `find_mentions` so short but meaningful terms (NCP, TEK, ILK,
        IPLC, MSA, ...) can still be detected.
        """
        norm = self._normalize(label)
        if not norm:
            return
        self._lexicon.setdefault(norm, uri)

        stripped = label.strip()
        # Heuristic: 2–6 characters, all letters, and uppercased in the
        # original label. Catches NCP, TEK, ILK, IPLC, MSA, BOD, GDP.
        if 2 <= len(stripped) <= 6 and stripped.isalpha() and stripped.isupper():
            self._acronyms.add(norm)

    def _build_scan_index(self) -> None:
        """Index labels by their word tokens so a text only has to be checked
        against the few labels whose words actually occur in it."""
        self._label_rank = {label: i for i, label in enumerate(self._sorted_labels)}
        self._labels_by_tokens: dict[tuple[str, ...], list[str]] = {}
        self._first_token_max: dict[str, int] = {}

        for label in self._sorted_labels:
            # Short labels are only accepted when they look like acronyms.
            if len(label) < 5 and label not in self._acronyms:
                continue
            tokens = tuple(_TOKEN_RE.findall(label))
            if not tokens:
                continue
            self._labels_by_tokens.setdefault(tokens, []).append(label)
            if len(tokens) > self._first_token_max.get(tokens[0], 0):
                self._first_token_max[tokens[0]] = len(tokens)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _get_concept(self, uri: str) -> Concept:
        if uri not in self._concept_cache:
            self._concept_cache[uri] = self.graph.get_concept(uri)
        return self._concept_cache[uri]

    def _pattern(self, label: str) -> re.Pattern:
        pattern = self._patterns.get(label)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(label) + r"\b")
            self._patterns[label] = pattern
        return pattern

    # ------------------------------------------------------------------
    # (1) Mention detection + query expansion
    # ------------------------------------------------------------------
    def _scan_mentions(self, text: str) -> tuple[ConceptMention, ...]:
        """Uncached mention detection (used for chunk texts)."""
        if not text:
            return ()
        text_norm = self._normalize(text)
        tokens = _TOKEN_RE.findall(text_norm)

        # Candidate labels: any label whose word sequence occurs in the text.
        candidates: set[str] = set()
        limit = len(tokens)
        for i, token in enumerate(tokens):
            longest = self._first_token_max.get(token)
            if not longest:
                continue
            for n in range(1, min(longest, limit - i) + 1):
                labels = self._labels_by_tokens.get(tuple(tokens[i:i + n]))
                if labels:
                    candidates.update(labels)
        if not candidates:
            return ()

        # Verify with the exact word-boundary regex, longest label first,
        # so results (and their order) match the original implementation.
        found: dict[str, ConceptMention] = {}
        for label in sorted(candidates, key=self._label_rank.__getitem__):
            if not self._pattern(label).search(text_norm):
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

    @lru_cache(maxsize=512)
    def find_mentions(self, text: str) -> tuple[ConceptMention, ...]:
        """Detect explicit thesaurus terms in `text` (cached; use for queries)."""
        return self._scan_mentions(text)

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

        def add(term: str) -> None:
            norm = self._normalize(term)
            if norm and norm not in query_norm and term not in extras:
                extras.append(term)

        for m in mentions:
            c = self._get_concept(m.uri)
            # The concept's own name when the query used another label
            # ("cities" -> "Urban"), then synonyms (safest signal)
            add(m.pref_label)
            for alt in c.alt_labels[:2]:
                add(alt)
            # Then one related concept: a formal one, else an inferred one
            # sharing its name ("Pollution" -> "Soil pollution")
            for rel_uri in (c.related or self.inferred_links(m.uri)["same_name"])[:1]:
                add(self.graph.pref_label_of(rel_uri))
            if len(extras) >= max_extra_terms:
                break

        if not extras:
            return query

        expanded = query + " " + " ".join(extras[:max_extra_terms])
        logger.debug("Expanded query: %r -> %r", query, expanded)
        return expanded

    # ------------------------------------------------------------------
    # Inferred links
    # ------------------------------------------------------------------
    def _definition_index(self):
        """(normalised definition text per concept, the concepts each
        definition names in order of appearance, how many definitions name
        each concept). Built once, on first use."""
        if self._def_index is None:
            texts: dict[str, str] = {}
            names: dict[str, list[str]] = {}
            counts: Counter[str] = Counter()
            for uri in self.graph.all_concept_uris():
                text = self._normalize(" ".join(self._get_concept(uri).definitions))
                if not text:
                    continue
                texts[uri] = text
                found = [m for m in self._scan_mentions(text) if m.uri != uri]
                found.sort(key=lambda m: self._pattern(m.label).search(text).start())
                names[uri] = [m.uri for m in found]
                counts.update(set(names[uri]))
            self._def_index = (texts, names, counts)
        return self._def_index

    def inferred_links(self, uri: str) -> dict[str, list[str]]:
        """Links the thesaurus does not state, found by matching words.

        same_name     concepts whose names contain this one's ("Soil pollution")
        in_definition concepts named in this concept's definition
        defined_by    concepts whose definitions mention this one, those that
                      mention it most (and most briefly) first

        Formally linked concepts and over-generic ones are left out, so these
        only fill the gaps in broader / narrower / related. They are word
        matches, not curated, and are labelled as inferred wherever they are
        shown.
        """
        cached = self._inferred_cache.get(uri)
        if cached is not None:
            return cached

        c = self._get_concept(uri)
        texts, names, counts = self._definition_index()
        taken = {uri, *c.broader, *c.narrower, *c.related}

        def usable(u: str) -> bool:
            return u not in taken and counts[u] <= GENERIC_DEFINITION_COUNT

        pattern = _stem_pattern(c.pref_label or "")
        same_name: list[str] = []
        defined_by: list[tuple[int, int, str]] = []
        if pattern:
            for other in self.graph.all_concept_uris():
                if not usable(other):
                    continue
                oc = self._get_concept(other)
                if not oc.pref_label or oc.pref_label.startswith("http"):
                    continue
                labels = [oc.pref_label, *oc.alt_labels]
                if any(pattern.search(self._normalize(label)) for label in labels):
                    same_name.append(other)
                elif other in texts:
                    hits = len(pattern.findall(texts[other]))
                    if hits:
                        defined_by.append((-hits, len(texts[other]), other))

        same_name.sort(key=lambda u: len(self.graph.pref_label_of(u)))
        same_name = same_name[:MAX_INFERRED_LINKS]
        in_definition = [u for u in names.get(uri, []) if usable(u)][:MAX_INFERRED_LINKS]
        shown = set(same_name) | set(in_definition)
        links = {
            "same_name": same_name,
            "in_definition": in_definition,
            "defined_by": [u for _, _, u in sorted(defined_by)
                           if u not in shown][:MAX_INFERRED_LINKS],
        }
        self._inferred_cache[uri] = links
        return links

    # ------------------------------------------------------------------
    # (2) Re-ranking
    # ------------------------------------------------------------------
    def _neighborhood_terms(self, uri: str) -> frozenset[str]:
        cached = self._neighborhood_cache.get(uri)
        if cached is None:
            c = self._get_concept(uri)
            inferred = self.inferred_links(uri)
            terms: set[str] = set()
            for u in (c.broader + c.narrower + c.related + inferred["same_name"]
                      + inferred["in_definition"] + inferred["defined_by"]):
                terms.add(self._normalize(self.graph.pref_label_of(u)))
            terms.discard("")
            cached = frozenset(terms)
            self._neighborhood_cache[uri] = cached
        return cached

    @lru_cache(maxsize=256)
    def _scoring_terms(self, query: str) -> tuple[str, ...]:
        """Normalised terms for the query's concepts and their neighbours.

        Computed once per query instead of once per scored chunk.
        """
        mentions = self.find_mentions(query)
        if not mentions:
            return ()

        preferred: set[str] = set()
        for m in mentions:
            preferred.add(self._normalize(m.pref_label))
            preferred.update(self._neighborhood_terms(m.uri))
        preferred.discard("")
        return tuple(sorted(preferred))

    @staticmethod
    def _term_in(term: str, text: str) -> bool:
        """Is `term` in `text`, starting at a word boundary?

        Uses str.find (C speed) and only inspects the preceding character.
        """
        if not SCORE_REQUIRE_WORD_START:
            return term in text
        i = text.find(term)
        while i != -1:
            if i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"):
                return True
            i = text.find(term, i + 1)
        return False

    def score_chunk(self, query: str, chunk_text: str) -> float:
        """
        Bonus score in [0, 1] reflecting how many thesaurus-neighborhood
        terms appear in the chunk. Saturates at 4 hits.
        """
        terms = self._scoring_terms(query)
        if not terms or not chunk_text:
            return 0.0

        chunk_norm = self._normalize(chunk_text)
        hits = 0
        for term in terms:
            if self._term_in(term, chunk_norm):
                hits += 1
                if hits >= 4:
                    break
        return min(hits / 4.0, 1.0)

    def rerank_order(
        self,
        query: str,
        docs: list[str],
        distances: list[float],
        alpha: float,
    ) -> list[int]:
        """Indices of `docs`, best first, blending cosine similarity with the
        thesaurus score: (1 - alpha) * (1 - distance) + alpha * score.

        Ties keep the original (vector) order.
        """
        if alpha <= 0.0 or not docs:
            return list(range(len(docs)))
        sims = [max(0.0, 1.0 - d) for d in distances]
        thes = [self.score_chunk(query, doc) for doc in docs]
        combined = [(1.0 - alpha) * s + alpha * t for s, t in zip(sims, thes)]
        return sorted(range(len(docs)), key=lambda i: combined[i], reverse=True)

    # ------------------------------------------------------------------
    # (3) Glossary + definitions for the prompt
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
            # Uncached: chunk texts are large and rarely repeated, and would
            # evict the query entries from the mention cache.
            for m in self._scan_mentions(ch):
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

    def definitions_block(self, query: str) -> str:
        """Full definitions of the concepts the QUERY names, as an
        authoritative context block. Empty string if it names none.

        Matters when the corpus assumes a term is known to the reader
        (e.g. LDR uses 'NCP' without defining it).
        """
        mentions = self.find_mentions(query)
        if not mentions:
            return ""

        blocks = []
        for m in mentions:
            c = self._get_concept(m.uri)
            lines = [f"### {m.pref_label}  (IPBES thesaurus)"]
            lines.extend(c.definitions)

            label = self.graph.pref_label_of
            broader_labels = [label(u) for u in c.broader[:2]]
            narrower_labels = [label(u) for u in c.narrower[:6]]
            related_labels = [label(u) for u in c.related[:4]]
            if broader_labels:
                lines.append(f"Broader: {', '.join(broader_labels)}")
            if narrower_labels:
                lines.append(f"Narrower: {', '.join(narrower_labels)}")
            if related_labels:
                lines.append(f"Related: {', '.join(related_labels)}")

            inferred = self.inferred_links(m.uri)
            inferred_lines = [
                f"- {title}: {', '.join(label(u) for u in inferred[kind])}"
                for kind, title in (
                    ("same_name", "Concepts sharing its name"),
                    ("in_definition", "Concepts named in its definition"),
                    ("defined_by", "Concepts whose definitions mention it"),
                )
                if inferred[kind]
            ]
            if inferred_lines:
                lines.append("Inferred links (word matches in the thesaurus, "
                             "not curated relationships):")
                lines.extend(inferred_lines)
            blocks.append("\n".join(lines))

        return "## Authoritative IPBES definitions\n\n" + "\n\n---\n\n".join(blocks)


# ----------------------------------------------------------------------
# Shared instance
# ----------------------------------------------------------------------
# Building the lexicon takes a while, so ttl_rag and pdf_rag share one
# instance per process. A failed load is remembered (and logged once) rather
# than retried on every query; call reset_thesaurus() to try again.

_INSTANCE: ThesaurusHelper | None = None
_FAILED = False
_LOCK = threading.Lock()


def get_thesaurus() -> ThesaurusHelper | None:
    """Return the shared ThesaurusHelper, or None if it cannot be loaded."""
    global _INSTANCE, _FAILED
    if _INSTANCE is not None or _FAILED:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is None and not _FAILED:
            try:
                _INSTANCE = ThesaurusHelper()
            except FileNotFoundError as e:
                logger.warning("Thesaurus unavailable: %s", e)
                _FAILED = True
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load thesaurus: %s", e)
                _FAILED = True
    return _INSTANCE


def reset_thesaurus() -> None:
    """Forget the shared instance (and any remembered failure)."""
    global _INSTANCE, _FAILED
    with _LOCK:
        _INSTANCE = None
        _FAILED = False
