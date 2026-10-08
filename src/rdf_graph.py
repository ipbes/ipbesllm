# src/rdf_graph.py
"""
Wraps the IPBES thesaurus RDF as a queryable graph.
Provides concept lookup and traversal along SKOS relations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS

logger = logging.getLogger(__name__)

# Namespaces used in the IPBES file
IPBES_NS = "https://ibok.ipbes.net/thesaurus/"
DEF_NS = "http://ibok.ipbes.net/thesaurus/"


@dataclass
class Concept:
    """A normalized view of a SKOS concept."""
    uri: str
    pref_label: str | None = None
    alt_labels: list[str] = field(default_factory=list)
    hidden_labels: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    broader: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    exact_match: list[str] = field(default_factory=list)

    @property
    def concept_id(self) -> str:
        """Return trailing numeric/identifier part of the URI."""
        return self.uri.rstrip("/").rsplit("/", 1)[-1]


class ThesaurusGraph:
    """
    In-memory RDFLib graph of the IPBES thesaurus.
    Loads once, serves many queries.
    """

    def __init__(self, rdf_path: str | Path):
        self.rdf_path = Path(rdf_path)
        self.graph = Graph()
        self._load()

    # ---------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------
    def _load(self) -> None:
        if not self.rdf_path.exists():
            raise FileNotFoundError(f"RDF file not found: {self.rdf_path}")

        logger.info("Loading RDF graph from %s", self.rdf_path)
        # rdflib auto-detects format; xml is the default for .rdf
        self.graph.parse(str(self.rdf_path), format="xml")
        logger.info("Loaded %d triples", len(self.graph))

    # ---------------------------------------------------------------
    # Concept extraction
    # ---------------------------------------------------------------
    def all_concept_uris(self) -> list[str]:
        """All subjects typed as skos:Concept, plus the 'Description' ones
        that have a skos:prefLabel (some IPBES entries omit rdf:type)."""
        uris: set[str] = set()
        for s in self.graph.subjects(RDF.type, SKOS.Concept):
            uris.add(str(s))
        # Some entries (e.g. 9195) lack rdf:type but have skos:prefLabel
        for s in self.graph.subjects(SKOS.prefLabel, None):
            if str(s).startswith(IPBES_NS) and str(s) != IPBES_NS:
                uris.add(str(s))
        return sorted(uris)

    def get_concept(self, uri: str) -> Concept:
        """Build a normalized Concept dataclass for a given URI."""
        subj = URIRef(uri)
        c = Concept(uri=uri)

        for label in self.graph.objects(subj, SKOS.prefLabel):
            if isinstance(label, Literal):
                c.pref_label = str(label)

        for label in self.graph.objects(subj, SKOS.altLabel):
            if isinstance(label, Literal):
                c.alt_labels.append(str(label))

        for label in self.graph.objects(subj, SKOS.hiddenLabel):
            if isinstance(label, Literal):
                c.hidden_labels.append(str(label))

        for defn_ref in self.graph.objects(subj, SKOS.definition):
            # Definitions are separate resources with an rdf:value
            for val in self.graph.objects(defn_ref, RDF.value):
                if isinstance(val, Literal):
                    c.definitions.append(str(val))

        c.broader = [str(o) for o in self.graph.objects(subj, SKOS.broader)]
        c.narrower = [str(o) for o in self.graph.objects(subj, SKOS.narrower)]
        c.related = [str(o) for o in self.graph.objects(subj, SKOS.related)]
        c.exact_match = [str(o) for o in self.graph.objects(subj, SKOS.exactMatch)]

        return c

    # ---------------------------------------------------------------
    # Traversal helpers
    # ---------------------------------------------------------------
    def pref_label_of(self, uri: str) -> str:
        """Best-effort display label for any URI."""
        for label in self.graph.objects(URIRef(uri), SKOS.prefLabel):
            return str(label)
        # Fall back to trailing ID
        return uri.rstrip("/").rsplit("/", 1)[-1]

    def ancestors(self, uri: str, max_depth: int = 10) -> list[str]:
        """Walk up skos:broader links, returns list of ancestor URIs."""
        seen: list[str] = []
        frontier = [uri]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = []
            for u in frontier:
                for parent in self.graph.objects(URIRef(u), SKOS.broader):
                    p = str(parent)
                    if p not in seen:
                        seen.append(p)
                        next_frontier.append(p)
            frontier = next_frontier
            depth += 1
        return seen

    def descendants(self, uri: str, max_depth: int = 10) -> list[str]:
        """Walk down skos:narrower links."""
        seen: list[str] = []
        frontier = [uri]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = []
            for u in frontier:
                for child in self.graph.objects(URIRef(u), SKOS.narrower):
                    c = str(child)
                    if c not in seen:
                        seen.append(c)
                        next_frontier.append(c)
            frontier = next_frontier
            depth += 1
        return seen

    def siblings(self, uri: str) -> list[str]:
        """Other concepts sharing the same direct broader term."""
        siblings: set[str] = set()
        for parent in self.graph.objects(URIRef(uri), SKOS.broader):
            for sib in self.graph.subjects(SKOS.broader, parent):
                s = str(sib)
                if s != uri:
                    siblings.add(s)
        return sorted(siblings)