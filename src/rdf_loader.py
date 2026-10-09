# src/rdf_loader.py
"""
Converts the thesaurus graph into embeddable text 'profiles'.
Each profile is a self-contained description of one concept,
enriched with graph context (parents, children, related terms).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from rdf_graph import ThesaurusGraph, Concept

logger = logging.getLogger(__name__)


@dataclass
class ConceptDocument:
    """A concept plus the text we will embed and store."""
    concept_id: str
    uri: str
    pref_label: str
    text: str
    metadata: dict


def build_concept_document(graph: ThesaurusGraph, uri: str) -> ConceptDocument:
    c: Concept = graph.get_concept(uri)

    # -- Human-readable labels for linked concepts --
    broader_labels = [graph.pref_label_of(u) for u in c.broader]
    narrower_labels = [graph.pref_label_of(u) for u in c.narrower]
    related_labels = [graph.pref_label_of(u) for u in c.related]

    # -- Extract source tags from definitions: "(Source: tca)" --
    sources: list[str] = []
    for d in c.definitions:
        if "(Source:" in d:
            for chunk in d.split("(Source:"):
                if ")" in chunk:
                    sources.append(chunk.split(")")[0].strip())

    # -- Compose the profile text --
    lines: list[str] = []
    lines.append(f"Concept: {c.pref_label}")
    if c.alt_labels:
        lines.append(f"Also known as: {', '.join(c.alt_labels)}")
    if c.hidden_labels:
        lines.append(f"Hidden variants: {', '.join(c.hidden_labels)}")

    lines.append("")
    lines.append("Definitions:")
    for d in c.definitions:
        lines.append(f"  - {d}")

    if broader_labels:
        lines.append("")
        lines.append(f"Broader (parent) concept(s): {', '.join(broader_labels)}")
    if narrower_labels:
        lines.append(f"Narrower (child) concept(s): {', '.join(narrower_labels)}")
    if related_labels:
        lines.append(f"Related concepts: {', '.join(related_labels)}")
    if c.exact_match:
        lines.append(f"Exact match in external vocabulary: {', '.join(c.exact_match)}")

    text = "\n".join(lines)

    metadata = {
        "concept_id": c.concept_id,
        "uri": c.uri,
        "pref_label": c.pref_label or "",
        "alt_labels": "|".join(c.alt_labels),
        "broader_uris": "|".join(c.broader),
        "narrower_uris": "|".join(c.narrower),
        "related_uris": "|".join(c.related),
        "sources": "|".join(sorted(set(sources))),
        "source_type": "ipbes_thesaurus",
    }

    return ConceptDocument(
        concept_id=c.concept_id,
        uri=c.uri,
        pref_label=c.pref_label or c.concept_id,
        text=text,
        metadata=metadata,
    )


def build_all_documents(graph: ThesaurusGraph) -> list[ConceptDocument]:
    docs: list[ConceptDocument] = []
    for uri in graph.all_concept_uris():
        try:
            doc = build_concept_document(graph, uri)
            # Skip empties (top-level grouping nodes with no label)
            if doc.pref_label and doc.pref_label.startswith("http"):
                continue
            docs.append(doc)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s: %s", uri, e)
    logger.info("Built %d concept documents", len(docs))
    return docs