from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, DCTERMS, SKOS, FOAF, OWL


# Fallback only; the real value comes from the file's own @prefix.
IPBES = "http://ontology.ipbes.net/report"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_literal(value) -> bool:
    return isinstance(value, Literal)


def _first(graph, subject, predicate):
    for obj in graph.objects(subject, predicate):
        return obj
    return None


def _all(graph, subject, predicate):
    return list(graph.objects(subject, predicate))


def _text(value) -> str:
    if value is None:
        return ""
    if _is_literal(value):
        return str(value).strip()
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def _local_name(uri) -> str:
    if uri is None:
        return ""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _qualifier(graph, subject) -> str:
    """
    IPBES evidence qualifier for a message: 'well established',
    'established but incomplete', or 'unresolved'.
    """
    if _first(graph, subject, URIRef(IPBES + "hasWellestablished")):
        return "well established"
    if _first(graph, subject, URIRef(IPBES + "hasEstablishedIncomplete")):
        return "established but incomplete"
    if _first(graph, subject, URIRef(IPBES + "hasUnresolved")):
        return "unresolved"
    return ""


def _submessage_sort_key(sm_uri) -> int:
    """subm/LDR18-1-SM3 -> 3."""
    name = _local_name(sm_uri)
    if "-SM" not in name:
        return 10**6
    try:
        return int(name.rsplit("-SM", 1)[-1])
    except ValueError:
        return 10**6

def _report_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Report"))


def _chapter_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Chapter"))


def _subchapter_uris(graph, subject):
    return _all(graph, subject, URIRef(IPBES + "SubChapter"))


def _illustration_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Illustration"))

def _describe_heading(graph, uri) -> str:
    """
    '1.5 Conclusion' style label for a Chapter or SubChapter URI.
    """
    if uri is None:
        return ""

    label = _text(_first(graph, uri, SKOS.prefLabel))
    ident = _text(_first(graph, uri, DCTERMS.identifier))

    parts = [p for p in (ident, label) if p]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Namespace discovery
# ---------------------------------------------------------------------------

def _discover_ipbes_namespace(graph: Graph) -> str:
    """
    Find the IPBES namespace in the parsed graph.

    Preference:
      1. A prefix literally named 'ipbes'.
      2. Any namespace whose URI contains 'ipbes.net'.

    Returns the namespace string exactly as the file uses it,
    including or excluding a trailing separator as declared.
    """
    candidates = []

    for prefix, ns in graph.namespaces():
        ns_str = str(ns)
        if prefix == "ipbes":
            return ns_str
        if "ipbes.net" in ns_str:
            candidates.append(ns_str)

    if candidates:
        return sorted(candidates, key=len)[0]

    raise RuntimeError(
        "Could not find an 'ipbes' namespace in the TTL file."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_ttl_file(ttl_path: str) -> list[dict]:
    """
    Parse an IPBES ontology TTL file into retrieval-oriented chunks.

    Chunk types:
      - background_message
      - sub_message
      - key_message
      - knowledge_gap
      - subchapter
      - illustration
      - reference
      - person
    """
    global IPBES

    path = Path(ttl_path)

    graph = Graph()
    graph.parse(path, format="ttl")

    IPBES = _discover_ipbes_namespace(graph)
    print(f"  Using IPBES namespace: {IPBES!r}")

    # -----------------------------------------------------------------
    # Document-level metadata
    # -----------------------------------------------------------------

    report_uri = None
    for _, _, obj in graph.triples(
        (None, URIRef(IPBES + "Report"), None)
    ):
        report_uri = obj
        break

    doc_meta = {
        "document_type": "ttl",
        "title": "",
        "date": "",
        "language": "",
        "country": "",
        "subtype": "",
        "number": "",
    }

    if report_uri is not None:
        title = _text(_first(graph, report_uri, SKOS.prefLabel))
        alt_title = _text(_first(graph, report_uri, SKOS.altLabel))
        year = _text(_first(graph, report_uri, URIRef(IPBES + "year")))
        date = _text(_first(graph, report_uri, DCTERMS.date))

        doc_meta["title"] = title or alt_title
        doc_meta["date"] = date or year
        doc_meta["language"] = _text(
            _first(graph, report_uri, DCTERMS.language)
        )
        doc_meta["subtype"] = _local_name(report_uri)

    chunks: list[dict] = []
    chunk_index = 0

    def emit(
        subject,
        chunk_type: str,
        heading: str,
        body: str,
        subchapter_uris: list,
        extra_meta: dict | None = None,
    ):
        nonlocal chunk_index

        body = (body or "").strip()
        if not body:
            return

        lines = []

        if doc_meta["title"]:
            lines.append(f"Document: {doc_meta['title']}")
        if doc_meta["date"]:
            lines.append(f"Date: {doc_meta['date']}")

        # Chapter heading
        chapter_uri = _first(graph, subject, URIRef(IPBES + "Chapter"))
        chapter_heading = _describe_heading(graph, chapter_uri)
        if chapter_heading:
            lines.append(f"Chapter: {chapter_heading}")

        # Subchapter headings
        for sub_uri in subchapter_uris:
            sub_heading = _describe_heading(graph, sub_uri)
            if sub_heading:
                lines.append(f"Subchapter: {sub_heading}")

        if heading:
            lines.append(f"Heading: {heading}")

        if extra_meta:
            for key, value in extra_meta.items():
                if value:
                    lines.append(f"{key}: {value}")

        lines.append("")
        lines.append(body)

        chunk_id = f"{path.stem}-{chunk_type}-{_local_name(subject)}"

        chunks.append(
            {
                "text": "\n".join(lines),
                "source_file": path.name,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "document_type": doc_meta["document_type"],
                "title": doc_meta["title"],
                "date": doc_meta["date"],
                "language": doc_meta["language"],
                "country": doc_meta["country"],
                "subtype": doc_meta["subtype"],
                "number": doc_meta["number"],
                "division": chapter_heading,
                "subdivision": " | ".join(
                    _describe_heading(graph, s)
                    for s in subchapter_uris
                    if _describe_heading(graph, s)
                ),
                "paragraph": "",
                "eId": _local_name(subject),
                "xpath": str(subject),
            }
        )

        chunk_index += 1

    # -----------------------------------------------------------------
    # BackgroundMessage — its own description + all SubMessage bodies
    # -----------------------------------------------------------------

    bgm_type = URIRef(IPBES + "BackgroundMessage")

    for bgm in graph.subjects(RDF.type, bgm_type):
        description = _text(
            _first(graph, bgm, URIRef(IPBES + "hasDescription"))
        )

        sm_uris = sorted(
            _all(graph, bgm, URIRef(IPBES + "SubMessage")),
            key=_submessage_sort_key,
        )

        sm_bodies = []
        subchapters = []

        for sm in sm_uris:
            sm_desc = _text(
                _first(graph, sm, URIRef(IPBES + "hasDescription"))
            )
            if sm_desc:
                sm_bodies.append(f"[{_local_name(sm)}] {sm_desc}")

            subchapters.extend(_subchapter_uris(graph, sm))

        combined = description
        if sm_bodies:
            combined = (
                description + "\n\n" + "\n\n".join(sm_bodies)
            ).strip()

        emit(
            subject=bgm,
            chunk_type="bgm",
            heading="",
            body=combined,
            subchapter_uris=subchapters,
            extra_meta={
                "Identifier": _text(
                    _first(graph, bgm, DCTERMS.identifier)
                ),
                "Qualifier": _qualifier(graph, bgm),
            },
        )

    # -----------------------------------------------------------------
    # SubMessage — individual
    # -----------------------------------------------------------------

    sm_type = URIRef(IPBES + "SubMessage")

    for sm in graph.subjects(RDF.type, sm_type):
        description = _text(
            _first(graph, sm, URIRef(IPBES + "hasDescription"))
        )

        ill_uri = _first(graph, sm, URIRef(IPBES + "Illustration"))
        ill_label = ""
        if ill_uri is not None:
            ill_label = _text(
                _first(graph, ill_uri, SKOS.prefLabel)
            )

        emit(
            subject=sm,
            chunk_type="subm",
            heading=ill_label,
            body=description,
            subchapter_uris=_subchapter_uris(graph, sm),
            extra_meta={
                "Identifier": _text(
                    _first(graph, sm, DCTERMS.identifier)
                ),
                "Qualifier": _qualifier(graph, sm),
            },
        )

    # -----------------------------------------------------------------
    # KeyMessage — headline + narrative
    # -----------------------------------------------------------------

    km_type = URIRef(IPBES + "KeyMessage")

    for km in graph.subjects(RDF.type, km_type):
        headline = _text(_first(graph, km, SKOS.prefLabel))
        description = _text(
            _first(graph, km, URIRef(IPBES + "hasDescription"))
        )

        body = headline
        if description:
            body = f"{headline}\n\n{description}".strip()

        emit(
            subject=km,
            chunk_type="key",
            heading="",
            body=body,
            subchapter_uris=[],
            extra_meta={
                "Identifier": _text(
                    _first(graph, km, DCTERMS.identifier)
                ),
            },
        )

    # -----------------------------------------------------------------
    # KnowledgeGap
    # -----------------------------------------------------------------

    kg_type = URIRef(IPBES + "KnowledgeGap")

    for kg in graph.subjects(RDF.type, kg_type):
        description = _text(
            _first(graph, kg, URIRef(IPBES + "hasDescription"))
        )

        emit(
            subject=kg,
            chunk_type="kg",
            heading="",
            body=description,
            subchapter_uris=[],
            extra_meta={
                "Identifier": _text(
                    _first(graph, kg, DCTERMS.identifier)
                ),
            },
        )

    # -----------------------------------------------------------------
    # SubChapter — the big narrative blocks
    # -----------------------------------------------------------------

    sch_type = URIRef(IPBES + "SubChapter")

    for sch in graph.subjects(RDF.type, sch_type):
        description = _text(
            _first(graph, sch, URIRef(IPBES + "hasDescription"))
        )

        emit(
            subject=sch,
            chunk_type="sch",
            heading=_describe_heading(graph, sch),
            body=description,
            subchapter_uris=[sch],
            extra_meta={},
        )

    # -----------------------------------------------------------------
    # Illustration — only if it has a caption
    # -----------------------------------------------------------------

    il_type = URIRef(IPBES + "Illustration")

    for il in graph.subjects(RDF.type, il_type):
        ident = _text(_first(graph, il, DCTERMS.identifier))
        caption = _text(_first(graph, il, SKOS.prefLabel))

        if not caption:
            continue

        emit(
            subject=il,
            chunk_type="il",
            heading=ident,
            body=caption,
            subchapter_uris=[],
            extra_meta={},
        )

    # -----------------------------------------------------------------
    # Reference
    # -----------------------------------------------------------------

    ref_type = URIRef(IPBES + "Reference")

    for ref in graph.subjects(RDF.type, ref_type):
        doi = _text(_first(graph, ref, URIRef(IPBES + "hasDoi")))
        zotero = _text(_first(graph, ref, OWL.sameAs))

        body = f"DOI: {doi}" if doi else ""
        if zotero:
            body = (body + f"\nZotero: {zotero}").strip()

        if not body:
            continue

        emit(
            subject=ref,
            chunk_type="ref",
            heading="",
            body=body,
            subchapter_uris=_subchapter_uris(graph, ref),
            extra_meta={},
        )

    # -----------------------------------------------------------------
    # Person
    # -----------------------------------------------------------------

    for person in graph.subjects(RDF.type, FOAF.Person):
        name = _text(_first(graph, person, SKOS.prefLabel))
        if not name:
            continue

        country = _text(
            _first(graph, person, URIRef(IPBES + "country"))
        )

        roles = []
        for role in ("ca", "la", "cl", "re", "fl", "cs"):
            value = _text(
                _first(graph, person, URIRef(IPBES + role))
            )
            if value:
                roles.append(value)

        body_parts = []
        if country:
            body_parts.append(f"Country: {country}")
        if roles:
            body_parts.append("Roles: " + "; ".join(roles))

        body = "\n".join(body_parts)

        emit(
            subject=person,
            chunk_type="person",
            heading=name,
            body=body,
            subchapter_uris=[],
            extra_meta={},
        )

    return chunks