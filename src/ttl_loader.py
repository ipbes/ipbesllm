from pathlib import Path
from urllib.parse import urlparse

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, DCTERMS, SKOS, FOAF, OWL


IPBES = "http://ontology.ipbes.net/report/"


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
    # URIRef — fall back to the local name.
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def _local_name(uri: URIRef) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def _qualifier(graph, subject) -> str:
    """
    Evidence qualifier for SubMessage/BackgroundMessage prose.
    These come from IPBES's own confidence language.
    """
    if _first(graph, subject, URIRef(IPBES + "hasWellestablished")):
        return "well established"
    if _first(graph, subject, URIRef(IPBES + "hasEstablishedIncomplete")):
        return "established but incomplete"
    if _first(graph, subject, URIRef(IPBES + "hasUnresolved")):
        return "unresolved"
    return ""


def _submessage_sort_key(sm_uri: URIRef) -> int:
    """subm/LDR18-1-SM3 -> 3; used to order SubMessages under a BackgroundMessage."""
    name = _local_name(sm_uri)
    if "-SM" not in name:
        return 10**6
    return int(name.rsplit("-SM", 1)[-1])


def _report_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Report"))


def _chapter_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Chapter"))


def _subchapter_uris(graph, subject):
    return _all(graph, subject, URIRef(IPBES + "SubChapter"))


def _illustration_uri(graph, subject):
    return _first(graph, subject, URIRef(IPBES + "Illustration"))


def _describe_heading(graph, uri) -> str:
    if uri is None:
        return ""
    label = _first(graph, uri, SKOS.prefLabel)
    ident = _first(graph, uri, DCTERMS.identifier)
    parts = [p for p in (_text(ident), _text(label)) if p]
    return " ".join(parts)


def _build_context_lines(
    doc_meta: dict,
    subchapter_uris: list,
    graph,
    extra: dict | None = None,
) -> list[str]:
    lines = []

    if doc_meta.get("title"):
        lines.append(f"Document: {doc_meta['title']}")
    if doc_meta.get("date"):
        lines.append(f"Date: {doc_meta['date']}")

    for sub_uri in subchapter_uris:
        heading = _describe_heading(graph, sub_uri)
        if heading:
            lines.append(f"Subchapter: {heading}")

    if extra:
        for key, value in extra.items():
            if value:
                lines.append(f"{key}: {value}")

    return lines


def parse_ttl_file(ttl_path: str) -> list[dict]:
    """
    Parse an IPBES ontology TTL file into retrieval-oriented chunks.

    Chunk types mirror the TTL's RDF classes:
      - background_message
      - sub_message
      - key_message
      - knowledge_gap
      - subchapter
      - illustration
      - reference
      - person
    """
    path = Path(ttl_path)

    graph = Graph()
    graph.parse(path, format="ttl")

    # ------------------------------------------------------------
    # Document-level metadata, harvested from the report URI.
    # We discover it via the first ipbes:Report triple.
    # ------------------------------------------------------------

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
        doc_meta["title"] = _text(_first(graph, report_uri, SKOS.prefLabel))
        doc_meta["date"] = _text(_first(graph, report_uri, DCTERMS.date))
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

        context_lines = _build_context_lines(
            doc_meta, subchapter_uris, graph
        )

        if heading:
            context_lines.append(f"Heading: {heading}")

        if extra_meta:
            for key, value in extra_meta.items():
                if value:
                    context_lines.append(f"{key}: {value}")

        context_lines.append("")
        context_lines.append(body)

        chunk_id = f"{path.stem}-{chunk_type}-{_local_name(subject)}"

        chunks.append(
            {
                "text": "\n".join(context_lines),
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
                "division": (
                    _local_name(_chapter_uri(graph, subject))
                    if _chapter_uri(graph, subject)
                    else ""
                ),
                "subdivision": " | ".join(
                    _describe_heading(graph, s)
                    for s in subchapter_uris
                ),
                "paragraph": "",
                "eId": _local_name(subject),
                "xpath": str(subject),
            }
        )
        chunk_index += 1

    # ------------------------------------------------------------
    # BackgroundMessage — narrative summary + its SubMessages
    # ------------------------------------------------------------

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
        for sm in sm_uris:
            sm_desc = _text(
                _first(graph, sm, URIRef(IPBES + "hasDescription"))
            )
            if sm_desc:
                sm_bodies.append(
                    f"[{_local_name(sm)}] {sm_desc}"
                )

        combined = description
        if sm_bodies:
            combined = (description + "\n\n" + "\n\n".join(sm_bodies)).strip()

        subchapters = []
        for sm in sm_uris:
            subchapters.extend(_subchapter_uris(graph, sm))

        emit(
            subject=bgm,
            chunk_type="background_message",
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

    # ------------------------------------------------------------
    # SubMessage — each one individually
    # ------------------------------------------------------------

    sm_type = URIRef(IPBES + "SubMessage")

    for sm in graph.subjects(RDF.type, sm_type):
        description = _text(
            _first(graph, sm, URIRef(IPBES + "hasDescription"))
        )

        ill = _illustration_uri(graph, sm)
        ill_label = _text(_first(graph, ill, SKOS.prefLabel)) if ill else ""

        emit(
            subject=sm,
            chunk_type="sub_message",
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

    # ------------------------------------------------------------
    # KeyMessage — headline + supporting narrative
    # ------------------------------------------------------------

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
            chunk_type="key_message",
            heading="",
            body=body,
            subchapter_uris=[],
            extra_meta={
                "Identifier": _text(
                    _first(graph, km, DCTERMS.identifier)
                ),
            },
        )

    # ------------------------------------------------------------
    # KnowledgeGap — short, single-sentence facts
    # ------------------------------------------------------------

    kg_type = URIRef(IPBES + "KnowledgeGap")

    for kg in graph.subjects(RDF.type, kg_type):
        description = _text(
            _first(graph, kg, URIRef(IPBES + "hasDescription"))
        )

        emit(
            subject=kg,
            chunk_type="knowledge_gap",
            heading="",
            body=description,
            subchapter_uris=[],
            extra_meta={
                "Identifier": _text(
                    _first(graph, kg, DCTERMS.identifier)
                ),
            },
        )

    # ------------------------------------------------------------
    # SubChapter — the big narrative blocks
    # ------------------------------------------------------------

    sch_type = URIRef(IPBES + "SubChapter")

    for sch in graph.subjects(RDF.type, sch_type):
        description = _text(
            _first(graph, sch, URIRef(IPBES + "hasDescription"))
        )

        chapter = _chapter_uri(graph, sch)

        emit(
            subject=sch,
            chunk_type="subchapter",
            heading=_describe_heading(graph, sch),
            body=description,
            subchapter_uris=[sch],
            extra_meta={
                "Chapter": _local_name(chapter) if chapter else "",
            },
        )

    # ------------------------------------------------------------
    # Illustration — figure/box/table captions
    # ------------------------------------------------------------

    il_type = URIRef(IPBES + "Illustration")

    for il in graph.subjects(RDF.type, il_type):
        ident = _text(_first(graph, il, DCTERMS.identifier))
        caption = _text(_first(graph, il, SKOS.prefLabel))

        # Skip illustrations with no caption at all — common when the
        # ontology only records the identifier (e.g. Box SPM1).
        if not caption:
            continue

        emit(
            subject=il,
            chunk_type="illustration",
            heading=ident,
            body=caption,
            subchapter_uris=[],
            extra_meta={},
        )

    # ------------------------------------------------------------
    # Reference — bibliography entries
    # ------------------------------------------------------------

    ref_type = URIRef(IPBES + "Reference")

    for ref in graph.subjects(RDF.type, ref_type):
        doi = _text(_first(graph, ref, URIRef(IPBES + "hasDoi")))
        zotero = _text(_first(graph, ref, OWL.sameAs))

        # Description only — no free text in this class in the sample.
        body = f"DOI: {doi}" if doi else ""
        if zotero:
            body = (body + f"\nZotero: {zotero}").strip()

        if not body:
            continue

        emit(
            subject=ref,
            chunk_type="reference",
            heading="",
            body=body,
            subchapter_uris=_subchapter_uris(graph, ref),
            extra_meta={
                "Chapter": _local_name(_chapter_uri(graph, ref))
                if _chapter_uri(graph, ref)
                else "",
            },
        )

    # ------------------------------------------------------------
    # Person — author bios
    # ------------------------------------------------------------

    person_type = FOAF.Person

    for person in graph.subjects(RDF.type, person_type):
        name = _text(_first(graph, person, SKOS.prefLabel))
        if not name:
            continue

        country = _text(_first(graph, person, URIRef(IPBES + "country")))

        # Role strings.
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