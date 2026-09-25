# geo.py
from pathlib import Path
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SKOS, DCTERMS

GEO_PATH = Path("data/rdf/ipbes-geo.rdf")  # adjust to wherever you keep it

ISO3166 = URIRef("http://purl.org/dc/terms/ISO3166")

_graph = None


def _load():
    global _graph
    if _graph is None:
        g = Graph()
        g.parse(GEO_PATH, format="xml")
        _graph = g
    return _graph


def _en_labels(subject):
    g = _load()
    for p in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        for o in g.objects(subject, p):
            if getattr(o, "language", None) in (None, "en"):
                yield str(o).strip()


def country_code_for_label(label: str) -> str | None:
    """Map a free-text country label to its ISO 3166 alpha-3 code."""
    g = _load()
    needle = label.strip().lower()

    for country in g.subjects(DCTERMS.identifier, None):
        pass  # placeholder; we use the notation below instead

    for concept in g.subjects(
        RDF.type, URIRef("http://www.w3.org/2004/02/skos/core#Concept")
    ):
        # Country concepts carry a skos:notation typed as ISO3166.
        for notation in g.objects(concept, SKOS.notation):
            if getattr(notation, "datatype", None) != ISO3166:
                continue
            code = str(notation)
            for lab in _en_labels(concept):
                if lab.lower() == needle:
                    return code
            # Also try the last path segment (e.g. "KEN").
            if str(concept).rsplit("/", 1)[-1].lower() == needle:
                return code

    return None


def _expand(collection_uri) -> set[str]:
    """Recursively collect all country codes under a skos:Collection."""
    g = _load()
    codes: set[str] = set()

    stack = [collection_uri]
    seen = set()

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)

        for member in g.objects(node, SKOS.member):
            # If this member is a country concept, grab its notation.
            for notation in g.objects(member, SKOS.notation):
                if getattr(notation, "datatype", None) == ISO3166:
                    codes.add(str(notation))
                    break
            else:
                # Otherwise recurse (it's a sub-collection).
                stack.append(member)

    return codes


def country_codes_for_region(label: str) -> set[str]:
    """
    Return the set of country codes belonging to a named region or
    sub-region (e.g. 'Eastern Europe', 'Africa', 'Asia and the Pacific').
    """
    g = _load()
    needle = label.strip().lower()

    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            if lab.lower() == needle:
                return _expand(coll)

    return set()