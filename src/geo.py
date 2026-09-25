# geo.py
import re
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SKOS, DCTERMS



GEO_PATH = Path("data/rdf/ipbes-geo.rdf")
ISO3166 = URIRef("http://purl.org/dc/terms/ISO3166")

_graph = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load() -> Graph:
    global _graph
    if _graph is None:
        g = Graph()
        g.parse(GEO_PATH, format="xml")
        _graph = g
    return _graph


def _labels_of_type(subject, predicate) -> list[str]:
    g = _load()
    out = []
    for o in g.objects(subject, predicate):
        lang = getattr(o, "language", None)
        if lang in (None, "en"):
            out.append(str(o).strip())
    return out


def _en_labels(subject):
    g = _load()
    for p in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        for o in g.objects(subject, p):
            if getattr(o, "language", None) in (None, "en"):
                yield str(o).strip()


# ---------------------------------------------------------------------------
# Country concepts
# ---------------------------------------------------------------------------

def _country_concepts():
    """Yield (concept, iso_code) for every country in the geo file."""
    g = _load()
    for concept in g.subjects(
        RDF.type, URIRef("http://www.w3.org/2004/02/skos/core#Concept")
    ):
        for notation in g.objects(concept, SKOS.notation):
            if getattr(notation, "datatype", None) == ISO3166:
                yield concept, str(notation)
                break


def country_code_for_label(label: str) -> str | None:
    """Map a free-text country label to its ISO 3166 alpha-3 code."""
    if not label:
        return None
    needle = label.strip().lower()

    for concept, code in _country_concepts():
        for lab in _en_labels(concept):
            if lab.lower() == needle:
                return code
        if str(concept).rsplit("/", 1)[-1].lower() == needle:
            return code

    return None


def country_label_for_code(code: str) -> str | None:
    """
    Map an ISO 3166 alpha-3 code back to the plain English country name
    that the TTL files actually use for ipbes:country.

    Preference:
      1. skos:hiddenLabel   (e.g. 'Tanzania', 'Netherlands', 'Gambia')
      2. skos:prefLabel with any trailing parenthetical stripped
    """
    code = (code or "").upper()
    for concept, c in _country_concepts():
        if c != code:
            continue
        hidden = _labels_of_type(concept, SKOS.hiddenLabel)
        if hidden:
            return hidden[0]
        pref = _labels_of_type(concept, SKOS.prefLabel)
        if pref:
            short = re.sub(r"\s*\([^)]*\)\s*$", "", pref[0]).strip()
            return short or pref[0]
        return None
    return None


# ---------------------------------------------------------------------------
# Region / collection expansion
# ---------------------------------------------------------------------------

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
            for notation in g.objects(member, SKOS.notation):
                if getattr(notation, "datatype", None) == ISO3166:
                    codes.add(str(notation))
                    break
            else:
                stack.append(member)

    return codes


def country_codes_for_region(label: str) -> set[str]:
    """
    Return ISO codes for a named region or sub-region.

    Matching is tried in order:
      1. Exact label match.
      2. Prefix match — 'east africa' -> 'East Africa and adjacent islands'.
      3. Substring match, but skip continent-level collections unless the
         needle itself names that continent.
    """
    g = _load()
    needle = label.strip().lower()

    # 1. Exact.
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            if lab.lower() == needle:
                return _expand(coll)

    # 2. Prefix.
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            if lab.lower().startswith(needle):
                return _expand(coll)

    # 3. Substring (skip broad continents).
    broad = {
        "africa", "americas", "asia", "europe",
        "europe and central asia", "asia and the pacific",
    }
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            lab_l = lab.lower()
            if needle in lab_l and lab_l not in broad:
                return _expand(coll)

    return set()


# ---------------------------------------------------------------------------
# Top-level: question -> set of country names
# ---------------------------------------------------------------------------

# Longest names first so that 'east africa' wins over 'africa'.
_REGION_NAMES = [
    "east africa and adjacent islands",
    "east africa",
    "central and western europe",
    "europe and central asia",
    "asia and the pacific",
    "western europe",
    "eastern europe",
    "central asia",
    "north africa",
    "north america",
    "north-east asia",
    "south america",
    "south asia",
    "south-east asia",
    "southern africa",
    "west africa",
    "western asia",
    "central africa",
    "mesoamerica",
    "caribbean",
    "oceania",
    "africa",
    "americas",
    "asia",
    "europe",
]


def infer_country_names(question: str) -> set[str]:
    """
    Return the country *names* implied by the question, matching what is
    stored in the Chroma metadata (e.g. 'Kenya', 'Tanzania').

    Region -> set of ISO codes -> set of names.
    """
    q = question.lower()
    codes: set[str] = set()

    # 1. Region / sub-region: pick the longest matching name.
    matches = [n for n in _REGION_NAMES if n in q]
    matches.sort(key=len, reverse=True)
    for name in matches:
        codes = country_codes_for_region(name)
        if codes:
            break

    # 2. Individual country: try unigrams and bigrams from the question.
    if not codes:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", question)
        candidates = list(tokens) + [
            f"{tokens[i]} {tokens[i + 1]}"
            for i in range(len(tokens) - 1)
        ]
        for cand in candidates:
            code = country_code_for_label(cand)
            if code:
                codes = {code}
                break

    if not codes:
        return set()

    names = set()
    for c in codes:
        label = country_label_for_code(c)
        if label:
            names.add(label)
    return names