"""Country and region lookup backed by the IPBES geography RDF."""
import re
from functools import lru_cache
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF, SKOS
from settings import GEO_RDF

GEO_PATH = GEO_RDF

ISO3166_DATATYPE = "http://purl.org/dc/terms/ISO3166"
_graph = None

# Explicit colloquial aliases absent from, or inconsistently represented in, RDF.
# Each alias resolves to an RDF-backed ISO alpha-3 code; no concepts are fabricated.
_COUNTRY_ALIASES = {
    "south korea": "KOR", "north korea": "PRK", "iran": "IRN",
    "syria": "SYR", "vietnam": "VNM", "laos": "LAO",
    "bolivia": "BOL", "venezuela": "VEN", "moldova": "MDA",
    "palestine": "PSE", "vatican": "VAT", "vatican city": "VAT",
    "tanzania": "TZA", "netherlands": "NLD",
}

def _load() -> Graph:
    global _graph
    if _graph is None:
        graph = Graph()
        graph.parse(GEO_PATH, format="xml")
        _graph = graph
    return _graph


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _strip_parenthetical(label: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", _clean(label)).strip()


def _key(value: str) -> str:
    return _strip_parenthetical(_clean(value)).casefold()


def _labels(subject, predicates=(SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel)):
    graph = _load()
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            if getattr(obj, "language", None) in (None, "en"):
                label = _clean(obj)
                if label:
                    yield label


def _country_concepts():
    """Yield (concept, ISO alpha-3 code) from the RDF."""
    graph = _load()
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        for notation in graph.objects(concept, SKOS.notation):
            if str(getattr(notation, "datatype", "")) == ISO3166_DATATYPE:
                yield concept, str(notation).upper()
                break


@lru_cache(maxsize=1)
def _country_index():
    by_label, by_code = {}, {}
    for concept, code in _country_concepts():
        by_code[code] = concept
        for label in _labels(concept):
            for variant in (label, _strip_parenthetical(label)):
                if variant:
                    by_label.setdefault(_key(variant), code)
        uri_tail = str(concept).rstrip("/").rsplit("/", 1)[-1]
        by_label.setdefault(_key(uri_tail), code)
    for alias, code in _COUNTRY_ALIASES.items():
        if code in by_code:
            by_label[_key(alias)] = code
    return by_label, by_code


def country_code_for_label(label: str) -> str | None:
    """Map an ISO alpha-3 code or an English/RDF/colloquial label to alpha-3."""
    if not label or not _clean(label):
        return None
    value = _clean(label)
    by_label, by_code = _country_index()
    upper = value.upper()
    if len(upper) == 3 and upper in by_code:
        return upper
    return by_label.get(_key(value))


def country_label_for_code(code: str) -> str | None:
    """Return the canonical English preferred label from the RDF.

    Colloquial aliases are accepted for lookup only; they are never used as
    output labels. A trailing parenthetical is removed from the preferred
    label for a readable canonical name.
    """
    if not code:
        return None
    _, by_code = _country_index()
    concept = by_code.get(_clean(code).upper())
    if concept is None:
        return None
    preferred = list(_labels(concept, (SKOS.prefLabel,)))
    if preferred:
        return _strip_parenthetical(preferred[0]) or preferred[0]
    # Fallback only if the RDF concept has no English preferred label.
    return next(iter(_labels(concept, (SKOS.altLabel, SKOS.hiddenLabel))), None)


def _expand(collection_uri) -> set[str]:
    """Recursively collect ISO alpha-3 codes under a SKOS collection."""
    graph = _load()
    codes, stack, seen = set(), [collection_uri], set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for member in graph.objects(node, SKOS.member):
            notations = [str(n).upper() for n in graph.objects(member, SKOS.notation)
                         if str(getattr(n, "datatype", "")) == ISO3166_DATATYPE]
            if notations:
                codes.add(notations[0])
            elif (member, RDF.type, SKOS.Collection) in graph:
                stack.append(member)
    return codes


@lru_cache(maxsize=1)
def _collections():
    """Return labelled SKOS collections; region names are sourced from RDF."""
    graph = _load()
    return [(collection, list(_labels(collection)))
            for collection in graph.subjects(RDF.type, SKOS.Collection)
            if list(_labels(collection))]


def country_codes_for_region(label: str) -> set[str]:
    """Resolve any labelled RDF region/collection to its member country codes."""
    needle = _key(label or "")
    if not needle:
        return set()
    collections = _collections()
    for collection, labels in collections:
        if any(_key(name) == needle for name in labels):
            return _expand(collection)
    candidates = [(len(_key(name)), collection) for collection, labels in collections
                  for name in labels if _key(name).startswith(needle)]
    if candidates:
        return _expand(max(candidates, key=lambda item: item[0])[1])
    candidates = [(len(_key(name)), collection) for collection, labels in collections
                  for name in labels
                  if needle in _key(name) and _key(name) not in
                  {"ipbes region", "ipbes assessment regions"}]
    return _expand(max(candidates, key=lambda item: item[0])[1]) if candidates else set()


def _region_mentions(question: str):
    q = _clean(question).casefold()
    matches = []
    for collection, labels in _collections():
        for label in labels:
            variants = { _clean(label).casefold(), _strip_parenthetical(label).casefold() }
            found = [v for v in variants if len(v) >= 3 and
                     re.search(rf"(?<!\w){re.escape(v)}(?!\w)", q)]
            if found:
                matches.append((max(map(len, found)), collection))
                break
    matches.sort(key=lambda item: item[0], reverse=True)
    return [collection for _, collection in matches]


def infer_country_names(question: str) -> set[str]:
    """Infer mentioned countries and countries covered by RDF-defined regions."""
    if not question or not question.strip():
        return set()
    codes = set()
    for collection in _region_mentions(question):
        codes.update(_expand(collection))
    by_label, _ = _country_index()
    for label, code in sorted(by_label.items(), key=lambda item: len(item[0]), reverse=True):
        if len(label) >= 3 and re.search(rf"(?<!\w){re.escape(label)}(?!\w)",
                                         question, re.IGNORECASE):
            codes.add(code)
    return {name for code in codes if (name := country_label_for_code(code))}


def canonical_country(value: str | None) -> str | None:
    """Normalize country name/code; preserve trimmed unknown input."""
    if value is None:
        return None
    value = _clean(value)
    if not value:
        return None
    code = country_code_for_label(value)
    return country_label_for_code(code) if code else value
