"""Check the TTL files against the IPBES report ontology. Read-only.

    PYTHONPATH=src python src/ttl_audit.py                        # data/ttl/*.ttl
    PYTHONPATH=src python src/ttl_audit.py data/ttl/SUA_v02.ttl   # some files
    PYTHONPATH=src python src/ttl_audit.py --all                  # every property

For each class of the ontology (docs/IPBES_Ontology_v06.ttl) and each file it
reports:

  * the number of items, and the share of them that have each property the
    ontology lists, for properties below 100%. Properties that are partial by
    nature (confidence levels, expert roles, DOIs, links to illustrations ...)
    are only shown with --all;
  * properties the file uses that the ontology does not list;
  * items the indexer leaves out, found by running ttl_loader on the file
    (e.g. experts without a name, messages without text);
  * data quality findings (DATA QUALITY), each with a count and examples:
    garbled encoding ("Earthâ€™s"), invisible characters, identifiers padded
    with spaces, empty values, stray characters at the end of labels;
    knowledge gaps numbered from 0 or without a category; placeholder
    experts, experts without a name label or a role, role text not in the
    form "Lead Author in GA119-ch2"; country values that are not standard
    (misspelt, colloquial, several in one value) or unknown; references
    without a DOI, malformed Zotero links and placeholder ORCID links.
    The loader works around several of these, but they should be fixed in
    the TTL conversion.

ttl_index.py --dry-run also stops on unknown country values.
Run it on new or updated TTL files before indexing them.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import glob
import io
import re
from pathlib import Path

import rdflib
from rdflib import RDF

import geo
from settings import PROJECT_ROOT, TTL_DIR
from ttl_index import _resolve_countries
from ttl_loader import GARBLED, parse_ttl_file

ONTOLOGY = PROJECT_ROOT / "docs" / "IPBES_Ontology_v06.ttl"

_PREFIXES = [
    ("http://ontology.ipbes.net/report", "ipbes:"),
    ("http://purl.org/dc/terms/", "dcterms:"),
    ("http://www.w3.org/2004/02/skos/core#", "skos:"),
    ("http://xmlns.com/foaf/0.1/", "foaf:"),
    ("http://www.w3.org/2002/07/owl#", "owl:"),
]

# Properties an item normally has only some of: one confidence level per
# message, some roles per expert, optional links and identifiers.
PARTIAL_BY_NATURE = {
    "ipbes:hasWellestablished", "ipbes:hasEstablishedIncomplete",
    "ipbes:hasUnresolved", "ipbes:hasInconclusive",
    "ipbes:ca", "ipbes:cl", "ipbes:cs", "ipbes:fl", "ipbes:la", "ipbes:re",
    "ipbes:Illustration", "ipbes:hasDoi", "owl:sameAs",
    "ipbes:KeyMessage", "ipbes:Reference", "foaf:Person",
}

# Ontology class -> the indexer's chunk type for it.
CHUNK_TYPES = {
    "ipbes:KeyMessage": "key", "ipbes:BackgroundMessage": "bgm",
    "ipbes:SubMessage": "subm", "ipbes:KnowledgeGap": "kg",
    "ipbes:Chapter": "chapter", "ipbes:SubChapter": "sch",
    "ipbes:Illustration": "il", "foaf:Person": "person",
    "ipbes:Reference": "ref",
}


def short(uri) -> str:
    """A URI in prefix form; the ipbes: namespace has no trailing / or #."""
    text = str(uri)
    for namespace, prefix in _PREFIXES:
        if text.startswith(namespace):
            return prefix + text[len(namespace):].lstrip("/#")
    return text


def expected_properties(ontology: Path) -> dict[str, set[str]]:
    """Class -> the properties the ontology lists for it."""
    graph = rdflib.Graph()
    graph.parse(ontology, format="turtle")
    return {
        short(cls): {short(p) for p, _ in graph.predicate_objects(example)
                     if p != RDF.type}
        for example, cls in graph.subject_objects(RDF.type)
    }


def _local_name(uri) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


ROLE_PROPERTIES = ["ipbes:cs", "ipbes:cl", "ipbes:la", "ipbes:re", "ipbes:fl", "ipbes:ca"]
# The usual form of a role value; co-chairs need no chapter.
STANDARD_ROLE = re.compile(
    r"^(?:Coordinating Lead Author|Lead Author|Review Editor|Fellow|"
    r"Contributing Author) in [A-Z]+\d+-ch\d+(?:\.\d+)?$"
    r"|^Co-chair in [A-Z]+\d+(?:-ch\d+)?$"
)
INVISIBLE = re.compile("[\ufeff\u200b\u200c\u200d\u2060]")
PLACEHOLDER_NAME = re.compile(r"first ?name|last ?name|^[_\-\s]*$", re.IGNORECASE)
# A footnote letter run into a label: "... and pollutantsC".
STRAY_LETTER = re.compile(r"[a-z]{3,}[A-Z]$")


def _examples(values, n: int = 3) -> str:
    values = list(values)
    shown = ", ".join(repr(v) for v in values[:n])
    return shown + (" …" if len(values) > n else "")


@functools.lru_cache(maxsize=1)
def _rdf_country_labels() -> frozenset[str]:
    """Country labels from the geography RDF itself, without the aliases
    geo.py adds for colloquial names and misspellings."""
    labels = set()
    for concept, _ in geo._country_concepts():
        for label in geo._labels(concept):
            labels |= {geo._key(label), geo._key(geo._strip_parenthetical(label))}
    return frozenset(labels)


def _country_kind(value: str) -> str:
    """standard (an RDF label or ISO code); combined (several countries in
    one value, which should be separate ipbes:country values); non-standard
    (one country, but misspelt, colloquial or a local name); or unknown."""
    _, by_code = geo._country_index()
    if geo._key(value) in _rdf_country_labels() or value.strip().upper() in by_code:
        return "standard"
    codes, _, missed = _resolve_countries(value)
    if not codes or missed:
        return "unknown"
    return "combined" if len(codes) > 1 or len(re.split(r"[;/,]", value)) > 1 \
        else "non-standard"


def quality_checks(graph: rdflib.Graph) -> list[tuple[str, int, str]]:
    """Data quality findings in one file: (finding, count, examples)."""
    props: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    types: dict = collections.defaultdict(list)
    for s, p, o in graph:
        if p == RDF.type:
            types[short(o)].append(s)
        else:
            props[s][short(p)].append(o)
    literals = [(p, str(o)) for s in props for p, objs in props[s].items()
                for o in objs if isinstance(o, rdflib.Literal)]
    found = []

    def add(finding: str, values) -> None:
        values = list(values)
        if values:
            found.append((finding, len(values), _examples(values)))

    # Text
    add("garbled encoding (repaired when indexed)",
        [v for _, v in literals if GARBLED.search(v)])
    add("invisible characters (byte order mark, zero-width space)",
        [v for _, v in literals if INVISIBLE.search(v)])
    add("identifiers padded with spaces or tabs",
        [v for p, v in literals if p == "dcterms:identifier" and v != v.strip()])
    empty = collections.Counter(p for p, v in literals if not v.strip())
    for prop, n in sorted(empty.items()):
        found.append((f"empty values of {prop}", n, ""))
    add("labels ending in a stray capital letter (footnote mark?)",
        [v for p, v in literals if p == "skos:prefLabel" and STRAY_LETTER.search(v.strip())])

    # Knowledge gaps
    gaps = types.get("ipbes:KnowledgeGap", [])
    add("knowledge gaps numbered from 0 (IBOK numbers from 1)",
        [_local_name(g) for g in gaps if _local_name(g).endswith("-0")])
    if gaps:
        add("knowledge gaps without a category (ipbes:hasCategory)",
            [_local_name(g) for g in gaps if not props[g].get("ipbes:hasCategory")])

    # Experts
    people = types.get("foaf:Person", [])

    def name(person) -> str:
        label = " ".join(str(o) for o in props[person].get("skos:prefLabel", [])).strip()
        return label or " ".join(
            str(o) for p in ("foaf:firstName", "foaf:lastName")
            for o in props[person].get(p, [])).strip()

    add("placeholder or empty expert entries",
        [_local_name(p) for p in people
         if PLACEHOLDER_NAME.search(name(p)) or _local_name(p).strip("_") == ""])
    add("experts without skos:prefLabel (name only in first/last name)",
        [name(p) for p in people
         if not str(" ".join(map(str, props[p].get("skos:prefLabel", [])))).strip()
         and name(p)])
    def missing_surname(person) -> bool:
        label = " ".join(str(o) for o in props[person].get("skos:prefLabel", [])).strip()
        last = " ".join(str(o) for o in props[person].get("foaf:lastName", [])).strip()
        return bool(label and last) and last.casefold() not in label.casefold()

    add("expert labels without the last name (foaf:lastName)",
        [f"{name(p)} (last name: {' '.join(map(str, props[p].get('foaf:lastName', [])))})"
         for p in people if missing_surname(p)])
    add("experts without a role",
        [name(p) or _local_name(p) for p in people
         if not any(str(o).strip() for r in ROLE_PROPERTIES for o in props[p].get(r, []))])
    add("role values not in the form \"Lead Author in GA119-ch2\"",
        sorted({str(o).strip() for p in people for r in ROLE_PROPERTIES
                for o in props[p].get(r, [])
                if str(o).strip() and not STANDARD_ROLE.match(str(o).strip())}))
    countries = [str(o) for p in people for o in props[p].get("ipbes:country", [])
                 if str(o).strip()]
    kinds = {v: _country_kind(v) for v in set(countries)}
    add("several countries in one country value (should be separate values)",
        sorted(v for v, k in kinds.items() if k == "combined"))
    add("country names not in standard form (misspelt, colloquial or local name)",
        sorted(v for v, k in kinds.items() if k == "non-standard"))
    add("country values that cannot be matched to a country",
        sorted(v for v, k in kinds.items() if k == "unknown"))
    add("placeholder ORCID links (owl:sameAs ending in /NA)",
        [_local_name(p) for p in people
         if any(str(o).rstrip("/").endswith("/NA") for o in props[p].get("owl:sameAs", []))])

    # References
    references = types.get("ipbes:Reference", [])
    if references:
        add("references without a DOI",
            [_local_name(r) for r in references
             if not any(str(o).strip() for o in props[r].get("ipbes:hasDoi", []))])
        add("Zotero links with a double slash (\"…//items/…\")",
            [str(o) for r in references for o in props[r].get("owl:sameAs", [])
             if "//items" in str(o)])
    return found


def audit_file(path: Path, expected: dict[str, set[str]], show_all: bool) -> list[str]:
    graph = rdflib.Graph()
    graph.parse(path, format="turtle")
    loaded = collections.defaultdict(set)
    with contextlib.redirect_stdout(io.StringIO()):     # the loader's notes
        chunks = parse_ttl_file(str(path))
    for chunk in chunks:
        loaded[chunk["chunk_type"]].add(chunk["eId"])

    lines = [f"\n### {path.name}"]
    findings = quality_checks(graph)
    if findings:
        lines.append("  DATA QUALITY")
        for finding, count, examples in findings:
            lines.append(f"    {finding}: {count}" + (f"  e.g. {examples}" if examples else ""))
    for cls, properties in expected.items():
        items = [s for s, c in graph.subject_objects(RDF.type) if short(c) == cls]
        if not items:
            lines.append(f"  {cls}: none")
            continue
        have = collections.Counter(
            short(p) for item in items
            for p in {p for p, _ in graph.predicate_objects(item)} if p != RDF.type
        )
        partial = [
            f"{p} {100 * have[p] // len(items)}%"
            for p in sorted(properties)
            if have[p] < len(items) and (show_all or p not in PARTIAL_BY_NATURE)
        ]
        extra = sorted(p for p in have if p not in properties)
        left_out = sorted(
            _local_name(item) for item in items
            if _local_name(item) not in loaded[CHUNK_TYPES.get(cls, "")]
        ) if cls in CHUNK_TYPES else []

        lines.append(f"  {cls}: {len(items)}")
        if partial:
            lines.append(f"    below 100%:       {', '.join(partial)}")
        if extra:
            lines.append(f"    not in ontology:  {', '.join(extra)}")
        if left_out:
            examples = ", ".join(left_out[:5]) + (" …" if len(left_out) > 5 else "")
            lines.append(f"    not indexed:      {len(left_out)} ({examples})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help="TTL files (default: data/ttl/*.ttl)")
    parser.add_argument("--all", action="store_true",
                        help="also show properties that are partial by nature")
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY)
    args = parser.parse_args()

    files = [Path(f) for f in args.files] or [
        Path(f) for f in sorted(glob.glob(str(TTL_DIR / "*.ttl")))
    ]
    expected = expected_properties(args.ontology)
    print(f"Ontology: {args.ontology}")
    for path in files:
        print("\n".join(audit_file(path, expected, args.all)))


if __name__ == "__main__":
    main()
