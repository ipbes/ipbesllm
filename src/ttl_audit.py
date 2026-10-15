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
  * text with garbled encoding ("Earthâ€™s"), which the loader repairs but
    which should be fixed in the conversion.

Country values are checked separately: ttl_index.py --dry-run.
Run it on new or updated TTL files before indexing them.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import io
from pathlib import Path

import rdflib
from rdflib import RDF

from settings import PROJECT_ROOT, TTL_DIR
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


def audit_file(path: Path, expected: dict[str, set[str]], show_all: bool) -> list[str]:
    graph = rdflib.Graph()
    graph.parse(path, format="turtle")
    loaded = collections.defaultdict(set)
    with contextlib.redirect_stdout(io.StringIO()):     # the loader's notes
        chunks = parse_ttl_file(str(path))
    for chunk in chunks:
        loaded[chunk["chunk_type"]].add(chunk["eId"])

    lines = [f"\n### {path.name}"]
    garbled = sum(1 for o in graph.objects() if isinstance(o, rdflib.Literal)
                  and GARBLED.search(str(o)))
    if garbled:
        lines.append(f"  garbled text: {garbled} values (repaired when indexed)")
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
