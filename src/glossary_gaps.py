"""Glossary terms of the assessments (Annex I) that the IPBES thesaurus does
not have: candidates to add to the thesaurus, after checking.

    PYTHONPATH=src python src/glossary_gaps.py                # every assessment in config/pdf_to_ttl.toml
    PYTHONPATH=src python src/glossary_gaps.py NXS SUA        # some

Each glossary is read from the PDF as pdf_to_ttl.py reads it, and each term is
looked up in data/rdf/ipbes-thesaurus.rdf the same way (English preferred,
alternative and hidden labels; case, accents, punctuation, spacing, plurals
and -isation/-ization aside; also without brackets, either of "X or Y", either
side of "/").

Writes data/csv/glossary_not_in_thesaurus.csv, one row per term:
the term, the assessments whose glossary has it, the definition and page in
each, and the closest thesaurus labels, which may be the same concept under
another name. Rerun after the thesaurus is updated: terms added drop out.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

import pymupdf
from rdflib import Graph
from rdflib.namespace import SKOS

from pdf_to_ttl import (DEFAULT_CONFIG, ROOT, _norm, build_paras, glossary_entries, in_thesaurus, parse,
                        read_lines, thesaurus_labels)

THESAURUS = ROOT / "data" / "rdf" / "ipbes-thesaurus.rdf"
OUT = ROOT / "data" / "csv" / "glossary_not_in_thesaurus.csv"


def english_labels(path: Path) -> dict[str, str]:
    """Normalized label -> the label as written, preferred labels first."""
    g = Graph()
    g.parse(path)
    out: dict[str, str] = {}
    for prop in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        for _, lab in g.subject_objects(prop):
            if getattr(lab, "language", None) in (None, "en"):
                out.setdefault(_norm(str(lab)), str(lab))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("assessments", nargs="*", help="codes as in config/pdf_to_ttl.toml (default: all)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cfg = tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))["assessments"]
    codes = args.assessments or list(cfg)

    labels = thesaurus_labels(THESAURUS)
    written = english_labels(THESAURUS)
    rows: dict[str, dict] = {}                    # normalized term -> row
    totals = {}
    for code in codes:
        doc = pymupdf.open(ROOT / cfg[code]["pdf"])
        paras = build_paras(read_lines(doc))
        entries = glossary_entries(paras, parse(doc, paras))
        missing = [e for e in entries if not in_thesaurus(e["term"], labels)]
        totals[code] = (len(entries), len(missing))
        print(f"{code}: {len(entries)} glossary terms, {len(missing)} not in the thesaurus", file=sys.stderr)
        for e in missing:
            row = rows.setdefault(_norm(e["term"]), {"term": e["term"], "in": defaultdict(list)})
            row["in"][code].append(e)

    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["term", "assessments", "definition", "pages", "closest thesaurus labels", "checked", "note"])
        for key, row in sorted(rows.items()):
            close = difflib.get_close_matches(key, list(written), n=3, cutoff=0.75)
            defs = " || ".join(f"[{c}] {e['definition']}" for c, es in row["in"].items() for e in es)
            pages = "; ".join(f"{c} p{e['page']}" for c, es in row["in"].items() for e in es)
            w.writerow([row["term"], ", ".join(row["in"]), defs[:32000], pages,
                        "; ".join(written[c] for c in close), "", ""])
    print(f"-> {args.out}: {len(rows)} terms (thesaurus file of "
          f"{date.fromtimestamp(THESAURUS.stat().st_mtime)})", file=sys.stderr)
    for code, (n, m) in totals.items():
        print(f"  {code:4} {n:4} terms  {m:3} not in the thesaurus", file=sys.stderr)


if __name__ == "__main__":
    main()
