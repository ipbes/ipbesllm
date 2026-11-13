"""Compare the structure of translated Plenary reports with the English ones.

    PYTHONPATH=src python src/xml_compare_lang.py                  # every translation in data/xml/<lang>/
    PYTHONPATH=src python src/xml_compare_lang.py fr 2017-5        # where one translation differs
    PYTHONPATH=src python src/xml_compare_lang.py --csv out.csv    # the table as CSV too

docx_to_akn.py gives a translation the hierarchy and element identifiers
(eId) of the English report, so the two can be compared element by element.
For each report and language it prints:

  structure   share of the English structural elements (divisions,
              subdivisions, paragraphs, subparagraphs, attachments, tables)
              found in the translation at the same place in the tree, in order
  eIds        share of the English structural identifiers also in the
              translation
  attachments whether the decisions, annexes and appendices form the same tree

With a language and a report, it lists where the two differ, with the
number and the first words of each element, for review.

Differences are not all conversion errors: a translation sometimes numbers
a section differently from the English, types a heading as a bold
paragraph, or styles a note as a heading.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
STRUCTURE = ("division", "subdivision", "paragraph", "subparagraph", "attachment", "table")
_TAGS = {f"{{{NS}}}{t}" for t in STRUCTURE}


def _elements(path: Path) -> list[tuple[str, str, str, str]]:
    """(path of tags, eId, number, first words) of the structural elements."""
    root = etree.parse(str(path)).getroot()
    out = []

    def walk(el, tags):
        for child in el:
            if child.tag in _TAGS:
                tag = child.tag.split("}")[1]
                num = child.find(f"{{{NS}}}num")
                head = child.find(f"{{{NS}}}heading")
                if head is None:
                    head = next(child.iter(f"{{{NS}}}p"), None)
                words = " ".join("".join(head.itertext()).split())[:60] if head is not None else ""
                out.append(("/".join(tags + [tag]), child.get("eId", ""),
                            num.text if num is not None else "", words))
                walk(child, tags + [tag])
            else:
                walk(child, tags)
    walk(root, [])
    return out


def _matched(a: list, b: list) -> int:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks())


def compare(english: Path, translation: Path) -> dict:
    en, tr = _elements(english), _elements(translation)
    en_ids = {e[1] for e in en}
    en_att = [e[1] for e in en if e[0].endswith("attachment")]
    tr_att = [e[1] for e in tr if e[0].endswith("attachment")]
    return {
        "structure": _matched([e[0] for e in en], [e[0] for e in tr]) / max(len(en), 1),
        "eids": len(en_ids & {e[1] for e in tr}) / max(len(en_ids), 1),
        "attachments": en_att == tr_att,
        "elements": (len(en), len(tr)),
    }


def differences(english: Path, translation: Path, limit: int = 20):
    """The places where the translation's tree differs from the English."""
    en, tr = _elements(english), _elements(translation)
    sm = difflib.SequenceMatcher(None, [e[1] for e in en], [e[1] for e in tr], autojunk=False)
    shown = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        shown += 1
        if shown > limit:
            print(f"... (first {limit} differences shown)")
            break
        print(f"--- {op}: English {i1 + 1}-{i2}, translation {j1 + 1}-{j2}")
        for side, items in (("EN", en[i1:i2]), ("TR", tr[j1:j2])):
            for e in items[:4]:
                print(f"  {side} {e[1]:<50} {e[2]:<8} {e[3]}")
            if len(items) > 4:
                print(f"  {side} ... {len(items)} elements")


def _session_key(name: str):
    year, _, session = name.partition("-")
    return (int(year), int(session)) if year.isdigit() and session.isdigit() else (0, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("language", nargs="?", help="a language folder (fr), with a report")
    ap.add_argument("report", nargs="?", help="a report (2017-5): list its differences")
    ap.add_argument("--en", type=Path, default=ROOT / "data" / "xml", help="English reports")
    ap.add_argument("--tr", type=Path, default=ROOT / "data" / "xml",
                    help="folder with a subfolder per language (default: data/xml)")
    ap.add_argument("--csv", type=Path, help="also write the table to this CSV file")
    args = ap.parse_args(argv)

    if args.language and args.report:
        differences(args.en / f"{args.report}.xml", args.tr / args.language / f"{args.report}.xml")
        return 0
    langs = sorted(p.name for p in args.tr.iterdir()
                   if p.is_dir() and (not args.language or p.name == args.language))
    rows = []
    for lang in langs:
        for f in sorted((args.tr / lang).glob("*.xml"), key=lambda p: _session_key(p.stem)):
            english = args.en / f.name
            if english.exists():
                rows.append((f.stem, lang, compare(english, f)))
    if not rows:
        print(f"No translations found in {args.tr}/<language>/")
        return 1
    sessions = sorted({r[0] for r in rows}, key=_session_key)
    table = {(s, l): c for s, l, c in rows}
    print(f"{'report':8} " + " ".join(f"{l:>20}" for l in langs))
    for s in sessions:
        cells = []
        for l in langs:
            c = table.get((s, l))
            cells.append(f"{c['structure'] * 100:5.1f}% {c['eids'] * 100:5.1f}% {'=' if c['attachments'] else '≠'}"
                         if c else "-")
        print(f"{s:8} " + " ".join(f"{c:>20}" for c in cells))
    print("\nper language: structure found in order, structural eIds shared, attachments (= same tree)")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["report", "language", "structure", "eids", "same_attachments",
                        "english_elements", "translation_elements"])
            for s, l, c in rows:
                w.writerow([s, l, f"{c['structure']:.3f}", f"{c['eids']:.3f}", c["attachments"], *c["elements"]])
        print(f"written: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
