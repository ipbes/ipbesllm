"""Compare Akoma Ntoso reports converted from Word (data/xml_compare) with
the reference files (data/xml), to check and improve docx_to_akn.py.

    PYTHONPATH=src python src/xml_compare.py                 # every file in both folders
    PYTHONPATH=src python src/xml_compare.py 2017-5 --details

For each report it prints:

  text        share of the reference's words found in the conversion, in
              order (and the reverse), footnotes left out; the passages of
              the reference that are missing, with --details
  paragraphs  numbered paragraphs of the main report (outside the annexes):
              same numbers, same section, same text
  decisions   the decisions (IPBES-5/1 ...) found as attachments
  structure   counts of divisions, paragraphs, subparagraphs, attachments,
              tables and footnotes
  eIds        share of the reference's element identifiers (eId) found
              in the conversion: the same element in the same place
  loader      chunks made by xml_loader.py from each file
  schema      validity against the Akoma Ntoso 3.0 schema (docs/akn/akomantoso30.xsd,
              from https://docs.oasis-open.org/legaldocml/akn-core/v1.0/os/part2-specs/schemas/)

Reports without a reference (sessions not yet in data/xml) are checked on
their own: paragraph numbering without gaps, decisions found, schema.

The reference files were marked up by hand in Indigo and are not perfect
either (some decisions miss their numbers, some lists are tables), so the
differences are for review, not all errors in the conversion.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
N = {"a": NS}
# A heading naming a decision: "IPBES-5/1: ...", "Decision IPBES-7/1: ...",
# or "Annex III - Decision IPBES/1/2" (IPBES-1); not "Annex I to decision ...".
def _decision_pattern(config: Path = ROOT / "config" / "akn4un_ipbes.toml") -> re.Pattern:
    with open(config, "rb") as f:
        body = re.escape(tomllib.load(f)["document"]["body"])
    return re.compile(rf"(?:^|^Annex [IVX]+\s*[-–]\s*)(?:Decision\s+)?({body}[-‑/]\d+/\d+)",
                      re.IGNORECASE)


_DECISION = _decision_pattern()


def local(el) -> str:
    return etree.QName(el).localname


def text_of(el, notes: bool = False) -> str:
    """Visible text of an element, without footnotes unless asked."""
    parts = []

    def walk(node):
        if not notes and local(node) == "authorialNote":
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
    walk(el)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def words(text: str) -> list[str]:
    text = text.replace("‑", "-").casefold()
    return re.findall(r"[^\W_]+", text)


def in_attachment(el) -> bool:
    return any(local(a) == "attachment" for a in el.iterancestors())


def body_paragraphs(root) -> list[tuple[str, str, str]]:
    """(division heading, number, text) of the numbered paragraphs of the
    main report."""
    out = []
    for para in root.iter(f"{{{NS}}}paragraph"):
        if in_attachment(para):
            continue
        num = para.find("a:num", N)
        division = next((a for a in para.iterancestors() if local(a) == "division"), None)
        heading = division.find("a:heading", N) if division is not None else None
        out.append((text_of(heading) if heading is not None else "",
                    (num.text or "").strip(" .") if num is not None else "",
                    text_of(para)))
    return out


def decisions(root) -> list[str]:
    """Decision symbols heading an attachment, in order."""
    found = []
    for att in root.iter(f"{{{NS}}}attachment"):
        heading = att.find("a:heading", N)
        m = _DECISION.search(text_of(heading)) if heading is not None else None
        symbol = re.sub(r"^(\w+)[‑/]", r"\1-", m.group(1)) if m else ""
        if symbol and symbol not in found:
            found.append(symbol)
    return found


def structure(root) -> Counter:
    tags = ("division", "subdivision", "paragraph", "subparagraph", "attachment", "table",
            "authorialNote")
    return Counter({t: sum(1 for _ in root.iter(f"{{{NS}}}{t}")) for t in tags})


def body_text(root) -> str:
    """All text after the metadata and preface."""
    report = root[0]
    return " ".join(text_of(child) for child in report if local(child) not in ("meta", "preface"))


def note_texts(root) -> list[str]:
    return [text_of(n, notes=True) for n in root.iter(f"{{{NS}}}authorialNote")]


def coverage(ref: list[str], new: list[str]) -> tuple[float, list[str]]:
    """Share of `ref` words matched in order in `new`, and the missing
    passages (runs of 8 words or more)."""
    matcher = difflib.SequenceMatcher(None, ref, new, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    missing = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("delete", "replace") and i2 - i1 >= 8:
            missing.append(" ".join(ref[i1:i2]))
    return matched / max(len(ref), 1), missing


def loader_chunks(path: Path) -> Counter:
    sys.path.insert(0, str(ROOT / "src"))
    from xml_loader import parse_akn_file
    return Counter(c["chunk_type"] for c in parse_akn_file(str(path)))


SCHEMA = ROOT / "docs" / "akn" / "akomantoso30.xsd"
_schema = None


def schema_check(path: Path) -> str:
    global _schema
    if not SCHEMA.exists():
        return "schema not found"
    if _schema is None:
        _schema = etree.XMLSchema(etree.parse(str(SCHEMA)))
    if _schema.validate(etree.parse(str(path))):
        return "valid"
    errors = [e for e in _schema.error_log]
    return f"{len(errors)} errors, first: line {errors[0].line}: {errors[0].message[:200]}"


def check_alone(name: str, new_dir: Path) -> dict:
    """A conversion without a reference file."""
    path = new_dir / f"{name}.xml"
    root = etree.parse(str(path)).getroot()
    print(f"\n=== {name} (no reference) ===")
    nums = [int(m.group()) for _, n, _ in body_paragraphs(root) if (m := re.match(r"\d+", n))]
    breaks = [f"{a}->{b}" for a, b in zip(nums, nums[1:]) if b != a + 1]
    print(f"paragraphs  main report: {len(nums)} numbered"
          + (f", {nums[0]}-{nums[-1]}" if nums else "")
          + (f"; breaks in the sequence: {', '.join(breaks[:10])}" if breaks else "; no gaps"))
    found = decisions(root)
    print(f"decisions   {len(found)} {' '.join(found)}")
    s = structure(root)
    print("structure   " + ", ".join(f"{k} {s[k]}" for k in s))
    print(f"schema      {schema_check(path)}")
    return {"name": name, "paragraphs": len(nums), "breaks": len(breaks), "decisions": len(found)}


def compare(name: str, ref_dir: Path, new_dir: Path, details: bool) -> dict:
    ref_path, new_path = ref_dir / f"{name}.xml", new_dir / f"{name}.xml"
    ref, new = etree.parse(str(ref_path)).getroot(), etree.parse(str(new_path)).getroot()
    print(f"\n=== {name} ===")

    ref_words, new_words = words(body_text(ref)), words(body_text(new))
    found, missing = coverage(ref_words, new_words)
    extra, added = coverage(new_words, ref_words)
    print(f"text        {found:.1%} of the reference's {len(ref_words)} words found; "
          f"{extra:.1%} of the conversion's {len(new_words)} words in the reference")
    if details:
        for passage in missing[:15]:
            print(f"   missing: {passage[:160]}")
        for passage in added[:10]:
            print(f"   added:   {passage[:160]}")

    rp, np_ = body_paragraphs(ref), body_paragraphs(new)
    r_nums = [n for _, n, _ in rp if n]
    n_nums = [n for _, n, _ in np_ if n]
    by_num_ref = {n: (d, t) for d, n, t in rp if n}
    by_num_new = {n: (d, t) for d, n, t in np_ if n}
    same_text = sum(1 for n in by_num_ref if n in by_num_new and
                    difflib.SequenceMatcher(None, words(by_num_ref[n][1]), words(by_num_new[n][1]),
                                            autojunk=False).ratio() > 0.9)
    same_section = sum(1 for n in by_num_ref if n in by_num_new and
                       words(by_num_ref[n][0]) == words(by_num_new[n][0]))
    print(f"paragraphs  main report: reference {len(r_nums)} numbered, conversion {len(n_nums)}; "
          f"same text {same_text}, same section {same_section}")
    if details:
        only_ref = [n for n in r_nums if n not in by_num_new]
        only_new = [n for n in n_nums if n not in by_num_ref]
        if only_ref:
            print(f"   only in reference: {', '.join(only_ref[:30])}")
        if only_new:
            print(f"   only in conversion: {', '.join(only_new[:30])}")
        shown = 0
        for n in by_num_ref:
            if n in by_num_new and shown < 5:
                ratio = difflib.SequenceMatcher(None, words(by_num_ref[n][1]),
                                                words(by_num_new[n][1]), autojunk=False).ratio()
                if ratio <= 0.9:
                    shown += 1
                    print(f"   para {n} differs ({ratio:.0%}):\n      ref: {by_num_ref[n][1][:150]}"
                          f"\n      new: {by_num_new[n][1][:150]}")

    rd, nd = decisions(ref), decisions(new)
    print(f"decisions   reference {len(rd)} {' '.join(rd)}\n            conversion {len(nd)} {' '.join(nd)}")

    rs, ns = structure(ref), structure(new)
    print("structure   " + ", ".join(f"{k} {rs[k]}/{ns[k]}" for k in rs) + "   (reference/conversion)")

    r_ids = {e.get("eId") for e in ref.iter() if e.get("eId")}
    n_ids = {e.get("eId") for e in new.iter() if e.get("eId")}
    print(f"eIds        {len(r_ids & n_ids)} of the reference's {len(r_ids)} element identifiers "
          f"also in the conversion ({len(r_ids & n_ids) / max(len(r_ids), 1):.0%})")
    if details:
        only = sorted(r_ids - n_ids, key=lambda x: (len(x), x))
        print(f"   only in reference (first): {', '.join(only[:8])}")

    r_notes, n_notes = note_texts(ref), note_texts(new)
    note_found, _ = coverage(words(" ".join(r_notes)), words(" ".join(n_notes)))
    print(f"footnotes   {len(r_notes)}/{len(n_notes)}, {note_found:.1%} of the reference's footnote words found")

    try:
        rc, nc = loader_chunks(ref_path), loader_chunks(new_path)
        print("loader      " + ", ".join(f"{k} {rc[k]}/{nc[k]}" for k in sorted(set(rc) | set(nc))))
    except Exception as e:  # noqa: BLE001
        print(f"loader      failed: {e}")
    print(f"schema      {schema_check(new_path)}")
    return {"name": name, "text": found, "reverse": extra, "eids": len(r_ids & n_ids) / max(len(r_ids), 1), "paragraphs": (len(r_nums), len(n_nums), same_text),
            "decisions": (len(rd), len(nd), len(set(rd) & set(nd)))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="report names, e.g. 2017-5 (default: all converted)")
    ap.add_argument("--ref", type=Path, default=ROOT / "data" / "xml")
    ap.add_argument("--new", type=Path, default=ROOT / "data" / "xml_compare")
    ap.add_argument("--details", action="store_true", help="list differences")
    args = ap.parse_args(argv)
    names = args.names or sorted((p.stem for p in args.new.glob("*.xml")),
                                 key=lambda n: [int(x) for x in re.findall(r"\d+", n)])
    results, alone = [], []
    for n in names:
        if (args.ref / f"{n}.xml").exists():
            results.append(compare(n, args.ref, args.new, args.details))
        else:
            alone.append(check_alone(n, args.new))
    if len(results) + len(alone) > 1:
        print("\n=== summary ===")
        for r in results:
            p = r["paragraphs"]
            d = r["decisions"]
            print(f"{r['name']:10} text {r['text']:.1%} / {r['reverse']:.1%}   paragraphs {p[0]}/{p[1]} "
                  f"(same text {p[2]})   decisions {d[0]}/{d[1]} (both {d[2]})   eIds {r['eids']:.0%}")
        for r in alone:
            print(f"{r['name']:10} no reference   paragraphs {r['paragraphs']} "
                  f"({r['breaks']} breaks)   decisions {r['decisions']}")


if __name__ == "__main__":
    main()
