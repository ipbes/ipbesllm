"""Compare two TTL files of the same assessment, item by item: a manual
conversion (data/ttl) with one made from the PDF by pdf_to_ttl.py.

    PYTHONPATH=src python src/ttl_compare.py data/ttl/BBA_v01.ttl data/pdf_compare/BBA/BBA_pdf.ttl \\
        --pdf data/pdf/bba_report.pdf

Items are paired by their identifiers (sub-chapter 1.1, key message KM1,
Zotero key ...), people by name, sub-messages by their text. For each pair
it compares every property: texts (the same, differing only in spaces and
punctuation, or different) and links (sub-chapters, references, people ...).

With --pdf it also measures fidelity: the share of each file's sentences
found word for word in the PDF (footnote numbers aside). This shows text
that was changed in conversion, such as dropped commas or garbled characters.

Writes <out>.md (summary and examples) and <out>.csv (every difference),
by default next to the second file: <name>_compare.md / .csv.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDF, SKOS

from pdf_to_ttl import name_match
from ttl_loader import _discover_ipbes_namespace

CLASSES = ["Report", "Chapter", "SubChapter", "KeyMessage", "BackgroundMessage", "SubMessage",
           "Illustration", "KnowledgeGap", "Person", "Reference"]
TEXT_PROPS = {"hasDescription", "prefLabel", "altLabel", "firstName", "lastName", "hasCategory"}


def local(uri) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def pname(p, ns: str) -> str:
    s = str(p)
    if s.startswith(ns):
        return s[len(ns):]
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def squash(text: str) -> str:
    """Spaces normalized: what a reader would call the same text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def loose(text: str) -> str:
    """Letters and digits only: equal if only spaces, punctuation or case differ."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


class TTL:
    def __init__(self, path: Path):
        self.path = path
        self.g = Graph()
        self.g.parse(path, format="turtle")
        self.ns = _discover_ipbes_namespace(self.g)
        self.items: dict[str, dict[str, URIRef]] = {}
        for cls in CLASSES:
            t = FOAF.Person if cls == "Person" else URIRef(self.ns + cls)
            self.items[cls] = {}
            for s in self.g.subjects(RDF.type, t):
                self.items[cls][self.key(cls, s)] = s

    def key(self, cls: str, s) -> str:
        if cls == "Person":
            return squash(self.text(s, SKOS.prefLabel) or
                          f"{self.text(s, FOAF.firstName)} {self.text(s, FOAF.lastName)}")
        if cls == "Illustration":
            ident = self.text(s, DCTERMS.identifier) or local(s).split("-", 1)[-1]
            return re.sub(r"[\s.]", "", ident).lower()
        if cls == "Chapter":
            return re.sub(r"\D", "", local(s).rsplit("ch", 1)[-1]).lstrip("0")
        if cls == "Reference":
            return local(s)
        if cls == "Report":
            return "report"
        # BBA26-1.1, BBA26-KM1, BBA26-A1-SM0, BBA26-1: drop the report code.
        return re.sub(r"\.$", "", local(s).split("-", 1)[-1].strip()).lower()

    def text(self, s, p) -> str:
        v = self.g.value(s, p)
        return squash(v) if v is not None else ""

    def props(self, s) -> dict[str, list]:
        out = defaultdict(list)
        for p, o in self.g.predicate_objects(s):
            if p == RDF.type:
                continue
            out[pname(p, self.ns)].append(o)
        return out


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def pair(cls: str, a: TTL, b: TTL) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    ka, kb = a.items[cls], b.items[cls]
    if cls == "Person":
        pairs, left = [], dict(kb)
        only_a = []
        for name in ka:
            hit = next((n for n in left if name_match(name, n) == "same"), None) or \
                  next((n for n in left if name_match(name, n) == "close"), None)
            if hit:
                pairs.append((name, hit))
                del left[hit]
            else:
                only_a.append(name)
        return pairs, only_a, list(left)
    if cls == "SubMessage":
        return pair_submessages(a, b)
    both = [k for k in ka if k in kb]
    return [(k, k) for k in both], [k for k in ka if k not in kb], [k for k in kb if k not in ka]


def pair_submessages(a: TTL, b: TTL):
    """Within each background message, pair sub-messages by text."""
    def by_bgm(t: TTL):
        out = defaultdict(list)
        for k, s in t.items["SubMessage"].items():
            out[k.rsplit("-sm", 1)[0]].append(k)
        return out

    ga, gb = by_bgm(a), by_bgm(b)
    pairs, only_a, only_b = [], [], []
    for bgm in sorted(set(ga) | set(gb)):
        left = list(gb.get(bgm, []))
        for ka in sorted(ga.get(bgm, []), key=lambda k: int(re.sub(r"\D", "", k.rsplit("sm", 1)[-1]) or 0)):
            ta = loose(a.text(a.items["SubMessage"][ka], URIRef(a.ns + "hasDescription")))
            best, score = None, 0.0
            for kb in left:
                tb = loose(b.text(b.items["SubMessage"][kb], URIRef(b.ns + "hasDescription")))
                r = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio() if ta and tb else 0
                if r > score:
                    best, score = kb, r
            if best and score >= 0.6:
                pairs.append((ka, best))
                left.remove(best)
            else:
                only_a.append(ka)
        only_b += left
    return pairs, only_a, only_b


# ---------------------------------------------------------------------------
# Comparing properties
# ---------------------------------------------------------------------------

def text_kind(x: str, y: str) -> str:
    if x == y:
        return "same"
    if loose(x) == loose(y):
        return "spaces/punctuation"
    return "different"


def first_difference(x: str, y: str, width: int = 60) -> str:
    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            a0, b0 = max(0, i1 - 25), max(0, j1 - 25)
            return f"…{x[a0:i2 + 25][:width + 50]}… | …{y[b0:j2 + 25][:width + 50]}…"
    return ""


def compare(a: TTL, b: TTL, names=("manual", "PDF")):
    rows = []          # (class, item a, item b, property, kind, value a, value b)
    summary = []
    # A reverse index for links, built once.
    index = {}
    for t in (a, b):
        idx = {}
        for cls, items in t.items.items():
            for k, u in items.items():
                idx[u] = f"{cls}:{k}"
        index[id(t)] = idx

    def vkey(o, t):
        if isinstance(o, URIRef):
            return index[id(t)].get(o) or str(o).replace("http://zotero.org", "https://www.zotero.org")
        return squash(o)

    for cls in CLASSES:
        pairs, only_a, only_b = pair(cls, a, b)
        prop_stats = defaultdict(lambda: defaultdict(int))
        for k in only_a:
            rows.append((cls, k, "", "", f"only in {names[0]}", "", ""))
        for k in only_b:
            rows.append((cls, "", k, "", f"only in {names[1]}", "", ""))
        for ka, kb in pairs:
            sa, sb = a.items[cls][ka], b.items[cls][kb]
            pa, pb = a.props(sa), b.props(sb)
            for prop in sorted(set(pa) | set(pb)):
                va = [vkey(o, a) for o in pa.get(prop, [])]
                vb = [vkey(o, b) for o in pb.get(prop, [])]
                va = [v for v in va if v != ""]
                vb = [v for v in vb if v != ""]
                if prop in TEXT_PROPS and len(va) <= 1 and len(vb) <= 1:
                    x, y = (va or [""])[0], (vb or [""])[0]
                    if not x or not y:
                        kind = f"only in {names[0]}" if x else f"only in {names[1]}"
                    else:
                        kind = text_kind(x, y)
                    prop_stats[prop][kind] += 1
                    if kind != "same":
                        rows.append((cls, ka, kb, prop, kind, x, y))
                else:
                    sa_, sb_ = set(va), set(vb)
                    if sa_ == sb_:
                        prop_stats[prop]["same"] += 1
                        continue
                    if {loose(v) for v in sa_} == {loose(v) for v in sb_}:
                        prop_stats[prop]["spaces/punctuation"] += 1
                        rows.append((cls, ka, kb, prop, "spaces/punctuation", " ; ".join(sorted(sa_)),
                                     " ; ".join(sorted(sb_))))
                        continue
                    prop_stats[prop]["different"] += 1
                    rows.append((cls, ka, kb, prop, "different",
                                 " ; ".join(sorted(sa_ - sb_)), " ; ".join(sorted(sb_ - sa_))))
        summary.append((cls, len(a.items[cls]), len(b.items[cls]), len(pairs), len(only_a), len(only_b),
                        {p: dict(v) for p, v in prop_stats.items()}))
    return summary, rows


# ---------------------------------------------------------------------------
# Fidelity to the PDF
# ---------------------------------------------------------------------------

def pdf_corpus(pdf: Path) -> str:
    """The PDF's text as pdf_to_ttl reads it (paragraphs, tables and figure
    text included), footnote marks out."""
    import pymupdf
    from pdf_to_ttl import build_paras, clean, read_lines

    doc = pymupdf.open(pdf)
    paras = build_paras(read_lines(doc))
    # Running text apart from footnotes and from table and figure text, so
    # that a sentence broken by a footnote or a table is found whole.
    streams = [("body", "h2", "h3", "h4", "title"), ("small",), ("caption",), ("footnote",), ("other",)]
    return unbreak(" || ".join(squash(" ".join(clean(p.text) for p in paras if p.role in roles))
                               for roles in streams))


def unbreak(text: str) -> str:
    """'decision- making' (a hyphen at the end of a block) as 'decision-making'."""
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", "", text)      # exponents (km²): the reference text has the plain base
    return re.sub(r"(\w)[-‑] (?=\w)", r"\1-", text)


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n+", str(text)):
        parts = re.split(r"(?<=[.;:?!])\s+(?=[A-Z(“\"‘])", squash(para))
        out += [p.strip() for p in parts if len(p.strip()) >= 40]
    return out


def fidelity(t: TTL, corpus: str) -> dict[tuple[str, str], tuple[int, int, list[str]]]:
    """{(class, property): (sentences found, sentences, examples not found)}"""
    out = {}
    for cls in CLASSES:
        if cls == "Reference":
            continue
        for prop in ("hasDescription", "prefLabel"):
            found = total = 0
            missing = []
            p = SKOS.prefLabel if prop == "prefLabel" else URIRef(t.ns + prop)
            for s in t.items[cls].values():
                for v in t.g.objects(s, p):
                    for sent in sentences(str(v)):
                        if sent == "Notes":
                            continue
                        total += 1
                        if unbreak(sent) in corpus:
                            found += 1
                        elif len(missing) < 400:
                            missing.append(sent)
            if total:
                out[(cls, prop)] = (found, total, missing)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write(summary, rows, fid, a: TTL, b: TTL, out: Path, names, corpus: str | None) -> None:
    md = [f"# {a.path.name} and {b.path.name} compared", "",
          f"{names[0]}: `{a.path}`  ", f"{names[1]}: `{b.path}`", "",
          "## Items", "",
          f"| Class | {names[0]} | {names[1]} | paired | only {names[0]} | only {names[1]} |",
          "|---|---:|---:|---:|---:|---:|"]
    for cls, na, nb, np_, oa, ob, _ in summary:
        md.append(f"| {cls} | {na} | {nb} | {np_} | {oa} | {ob} |")
    md += ["", "## Properties of paired items", "",
           "For each property of the items found in both files: how many are the same, differ only in "
           "spaces, punctuation or case, differ, or are in one file only.", "",
           "| Class | Property | same | spaces/punctuation | different | "
           f"only {names[0]} | only {names[1]} |", "|---|---|---:|---:|---:|---:|---:|"]
    for cls, *_, stats in summary:
        for prop, st in sorted(stats.items()):
            md.append(f"| {cls} | {prop} | {st.get('same', 0)} | {st.get('spaces/punctuation', 0)} | "
                      f"{st.get('different', 0)} | {st.get('only in ' + names[0], 0)} | "
                      f"{st.get('only in ' + names[1], 0)} |")
    if fid:
        fa, fb = fid
        md += ["", "## Fidelity to the PDF", "",
               "Share of sentences (40 characters or more) found word for word in the PDF text.", "",
               f"| Class | Property | {names[0]} | {names[1]} |", "|---|---|---:|---:|"]
        for key in sorted(set(fa) | set(fb), key=lambda k: (CLASSES.index(k[0]), k[1])):
            def cell(f):
                if key not in f:
                    return "–"
                found, total, _ = f[key]
                return f"{found / total:.0%} ({found}/{total})"
            md.append(f"| {key[0]} | {key[1]} | {cell(fa)} | {cell(fb)} |")
        for label, f in ((names[0], fa), (names[1], fb)):
            miss = [(k, s) for k, (_, _, m) in f.items() for s in m]
            if miss:
                md += ["", f"### {label}: sentences not found in the PDF (first 25)", ""]
                md += [f"- {k[0]} {k[1]}: {s[:300]}" for k, s in miss[:25]]

    md += ["", "## Examples", "", f"Every difference is in `{out.with_suffix('.csv').name}`.", ""]
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[3], r[4])].append(r)
    for (cls, prop, kind), rs in sorted(by.items(), key=lambda kv: (CLASSES.index(kv[0][0]), kv[0][1], kv[0][2])):
        md += [f"### {cls} {prop or ''} — {kind} ({len(rs)})".replace("  ", " "), ""]
        for r in rs[:8]:
            item = r[1] if r[1] == r[2] or not r[2] else (f"{r[1]} / {r[2]}" if r[1] else r[2])
            if kind == "different" and prop in TEXT_PROPS:
                md.append(f"- **{item}**: {first_difference(r[5], r[6])}")
            elif r[5] or r[6]:
                md.append(f"- **{item}**: {names[0]}: {r[5][:200]} | {names[1]}: {r[6][:200]}")
            else:
                md.append(f"- {item}")
        md.append("")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["class", f"item ({names[0]})", f"item ({names[1]})", "property", "difference",
                    names[0], names[1]])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5][:32000], r[6][:32000]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("first", type=Path, help="e.g. the manual conversion, data/ttl/BBA_v01.ttl")
    ap.add_argument("second", type=Path, help="e.g. data/pdf_compare/BBA/BBA_pdf.ttl")
    ap.add_argument("--pdf", type=Path, help="the report, to measure fidelity to its text")
    ap.add_argument("--names", default="manual,PDF", help='labels of the two files (default "manual,PDF")')
    ap.add_argument("--out", type=Path, help="report path without extension")
    args = ap.parse_args(argv)
    names = tuple(args.names.split(","))
    a, b = TTL(args.first), TTL(args.second)
    summary, rows = compare(a, b, names)
    fid = None
    if args.pdf:
        corpus = pdf_corpus(args.pdf)
        fid = (fidelity(a, corpus), fidelity(b, corpus))
    out = args.out or args.second.with_name(args.second.stem.replace("_pdf", "") + "_compare")
    write(summary, rows, fid, a, b, out, names, None)
    print(f"-> {out.with_suffix('.md')}, {out.with_suffix('.csv')}")
    for cls, na, nb, np_, oa, ob, _ in summary:
        print(f"  {cls:18} {names[0]} {na:5}  {names[1]} {nb:5}  paired {np_:5}  only {names[0]} {oa:4}  "
              f"only {names[1]} {ob:4}")


if __name__ == "__main__":
    sys.exit(main())
