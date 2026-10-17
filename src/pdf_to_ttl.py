"""Convert an IPBES assessment report (PDF) to RDF/Turtle following the IPBES
report ontology (docs/ontology/IPBES_Ontology_v06.ttl), in the form of the
files in data/ttl.

    PYTHONPATH=src python src/pdf_to_ttl.py BBA                 # -> data/pdf_compare/BBA/
    PYTHONPATH=src python src/pdf_to_ttl.py BBA --out somewhere/
    PYTHONPATH=src python src/pdf_to_ttl.py BBA --refresh-zotero

The assessment's settings (PDF, report code, Zotero group, experts list) are
in config/pdf_to_ttl.toml. Each run writes, to data/pdf_compare/<A>/ (or --out):

    <A>.md               the report as read from the PDF: headings, paragraphs,
                         captions, footnotes, reference lists, with page numbers.
                         Everything in the TTL comes from here; check it first.
    <A>_pdf.ttl          the TTL
    <A>_conversion.md    what could not be read or matched, and where the
                         sources disagree (PDF chapter pages, the author annex, the
                         experts list, Zotero)
    <A>_zotero.json      the Zotero library, cached (--refresh-zotero to renew)

What is read from where:
  * Report: title and co-chairs (title pages), DOI (citation on page 2).
  * Chapter: number, title, DOI and authors by role (the chapter's first
    page); people's names and roles are checked against the author annex and the
    experts list, which also gives ORCID iDs.
  * SubChapter (1.1, 1.2 ...): title (PDF outline), text of the section and
    its sub-sections, without boxes, figures, tables and footnote marks.
    Footnotes follow the text under "Notes".
  * KeyMessage, BackgroundMessage, SubMessage: the SPM. A background message
    is cut into sub-messages at each {traceability} mark, as in the other
    files; each sub-message links to the sub-chapters it cites.
  * Illustration: every figure, table and box: caption title as
    skos:prefLabel, the legend (and a box's text) as ipbes:hasDescription.
  * KnowledgeGap: the SPM table of knowledge gaps, with category and
    traceability, numbered from 1 in table order (as on IBOK).
  * Reference: the chapters' reference lists. Each entry is matched to its
    Zotero item (DOI, else first author, year and title) for the item key
    used in the URI; the entry's text is kept as ipbes:hasDescription. A
    reference links to the sub-chapters that cite it (author-year citations
    in the text).

Countries are ISO 3166 alpha-3 codes from data/rdf/ipbes-geo.rdf, one value
per country. Text is kept as printed (UTF-8, curly quotes, dashes); empty
values are left out.

Compare the result with the manual conversion using ttl_compare.py.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "pdf_to_ttl.toml"
DEFAULT_OUT = ROOT / "data" / "pdf_compare"     # + /<assessment>


# ---------------------------------------------------------------------------
# Reading the PDF: lines, classified by their style
# ---------------------------------------------------------------------------
#
# The IPBES assessment template (InDesign): body text 8.5 pt, headings in
# teal (15.6 pt for 1.1, 12.7 pt for 1.1.1, 10.7 pt below), captions 10 pt
# bold, legends, boxes and reference lists 7.5 pt, footnotes 6.5-6.9 pt grey,
# labels such as KM1, A1 or executive-summary numbers in white on teal.
# Table cells (7 pt) and text inside figures are not read.

# The accent colour of headings, links and cross-references differs by
# assessment (BBA teal 0x0F8575, IAS red 0xC72127); read_lines() sets it.
ACCENT = 0x0F8575
GREY = 0x58595B
WHITE = 0xFFFFFF
def dark(color: int) -> bool:
    """Black or near-black text (0x231F20, 0x3C3C3B ...), not the mid-grey
    of footnotes and running headers (0x58595B)."""
    return all(((color >> shift) & 0xFF) < 0x48 for shift in (16, 8, 0))

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
                            " ": " ", "\t": " ", " ": " ", " ": " ",
                            "﻿": "", "​": ""})


@dataclass
class Span:
    text: str
    font: str
    size: float
    color: int
    bold: bool
    italic: bool
    sup: bool


@dataclass
class Line:
    page: int
    block: int
    x0: float
    y0: float
    x1: float
    y1: float
    spans: list[Span]
    horizontal: bool
    role: str = ""

    @property
    def main(self) -> Span:
        return max(self.spans, key=lambda s: len(s.text.strip()))

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")      # e.g. BEL in SPM captions


def _span(s: dict) -> Span:
    font = s["font"]
    return Span(
        text=_CONTROL.sub("", s["text"].translate(_LIGATURES)),
        font=font,
        size=round(s["size"], 1),
        color=s["color"],
        bold=bool(s["flags"] & 16) or any(k in font for k in ("Bd", "Bold", "Hv", "Black", "Heavy")),
        italic=bool(s["flags"] & 2) or "It" in font,
        sup=bool(s["flags"] & 1),
    )


def _near(a: float, b: float, tol: float = 0.25) -> bool:
    return abs(a - b) <= tol


CAPTION_START = re.compile(r"^\s*(Figure|Table|Box)\s+(SPM\s*(?:[A-Z]\s*)?\d+|[A-Z]?\d+\s*\.?\s*\d+[a-z]?)\b")


_URL_TEXT = re.compile(r"\S*(?:https?://|www\.|doi\.|\.org|\.com|/)\S*|[\w.%-]+[/.][\w./%-]*")


def _link_span(s: Span) -> bool:
    """A link printed in the accent colour (in IAS in another font and size
    than the text around it)."""
    return s.color == ACCENT and not s.bold and bool(_URL_TEXT.fullmatch(s.text.strip()))


def line_role(line: Line) -> str:
    # Judge a line by its text, not by a link in it.
    words = [s for s in line.spans if s.text.strip() and not _link_span(s)]
    if not words and line.horizontal and line.y0 > 62:
        size = line.main.size
        if 6.2 <= size < 6.8:
            return "footnote"   # a link continuing a footnote
        return "small" if 6.8 <= size <= 7.6 else "body" if _near(size, 8.5, 0.3) else "other"
    m = max(words, key=lambda s: len(s.text.strip())) if words else line.main
    if not line.horizontal:
        # Rotated: the side tab ("SUMMARY FOR POLICYMAKERS"), or a table
        # printed sideways, whose caption is still read.
        if m.bold and _near(m.size, 10.0) and CAPTION_START.match(line.text):
            return "caption"
        return "other"
    if line.y0 < 62 and m.size <= 7.6 and (m.color == GREY or dark(m.color)):
        return "header"
    if line.y0 > 790 or line.x0 < 45:
        return "folio"
    if m.size >= 20:
        return "title"
    if m.color == WHITE:
        return "other"
    if _near(m.size, 15.6) and m.color == ACCENT:
        return "h2"
    if _near(m.size, 12.7) and m.color == ACCENT:
        return "h3"
    if _near(m.size, 10.7) and "Hv" in m.font:
        return "h4"
    if _near(m.size, 10.0) and m.bold and dark(m.color):
        return "caption"
    if _near(m.size, 9.0) and m.bold and "DIN" in m.font and dark(m.color):
        return "h5"             # a heading inside a box (TCA Box SPM 9)
    if _near(m.size, 8.5, 0.3) and (dark(m.color) or m.color == ACCENT):
        return "body"
    if _near(m.size, 7.5) and (dark(m.color) or m.color == ACCENT) and "Roman" not in m.font:
        return "small"
    if 6.2 <= m.size <= 6.95 and (m.color == GREY or (not dark(m.color) and m.color != WHITE and (
            re.match(r"^\s*\d{1,3}\.\s", line.text) or _URL_TEXT.search(line.text)))):
        return "footnote"       # grey, or a numbered or linked footnote in a colour (IAS)
    return "other"


def accent_color(doc: pymupdf.Document) -> int:
    """The colour most used in the heading sizes (15.6 and 12.7 pt)."""
    import collections
    count: collections.Counter = collections.Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["text"].strip() and (_near(s["size"], 15.6) or _near(s["size"], 12.7)) \
                            and s["color"] not in (WHITE, GREY) and not dark(s["color"]):
                        count[s["color"]] += len(s["text"])
    return count.most_common(1)[0][0] if count else ACCENT


def read_lines(doc: pymupdf.Document) -> list[Line]:
    global ACCENT
    ACCENT = accent_color(doc)
    lines: list[Line] = []
    for pno, page in enumerate(doc, 1):
        for bno, b in enumerate(page.get_text("dict")["blocks"]):
            for l in b.get("lines", []):
                spans = [_span(s) for s in l["spans"]]
                if not any(s.text.strip() for s in spans):
                    continue
                line = Line(pno, bno, l["bbox"][0], l["bbox"][1], l["bbox"][2], l["bbox"][3], spans,
                            horizontal=abs(l["dir"][0] - 1) < 0.01)
                line.role = line_role(line)
                lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------

@dataclass
class Para:
    role: str           # body, small, footnote, caption, h2, h3, h4, title, other
    page: int
    x0: float
    y0: float
    lines: list[Line] = field(default_factory=list)
    label: str = ""     # white label at the start: KM1, A1, an executive-summary number
    lead: str = ""      # bold text at the start (a message's headline)
    text: str = ""      # whole text, footnote marks as [^n]

    @property
    def y1(self) -> float:
        return max(line.y1 for line in self.lines)

    @property
    def x1(self) -> float:
        return max(line.x1 for line in self.lines)

    @property
    def rest(self) -> str:
        """The text after the bold lead."""
        return self.text[len(self.lead):].strip() if self.lead else self.text


FOOTNOTE_MARK = re.compile(r"\[\^(\d+)\]")
_URLISH = re.compile(r"(https?://|www\.|doi\.org|/)\S*$")


def _span_text(s: Span, role: str) -> str:
    t = s.text
    # Footnote marks: superscript numbers in running text.
    if s.sup and role in ("body", "small", "caption") and re.fullmatch(r"\s*[\d,–-]+\s*", t):
        return "".join(f"[^{n}]" for n in re.findall(r"\d+", t))
    return t


def _join(prev: str, nxt: str, prev_link: bool, next_link: bool) -> str:
    """Join two printed lines of one paragraph."""
    if not prev:
        return nxt.lstrip()
    if prev.endswith("­"):                       # discretionary hyphen
        return prev[:-1] + nxt.lstrip()
    p = prev.rstrip()
    n = nxt.lstrip()
    # A link broken over lines: printed in teal, or ending in "https://", "/"...
    if (prev_link and next_link) or (p == prev and _URLISH.search(p) and not n[:1].isupper()):
        return p + n
    if p.endswith(("-", "/")) and p[-2:-1].isalnum():
        # Hyphens at line ends in IPBES reports are real ("decision-making").
        return p + n
    return p + " " + n


def _is_link(s: Span) -> bool:
    return s.color == ACCENT and not s.bold


def build_paras(lines: list[Line]) -> list[Para]:
    paras: list[Para] = []
    cur: Para | None = None
    for line in lines:
        if line.role in ("header", "folio"):
            continue
        same = (cur is not None and cur.page == line.page and cur.lines[-1].block == line.block
                and cur.role == line.role)
        # A footnote starts with its number: "7.\t This categorization..."
        if same and line.role == "footnote" and re.match(r"^\d+\.\s", line.text) \
                and line.x0 < cur.lines[-1].x0 - 3:
            same = False
        if not same:
            cur = Para(line.role, line.page, line.x0, line.y0)
            paras.append(cur)
        cur.lines.append(line)

    for p in paras:
        _fill(p)
    paras = reading_order([p for p in paras if p.text or p.label])

    # A label on its own (KM1 in white on teal) belongs to the paragraph that
    # starts on the same line just to its right.
    labelled: dict[int, str] = {}
    for p in paras:
        if p.role == "other" and LABEL.fullmatch(p.text) and _white(p):
            target = next((q for q in paras if q.page == p.page and q.role == "body"
                           and abs(q.y0 - p.y0) < 6 and 0 < q.x0 - p.x0 < 60), None)
            if target is not None:
                labelled[id(target)] = p.text

    out: list[Para] = []
    for p in paras:
        if p.role == "other" and LABEL.fullmatch(p.text) and _white(p):
            continue
        if id(p) in labelled:
            p.label = p.label or labelled[id(p)]
        if p.role == "caption" and not CAPTION_START.match(p.text):
            prev = out[-1] if out else None
            if prev and prev.role == "caption" and prev.page == p.page:
                prev.text = f"{prev.text} {p.text}"
                prev.lines += p.lines
                continue
            p.role = "other"        # bold 10 pt text inside a figure
        out.append(p)
    return out


def reading_order(paras: list[Para], page_width: float = 595.0) -> list[Para]:
    """Paragraphs in the order a reader takes them. The PDF's own order is
    not always that: a heading in the right column can come before the left
    column's text that ends the previous section. Full-width items (captions,
    wide headings and text) cut the page into bands; within a band, columns
    are read left to right, each top to bottom. Columns are found per page
    from where the paragraphs start (two columns of text, three of references)."""
    out: list[Para] = []
    pages: dict[int, list[Para]] = {}
    for p in paras:
        pages.setdefault(p.page, []).append(p)
    for page in sorted(pages):
        items = pages[page]

        def wide(p: Para) -> bool:
            return (p.role not in ("other", "footnote") and p.lines[0].horizontal
                    and p.x1 - p.x0 > page_width * 0.55)

        bands = sorted((p for p in items if wide(p)), key=lambda p: p.y0)
        starts = sorted(p.x0 for p in items if not wide(p) and p.lines[0].horizontal)
        columns: list[float] = []
        for x in starts:
            if not columns or x - columns[-1] > 60:
                columns.append(x)
            # a column is known by its leftmost start; indented lines join it

        def column(p: Para) -> int:
            return max((i for i, c in enumerate(columns) if p.x0 >= c - 5), default=0)

        def key(p: Para):
            if not p.lines[0].horizontal:
                return (10 ** 6, 0, p.y0)
            if p in bands:
                return (2 * bands.index(p) + 1, 0, p.y0)
            band = sum(1 for b in bands if b.y0 < p.y0)
            return (2 * band, column(p), p.y0, p.x0)

        out += sorted(items, key=key)
    return out


LABEL = re.compile(r"KM\s?-?\s?[A-F]?\d{1,2}|[A-F]\s?\d{1,2}|\d{1,2}")


def _white(p: Para) -> bool:
    return all(s.color == WHITE for line in p.lines for s in line.spans if s.text.strip())


def _fill(p: Para) -> None:
    """Set the paragraph's text, white label and bold lead from its spans."""
    text, label = "", ""
    lead_end: int | None = None     # length of text where the bold lead ends
    prev_link = False
    for line in p.lines:
        parts = []
        for s in line.spans:
            if (s.color == WHITE and p.role == "body" and not text.strip()
                    and not "".join(x.text for x in parts).strip()):
                label += s.text.strip()
                continue
            parts.append(s)
        if not parts:
            continue
        line_text = ""
        for s in parts:
            t = _span_text(s, p.role)
            if lead_end is None and t.strip() and not s.bold:
                lead_end = len(_join(text, line_text, prev_link, _is_link(parts[0]))) if text else len(line_text)
            line_text += t
        text = _join(text, line_text, prev_link, _is_link(parts[0])) if text else line_text
        prev_link = _is_link(parts[-1])
    if lead_end is None:
        lead_end = len(text)
    lead = text[:lead_end] if p.role in ("body", "small") else ""
    p.text = re.sub(r"\s+", " ", text).strip()
    p.lead = re.sub(r"\s+", " ", lead).strip()
    if p.lead and not p.text.startswith(p.lead):
        p.lead = ""
    p.label = label


# ---------------------------------------------------------------------------
# The report's structure
# ---------------------------------------------------------------------------

@dataclass
class Footnote:
    number: int
    text: str
    page: int


@dataclass
class Section:
    """A numbered section (1.1, 1.1.1 ...) with its paragraphs."""
    number: str
    title: str
    page: int
    paras: list[str] = field(default_factory=list)   # "## 1.1.1 Title" lines for sub-headings

    @property
    def chapter(self) -> int:
        return int(self.number.split(".")[0])


@dataclass
class Illustration:
    kind: str           # Figure, Table, Box
    number: str         # SPM.1, SPM.A1, 1.1
    title: str
    page: int
    legend: list[str] = field(default_factory=list)
    section: str = ""   # the section it is printed in (1.1), "" in the SPM

    @property
    def ident(self) -> str:
        return f"{self.kind}{self.number}"


@dataclass
class Message:
    """A key message (KM1), background message (A1) or executive-summary finding."""
    ident: str
    headline: str
    page: int
    paras: list[str] = field(default_factory=list)
    group: str = ""     # background messages: "A. Understanding the relationship ..."

    @property
    def text(self) -> str:
        return " ".join(self.paras)


@dataclass
class RefEntry:
    text: str
    page: int


@dataclass
class Chapter:
    number: int
    title: str = ""
    page: int = 0
    doi: str = ""
    cite_as: str = ""
    authors: dict[str, list[str]] = field(default_factory=dict)   # role heading -> "Name (Country)"
    executive_summary: list[Message] = field(default_factory=list)
    unnumbered: list[Section] = field(default_factory=list)       # "KNOWLEDGE GAPS" before 5.1
    sections: list[Section] = field(default_factory=list)
    references: list[RefEntry] = field(default_factory=list)


@dataclass
class Report:
    title: str = ""
    doi: str = ""
    editors: list[tuple[str, str]] = field(default_factory=list)  # (name, "affiliation, country")
    key_messages: list[Message] = field(default_factory=list)
    background_messages: list[Message] = field(default_factory=list)
    illustrations: list[Illustration] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    footnotes: dict[tuple[int, int], Footnote] = field(default_factory=dict)  # (chapter or 0, n)
    gap_table_pages: list[int] = field(default_factory=list)
    annex_pages: dict[str, list[int]] = field(default_factory=dict)
    outline: dict[str, str] = field(default_factory=dict)     # "1.1" -> title, from the PDF outline
    problems: list[str] = field(default_factory=list)

    def chapter(self, n: int) -> Chapter | None:
        return next((c for c in self.chapters if c.number == n), None)


SECTION_NUMBER = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,4})\.?\s+(.*)$")
_TERMINAL = tuple(".:;?!)”\"]}’")


def _caption(text: str) -> tuple[str, str, str]:
    """'Figure SPM 2  Creating ...' -> ("Figure", "SPM.2", "Creating ...")."""
    m = re.match(r"^\s*(Figure|Table|Box)\s+(SPM)?\s*([A-Z])?\s*(\d+)(?:\s*[.\s]\s*(\d+))?\s*(.*)$", text)
    if not m:
        return "", "", text
    kind, spm, letter, a, b, rest = m.groups()
    if spm:
        number = f"SPM.{letter or ''}{a}"
        title = f"{b} {rest}".strip() if b else rest    # "SPM 1 0 ..." does not occur
    else:
        number = f"{a}.{b}" if b else a
        title = rest
    return kind, number, title.strip()


def _continues(prev: str, nxt: str) -> bool:
    """Whether a paragraph block continues the previous one (a column or
    page break, or a figure, in the middle of a sentence)."""
    if not prev:
        return False
    first = nxt.lstrip()[:1]
    return first.islower() or not prev.rstrip().endswith(_TERMINAL) or prev.rstrip().endswith(",")


def _add_para(paras: list[str], text: str) -> None:
    if paras and not paras[-1].startswith("## ") and _continues(paras[-1], text):
        paras[-1] = _join(paras[-1], text, False, False) if paras[-1].endswith("-") else paras[-1] + " " + text
    else:
        paras.append(text)


def parse(doc: pymupdf.Document, paras: list[Para]) -> Report:
    rep = Report()
    _front_matter(doc, rep)
    for level, title, page in doc.get_toc():
        m = re.match(r"^\s*(\d{1,2}\.\d{1,2})\.?\s+(.+)$", re.sub(r"\s+", " ", title))
        if m:
            rep.outline[m.group(1)] = m.group(2).strip()

    mode = "front"          # front, km, bgm, appendix, chapter, exec, refs, annex
    chapter: Chapter | None = None
    section: Section | None = None
    message: Message | None = None
    group = ""
    box: Illustration | None = None     # the box being read (its text is 7.5 pt)
    legend_of: Illustration | None = None   # the figure or table whose legend may follow
    legend_end = (0, 0.0)               # page and bottom of its caption or legend so far
    unnumbered: Section | None = None

    def footnote_chapter() -> int:
        return chapter.number if chapter else 0

    for i, p in enumerate(paras):
        text = p.text

        # "Box 2 1" alone: a box, figure or table continued on another page.
        kind, number, title = _caption(text)
        if kind and not title and p.role in ("other", "small", "caption"):
            existing = next((x for x in rep.illustrations if x.kind == kind and x.number == number), None)
            if existing and kind == "Box":
                box, legend_of = existing, None
            continue

        if p.role == "other" and re.fullmatch(r"Chapter \d{1,2}", text.strip()) and _chapter_start(text, p, doc):
            p.role = "title"        # the chapter number, printed as "Chapter 3"

        if p.role == "title":
            t = text.upper()
            box = legend_of = None
            if t == "KEY MESSAGES":
                mode, message = "km", None
            elif t in ("BACKGROUND MESSAGES", "BACKGROUND"):
                mode, message = "bgm", None
            elif t.startswith("APPENDIX") and mode in ("km", "bgm", "appendix"):
                mode = "appendix"
            elif _chapter_start(text, p, doc):
                n = int(re.sub(r"\D", "", text))
                if rep.chapter(n) is None:
                    chapter = Chapter(n, page=p.page)
                    rep.chapters.append(chapter)
                    _chapter_page(doc, p.page, chapter, rep)
                mode, section, unnumbered = "chapter", None, None
            elif t == "REFERENCES" and chapter:
                mode = "refs"
            elif t.startswith("ANNEX"):
                mode = "annex"
                rep.annex_pages.setdefault(t.split("\n")[0], []).append(p.page)
            continue

        if mode == "annex":
            last = list(rep.annex_pages)[-1] if rep.annex_pages else ""
            if last and p.page not in rep.annex_pages[last]:
                rep.annex_pages[last].append(p.page)
            continue

        if p.role == "footnote":
            m = re.match(r"^(\d{1,3})\.?\s+(.*)$", text)
            if m:
                key = (footnote_chapter(), int(m.group(1)))
                rep.footnotes[key] = Footnote(int(m.group(1)), m.group(2), p.page)
            elif rep.footnotes:
                # A footnote continued on the next page or column.
                last = list(rep.footnotes)[-1]
                rep.footnotes[last].text += " " + text
            continue

        if p.role == "caption":
            kind, number, title = _caption(text)
            if not kind or not re.match(r"[\w“\"‘(]", title) or title.startswith(")"):
                continue
            spm = number.startswith("SPM")
            if not spm and chapter and not number.startswith(f"{chapter.number}."):
                rep.problems.append(f"p{p.page}: caption '{text[:60]}' numbered outside chapter {chapter.number}")
            existing = next((x for x in rep.illustrations if x.kind == kind and x.number == number), None)
            if existing:
                rep.problems.append(f"p{p.page}: {kind} {number} captioned twice")
                continue
            ill = Illustration(kind, number, title, p.page, section=section.number if section else "")
            rep.illustrations.append(ill)
            if kind == "Box":
                box, legend_of = ill, None
            else:
                legend_of, legend_end = ill, (p.page, p.y1)
            if spm and kind == "Table" and "gap" in title.lower() and "knowledge" in title.lower():
                rep.gap_table_pages.append(p.page)
            continue

        if p.role == "small":
            if mode == "refs":
                if chapter.references and _ref_continues(chapter.references[-1].text, text):
                    chapter.references[-1].text = _join(chapter.references[-1].text, text,
                                                        True, True) if _URLISH.search(chapter.references[-1].text) \
                        else chapter.references[-1].text + " " + text
                else:
                    chapter.references.append(RefEntry(text, p.page))
            elif legend_of and legend_end[0] == p.page and -2 < p.y0 - legend_end[1] < 30:
                # A legend sits right under its caption.
                _add_para(legend_of.legend, text)
                legend_end = (p.page, p.y1)
            elif box and re.search(r"[a-z]", text):
                # Labels of a figure inside the box are capitals only; skip them.
                _add_para(box.legend, text)
            continue

        if p.role == "h5":
            if box is not None:
                box.legend.append("## " + text)
            continue

        if p.role in ("h2", "h3", "h4"):
            box = legend_of = None
            if mode in ("km", "bgm"):
                if mode == "bgm" and p.role == "h2":
                    group = text
                continue
            if mode not in ("chapter", "exec") or chapter is None:
                continue
            m = SECTION_NUMBER.match(text)
            if p.role == "h2" and not m:
                if text.upper() == "EXECUTIVE SUMMARY":
                    mode = "exec"
                else:
                    mode = "chapter"
                    unnumbered = Section("", text.title(), p.page)
                    chapter.unnumbered.append(unnumbered)
                    section = None
                continue
            if not m:
                rep.problems.append(f"p{p.page}: heading without a number: {text[:80]}")
                continue
            number, title = m.groups()
            mode, unnumbered = "chapter", None
            if number.count(".") == 1:
                section = Section(number, rep.outline.get(number, title), p.page)
                chapter.sections.append(section)
                if number not in rep.outline:
                    rep.problems.append(f"p{p.page}: section {number} not in the PDF outline; title as printed")
            elif section and number.startswith(section.number + "."):
                section.paras.append(f"## {number} {title}")
            else:
                rep.problems.append(f"p{p.page}: heading {number} outside its section")
            continue

        if p.role != "body":
            continue

        # Body text
        legend_of = None

        if mode == "km":
            # KM1 (BBA) or KM-A1 (IAS)
            if re.fullmatch(r"KM\s?-?\s?[A-F]?\d{1,2}", p.label or ""):
                message = Message(re.sub(r"\s", "", p.label), p.lead or text, p.page)
                rep.key_messages.append(message)
                rest = p.rest if p.lead else ""
                if rest:
                    message.paras.append(rest)
            elif message:
                _add_para(message.paras, text)
        elif mode == "bgm":
            if re.fullmatch(r"[A-F]\s?\d{1,2}", p.label or ""):
                message = Message(p.label.replace(" ", ""), p.lead or text, p.page, group=group)
                rep.background_messages.append(message)
                rest = p.rest if p.lead else ""
                if rest:
                    message.paras.append(rest)
            elif message:
                _add_para(message.paras, text)
        elif mode == "exec" and chapter:
            if re.fullmatch(r"\d{1,2}", p.label or ""):
                finding = Message(p.label, p.lead or text, p.page)
                chapter.executive_summary.append(finding)
                if p.lead and p.rest:
                    finding.paras.append(p.rest)
            elif chapter.executive_summary:
                _add_para(chapter.executive_summary[-1].paras, text)
        elif mode == "chapter":
            target = section or unnumbered
            if target is not None:
                _add_para(target.paras, text)
    # Gaps in the numbering of figures, tables and boxes ("Figure SPM 14" after 12).
    seen: dict[tuple[str, str], list[int]] = {}
    for il in rep.illustrations:
        m = re.fullmatch(r"(SPM\.|\d+\.)(\d+)", il.number)
        if m:
            seen.setdefault((il.kind, m.group(1)), []).append(int(m.group(2)))
    for (kind, prefix), nums in seen.items():
        missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
        if missing:
            rep.problems.append(f"{kind} {prefix}{', '.join(map(str, missing))} not in the report, "
                                f"though {kind} {prefix}{max(nums)} is")
    return rep


def _chapter_start(text: str, p: Para, doc: pymupdf.Document) -> bool:
    """'3' or 'Chapter 3' on a chapter's first page, which (or the next) has
    the chapter's citation."""
    if not re.fullmatch(r"(?:Chapter )?\d{1,2}", text.strip()):
        return False
    return any("SHOULD BE CITED AS" in doc[q - 1].get_text()
               for q in (p.page, p.page + 1) if q <= doc.page_count)


_REF_START = re.compile(
    r"^[^\s(](?:[^()]|\([^()]{0,1000}\)){0,1500}?\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)[^)]{0,40}\)",
    re.IGNORECASE)


def _ref_continues(prev: str, nxt: str) -> bool:
    """Whether a reference-list block continues the previous entry. Entries
    break across columns and pages anywhere, even inside the author list."""
    if nxt.startswith(("http", "www.", "doi", "/")) or nxt[:1] in ",;:.&)–" or nxt[:1].isdigit():
        return True
    if not _YEAR.search(prev):
        return True             # the previous entry has not reached its year yet
    strong = bool(_REF_START.match(nxt) and not _INITIALS_START.match(nxt)
                  and (_AUTHOR_START.match(nxt) or _ORG_START.match(nxt)))
    if strong:
        return False            # "Surname, X., ... (2020)." starts an entry
    tail = prev[-300:]
    if tail.count("(") > tail.count(")"):
        return True             # inside "(Eds ...)"
    return not (_REF_START.match(nxt) and not _INITIALS_START.match(nxt) or _AUTHOR_START.match(nxt))


_YEAR = re.compile(r"\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)", re.IGNORECASE)
# "Bull, J. W., ..." or "IPBES (with ..."
_AUTHOR_START = re.compile(r"^(?:[A-Za-z][^\s,()]*(?:\s[^\s,()]+){0,3},\s(?:[A-Z]\.|[A-Z][a-z]{0,3}\.)|[A-Z][\w-]*\s\(with\b)")
# "FAO. (2018a)." or "European Commission. (2019)."
_ORG_START = re.compile(r"^[A-Z][^()\n]{0,120}?\.\s\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.)")
# "E. J. (2020)": the end of an author list, not a new entry.
_INITIALS_START = re.compile(r"^(?:[A-Z]\.\s?){1,3}(?:,|\(|&)")


def _front_matter(doc: pymupdf.Document, rep: Report) -> None:
    for pno in range(min(6, doc.page_count)):
        text = doc[pno].get_text()
        m = re.search(r"SUGGESTED CITATION\s*\n(.*?)(?:\n\s*\n|$)", text, re.S)
        if m and not rep.title:
            cite = re.sub(r"\s+", " ", m.group(1))
            t = re.match(r"IPBES \(\d{4}\)\.\s*(.+?)\.\s+[A-Z][\w’'-]+,? [A-Z]\.", cite)
            rep.title = t.group(1) if t else ""
            d = re.search(r"(10\.\d{4,9}/[^\s]+?)\.?(?:\s|$)", cite)
            rep.doi = f"https://doi.org/{d.group(1)}" if d else ""
        if "Edited by:" in text and not rep.editors:
            block = text.split("Edited by:", 1)[1]
            # Name, then one or more affiliation lines, then "Assessment Co-Chair".
            chunks = re.split(r"\n\s*Assessment Co-?chair\s*(?=\n)", block, flags=re.IGNORECASE)
            for chunk in chunks[:-1]:
                lines = [x.strip() for x in chunk.split("\n") if x.strip()]
                if lines:
                    rep.editors.append((lines[0], " ".join(lines[1:])))


ROLE_HEADINGS = {
    "COORDINATING LEAD AUTHORS": "cl",
    "COORDINATING LEAD AUTHOR": "cl",
    "LEAD AUTHORS": "la",
    "LEAD AUTHOR": "la",
    "FELLOWS": "fl",
    "FELLOW": "fl",
    "CONTRIBUTING AUTHORS": "ca",
    "CONTRIBUTING AUTHOR": "ca",
    "REVIEW EDITORS": "re",
    "REVIEW EDITOR": "re",
    "TECHNICAL SUPPORT UNIT": "tsu",
}


def _page_text(page: pymupdf.Page) -> str:
    """A page's text without its footnotes and footnote numbers, which can sit
    in the middle of an author list."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [_span(x) for x in l["spans"] if x["text"].strip()]
            if not spans:
                continue
            m = max(spans, key=lambda x: len(x.text))
            if m.size < 6.75 or m.size > 9:
                continue            # footnotes (6.5 pt) and titles; the lists are 7 pt
            out.append("".join(x.text for x in spans if not (x.sup and x.text.strip().isdigit())))
    return "\n".join(out)


def _chapter_page(doc: pymupdf.Document, page: int, chapter: Chapter, rep: Report) -> None:
    """Authors, citation and DOI from the chapter's first page (and the next,
    when the lists or the citation continue there)."""
    text = _page_text(doc[page - 1])
    if "SHOULD BE CITED AS" not in text and page < doc.page_count:
        text += "\n" + _page_text(doc[page])
    text = text.translate(_LIGATURES)
    # The map disclaimer on the second page can sit inside an author list.
    text = re.sub(r"Disclaimer on maps.*?represented therein\.?", "", text, flags=re.S)
    heads = "|".join(sorted(map(re.escape, ROLE_HEADINGS), key=len, reverse=True))
    # "REVIEW EDITORS 2:": a footnote number may follow the heading.
    parts = re.split(rf"\n\s*({heads})\s*\d*\s*:\s*\n", "\n" + text)
    for head, body in zip(parts[1::2], parts[2::2]):
        body = re.split(r"THIS CHAPTER SHOULD BE CITED AS|\nChapter \d+\n", body)[0]
        body = re.sub(r"\n\s*\d+\s*\n", "\n", body)        # stray footnote numbers
        chapter.authors[ROLE_HEADINGS[head]] = _split_people(re.sub(r"\s+", " ", body).strip())
    # The citation ends with the DOI, which may be broken after a "/".
    m = re.search(r"THIS CHAPTER SHOULD BE CITED AS:\s*(.*?(?:DOI:\s*)?https?://doi\.org/(?:\S*/[ \t]*\n\s*\S+|\S+))",
                  text, re.S)
    if m:
        cite = re.sub(r"(doi\.org/\S*/)\s+", r"\1", m.group(1))
        cite = re.sub(r"\s+", " ", cite).strip()
        chapter.cite_as = cite
        d = re.search(r"(10\.\d{4,9}/\S+?)\.?(?:\s|$)", cite)
        chapter.doi = f"https://doi.org/{d.group(1)}" if d else ""
        t = re.search(r"Chapter \d+:\s*(.+?)\s+In:", cite)
        chapter.title = t.group(1).strip().rstrip(".") if t else ""
    if not chapter.doi:
        rep.problems.append(f"chapter {chapter.number}: no DOI on its first pages (p{page})")


def _split_people(text: str) -> list[str]:
    """'A (X / Y), B (Z).' -> ['A (X / Y)', 'B (Z)']; names without countries
    (the technical support unit) are split on commas."""
    text = text.rstrip(". ")
    if "(" not in text:
        return [n.strip() for n in text.split(",") if n.strip()]
    # A name may hold a nickname: "Margaret (Peggy) Smith (Canada)".
    return [m.group(0).strip(" ,") for m in re.finditer(r"[^,()]+(?:\([^()]*\)[^,()]+)?\([^()]*\)", text)]


# ---------------------------------------------------------------------------
# Markdown: the report as read, for checking
# ---------------------------------------------------------------------------

def to_markdown(rep: Report, name: str) -> str:
    out = [f"# {rep.title}", "", f"Read from {name}. DOI: {rep.doi}", ""]
    out += [f"- Co-chair: {n} ({a})" for n, a in rep.editors] + [""]

    def ills(spm: bool, chapter: int = 0):
        for il in rep.illustrations:
            if il.number.startswith("SPM") != spm or (not spm and not il.number.startswith(f"{chapter}.")):
                continue
            out.append(f"### {il.kind} {il.number} (p{il.page}{', in ' + il.section if il.section else ''})")
            out.append(f"**{il.title}**")
            out.append("")
            for para in il.legend:
                out.extend([para, ""])

    out += ["## Summary for policymakers", "", "### Key messages", ""]
    for km in rep.key_messages:
        out += [f"#### {km.ident} (p{km.page})", f"**{km.headline}**", ""]
        out += [x for para in km.paras for x in (para, "")]
    out += ["### Background messages", ""]
    group = None
    for bgm in rep.background_messages:
        if bgm.group != group:
            group = bgm.group
            out += [f"#### {group}", ""]
        out += [f"##### {bgm.ident} (p{bgm.page})", f"**{bgm.headline}**", ""]
        out += [x for para in bgm.paras for x in (para, "")]
    out += ["### SPM figures and tables", ""]
    ills(True)
    fn0 = [f for (c, _), f in rep.footnotes.items() if c == 0]
    if fn0:
        out += ["### SPM footnotes", ""] + [f"[^{f.number}]: {f.text} (p{f.page})" for f in fn0] + [""]

    for ch in rep.chapters:
        out += [f"## Chapter {ch.number}. {ch.title} (p{ch.page})", "", f"DOI: {ch.doi}", ""]
        for role, people in ch.authors.items():
            out.append(f"- {role}: {'; '.join(people)}")
        out += ["", f"Cite as: {ch.cite_as}", "", "### Executive summary", ""]
        for f in ch.executive_summary:
            out += [f"{f.ident}. **{f.headline}** {f.text}", ""]
        for sec in ch.unnumbered + ch.sections:
            out += [f"### {sec.number} {sec.title} (p{sec.page})".replace("###  ", "### "), ""]
            for para in sec.paras:
                out += [("#" + para) if para.startswith("## ") else para, ""]
        out += [f"### Figures, tables and boxes of chapter {ch.number}", ""]
        ills(False, ch.number)
        fns = [f for (c, _), f in rep.footnotes.items() if c == ch.number]
        if fns:
            out += [f"### Footnotes of chapter {ch.number}", ""] + \
                   [f"[^{f.number}]: {f.text} (p{f.page})" for f in fns] + [""]
        out += [f"### References of chapter {ch.number} ({len(ch.references)})", ""]
        out += [f"- {r.text}" for r in ch.references] + [""]
    if rep.problems:
        out += ["## Problems while reading", ""] + [f"- {p}" for p in rep.problems]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Zotero: the assessment's library, for reference keys and DOIs
# ---------------------------------------------------------------------------

def zotero_items(group: int, cache: Path, refresh: bool = False) -> list[dict]:
    """All top-level items of a public Zotero group, with their collections'
    names, cached as JSON."""
    import json
    import httpx

    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    api = f"https://api.zotero.org/groups/{group}"
    with httpx.Client(timeout=60, headers={"Zotero-API-Version": "3"}) as client:
        collections = {}
        start = 0
        while True:
            r = client.get(f"{api}/collections", params={"limit": 100, "start": start})
            r.raise_for_status()
            batch = r.json()
            collections.update({c["key"]: c["data"]["name"] for c in batch})
            start += len(batch)
            if not batch or start >= int(r.headers.get("Total-Results", start)):
                break
        items, start = [], 0
        while True:
            r = client.get(f"{api}/items/top", params={"limit": 100, "start": start, "format": "json"})
            r.raise_for_status()
            batch = r.json()
            for it in batch:
                data = it["data"]
                items.append({
                    "key": data["key"],
                    "type": data.get("itemType", ""),
                    "title": data.get("title", ""),
                    "creators": [c.get("lastName") or c.get("name", "") for c in data.get("creators", [])],
                    "date": data.get("date", ""),
                    "doi": data.get("DOI", "") or _doi_in(data.get("extra", "")) or _doi_in(data.get("url", "")),
                    "url": data.get("url", ""),
                    "collections": [collections.get(k, k) for k in data.get("collections", [])],
                })
            start += len(batch)
            print(f"  Zotero: {start} of {r.headers.get('Total-Results')}", end="\r")
            if not batch or start >= int(r.headers.get("Total-Results", start)):
                break
        print()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)


def _doi_in(text: str) -> str:
    m = _DOI.search(text or "")
    return m.group(1).rstrip(".,;)") if m else ""


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

class Countries:
    """Country names as printed -> ISO 3166 alpha-3, from the IPBES geography.
    Names the geography does not have, even if geo.py has an alias for them
    ("America", "UK"), are reported rather than guessed."""

    def __init__(self, log: "Log"):
        import geo
        self.geo = geo
        self.log = log

    def code(self, name: str, where: str) -> str:
        name = name.strip().strip(".").strip()
        name = name.replace("[", "(").replace("]", ")")     # "Netherlands [Kingdom of the]" (IAS)
        tries = [name, name.replace("’", "'")]
        # "Kingdom of the Netherlands" -> "Netherlands (Kingdom of the)"
        m = re.match(r"^(.*\b(?:Republic|Kingdom|State|States|Federation|Commonwealth)(?: of)?(?: the)?) (?:of )?(?:the )?(\w[\w\s’']+)$", name)
        if m:
            tries.append(f"{m.group(2)} ({m.group(1)})")
            tries.append(f"{m.group(2)} ({m.group(1)} of)")
            tries.append(f"{m.group(2)} ({m.group(1)} of the)")
        alias = self.geo._COUNTRY_ALIASES.get(re.sub(r"\s+", " ", name.lower()))
        for t in tries:
            code = self.geo.country_code_for_label(t)
            if code and not (alias and code == alias and self.geo._COUNTRY_ALIASES.get(t.lower())):
                return code
        if alias:
            self.log.add("countries", f"{where}: '{name}' is not a country name in ipbes-geo.rdf; "
                                      f"read as {alias}")
            return alias
        if re.fullmatch(r"[A-Z]{2,}", name):
            self.log.add("countries", f"{where}: '{name}' is an organization, not a country; left out")
        else:
            self.log.add("countries", f"{where}: country '{name}' not found in ipbes-geo.rdf; left out")
        return ""

    def resolves(self, text: str) -> bool:
        """Whether every part of 'A, B / C' is a country (no logging)."""
        quiet = Countries.__new__(Countries)
        quiet.geo, quiet.log = self.geo, Log()
        parts = [x for x in re.split(r"\s*[/,;]\s*", text) if x.strip()]
        return bool(parts) and all(quiet.code(x, "") and not quiet.log.items for x in parts)

    def codes(self, text: str, where: str) -> list[str]:
        """'Plurinational State of Bolivia / Germany', 'India, United States of America'."""
        out = []
        for part in re.split(r"\s*[/,;]\s*", text):
            part = re.sub(r"^the\s+", "", part.strip(), flags=re.IGNORECASE)
            if part:
                code = self.code(part, where)
                if code and code not in out:
                    out.append(code)
        return out


# ---------------------------------------------------------------------------
# Log: what the conversion could not do, and where sources disagree
# ---------------------------------------------------------------------------

class Log:
    def __init__(self):
        self.items: dict[str, list[str]] = {}

    def add(self, topic: str, text: str) -> None:
        lst = self.items.setdefault(topic, [])
        if text not in lst:
            lst.append(text)


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

ROLE_NAMES = {
    "cs": "Co-chair",
    "cl": "Coordinating Lead Author",
    "la": "Lead Author",
    "fl": "Fellow",
    "ca": "Contributing Author",
    "re": "Review Editor",
}
_CSV_ROLES = {"chair": "cs", "chair - cla": "cs", "coordinating lead author": "cl", "lead author": "la", "fellow": "fl",
              "review editor": "re", "contributing author": "ca"}
_PARTICLES = {"de", "da", "das", "do", "dos", "del", "della", "der", "den", "van", "von", "zu", "di",
              "du", "la", "le", "bin", "binti", "al", "el", "ten", "ter", "mc"}


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(?:dr|prof|mr|mrs|ms)\b\.?", " ", s.lower())
    return re.sub(r"[^a-z ]+", " ", s.replace("-", " ")).split()


def name_match(a: str, b: str) -> str:
    """How two printed names match: "same", "close" (middle names, initials,
    particles, a short form or a one-letter typo differ), or ""."""
    import difflib

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ""
    if ta == tb:
        return "same"
    if len(ta) < 2 or len(tb) < 2:
        return ""
    first_same = ta[0] == tb[0]
    last_same = ta[-1] == tb[-1] or "".join(ta[-2:]) == "".join(tb[-2:])
    if first_same and (last_same or set(ta) <= set(tb) or set(tb) <= set(ta)):
        return "close"
    if last_same and (set(ta) <= set(tb) or set(tb) <= set(ta)):
        return "close"              # "Margaret (Peggy) Smith" / "Peggy Smith"
    if last_same and ta[0][0] == tb[0][0] and (ta[0][:3] == tb[0][:3] or ta[0][-3:] == tb[0][-3:]):
        return "close"              # Steve / Stephen, Pippa / Philippa, Gabi / Gabriella
    if first_same and difflib.SequenceMatcher(None, ta[-1], tb[-1]).ratio() >= 0.8 and len(ta[-1]) > 4:
        return "close"              # Rawlins / Rowlins
    return ""


@dataclass
class Person:
    name: str
    first: str = ""
    last: str = ""
    countries: list[str] = field(default_factory=list)
    roles: list[tuple[str, int]] = field(default_factory=list)     # (cl, 1); co-chair: ("cs", 0)
    orcid: str = ""
    full_name: str = ""     # from the author annex, with middle names

    @property
    def slug(self) -> str:
        def clean(s):
            return re.sub(r"[^\w-]", "", s.replace(" ", ""))
        return f"{clean(self.last)}_{clean(self.first)}"


def _split_name(name: str, surnames: list[str]) -> tuple[str, str]:
    """(first, last). A surname from the chapter citation wins; otherwise the
    last word, with particles (de, van ...) kept with it."""
    toks = name.split()
    for s in sorted(surnames, key=len, reverse=True):
        if _tokens(name)[-len(_tokens(s)):] == _tokens(s) and len(_tokens(s)) < len(_tokens(name)):
            n = len(s.split())
            return " ".join(toks[:-n]), " ".join(toks[-n:])
    i = len(toks) - 1
    while i > 1 and toks[i - 1].lower() in _PARTICLES:
        i -= 1
    return " ".join(toks[:i]), " ".join(toks[i:])


def _citation_surnames(cite: str) -> list[str]:
    """'Rodriguez Osuna, V., Ingram J.C., ... (2026).' -> ['Rodriguez Osuna', 'Ingram', ...]"""
    authors = cite.split("(20", 1)[0]
    out = []
    for part in re.split(r"(?<=\.)\s*,\s*|(?<=\.)\s+(?=[A-Z][a-z])", authors):
        m = re.match(r"^\s*(.+?),?\s+(?:[A-Z]\.\s*-?)+[A-Z]?\.?\s*$", part.strip().rstrip(","))
        if m:
            out.append(m.group(1).strip(" ,"))
    return out


def read_annex_people(doc: pymupdf.Document, pages: list[int]) -> list[dict]:
    """The author annex (Annex III or IV): [{name, role, affiliation, countries, group}] (group: Co-chairs,
    Chapter 1 ...), as printed."""
    people, group, cur = [], "", None
    for pno in pages:
        page = doc[pno - 1]
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                spans = [_span(s) for s in l["spans"] if s["text"].strip()]
                if not spans or l["bbox"][1] < 62 or l["bbox"][1] > 790:
                    continue
                text = re.sub(r"\s+", " ", "".join(s.text for s in spans)).strip().lstrip("﻿")
                m = spans[0]
                if _near(m.size, 11.0) and "Hv" in m.font:
                    group, cur = text, None
                elif _near(m.size, 7.5) and m.bold:
                    cur = {"name": text, "lines": [], "group": group, "page": pno}
                    people.append(cur)
                elif cur is not None and _near(m.size, 7.5):
                    cur["lines"].append(text)
    return people


ANNEX_ROLES = {"coordinating lead author": "cl", "lead author": "la", "fellow": "fl",
               "contributing author": "ca", "review editor": "re", "chair": "cs", "co-chair": "cs"}


def _annex_role(lines: list[str]) -> tuple[str, int]:
    """The role among a person's lines, allowing a cut-off ('Lead Autho')."""
    best = ("", -1)
    for i, line in enumerate(lines):
        low = line.lower().strip()
        for text, code in ANNEX_ROLES.items():
            if low == text or (len(low) >= 6 and text.startswith(low) and len(low) >= len(text) - 2):
                if len(text) > len(best[0]):
                    best = (code, i)
                    return code, i
    return best


def _chapter_codes(cfg: dict, log: Log) -> set[str]:
    """The chapter codes of the assessment in ipbes_chapters.csv
    ('business_biodiversity_chapter1' ...), to check the experts list."""
    import csv

    if not cfg.get("chapters_csv"):
        return set()
    path = ROOT / cfg["chapters_csv"]
    if not path.exists():
        log.add("people", f"{path.name} not found; chapter codes not checked")
        return set()
    with path.open(encoding="utf-8-sig") as f:
        return {r["Name"].strip() for r in csv.DictReader(f) if r["Abbreviation"].strip() == cfg["code"]}


def build_people(rep: Report, doc: pymupdf.Document, cfg: dict, countries: Countries,
                 log: Log) -> list[Person]:
    import csv

    people: list[Person] = []

    def find(name: str, source: str = "") -> Person | None:
        found = [(p, name_match(p.name, name)) for p in people]
        for p, how in found:
            if how == "same":
                return p
        for p, how in found:
            if how == "close":
                if source:
                    log.add("names", f"'{p.name}' (chapter or title page) is '{name}' in {source}")
                return p
        return None

    # Chapter pages: the authors of each chapter, by role.
    for ch in rep.chapters:
        surnames = _citation_surnames(ch.cite_as)
        for role, entries in ch.authors.items():
            if role not in ROLE_NAMES:
                continue
            for entry in entries:
                m = re.match(r"^(.*)\s*\(([^()]*)\)\s*$", entry)      # the last (...) holds the countries
                name, ctext = (m.group(1), m.group(2)) if m else (entry, "")
                name = re.sub(r"\s+", " ", name).strip()
                # The same person may be printed differently in two chapters
                # ("Laura Pereira", "Laura M. Pereira"); reported as a variant.
                p = find(name, f"the chapter {ch.number} page")
                if p is None:
                    first, last = _split_name(name, surnames)
                    p = Person(name, first, last)
                    people.append(p)
                for c in countries.codes(ctext, f"chapter {ch.number} page, {name}"):
                    if c not in p.countries:
                        p.countries.append(c)
                if (role, ch.number) not in p.roles:
                    p.roles.append((role, ch.number))

    # Co-chairs: the title page.
    for name, aff in rep.editors:
        p = find(name)
        if p is None:
            first, last = _split_name(name, [])
            p = Person(name, first, last)
            people.append(p)
        country = aff.rsplit(",", 1)[-1] if "," in aff else ""
        for c in countries.codes(country, f"title page, {name}"):
            if c not in p.countries:
                p.countries.append(c)
        if ("cs", 0) not in p.roles:
            p.roles.append(("cs", 0))

    # The author annex: checks names, roles and countries; adds full names.
    # The annex listing authors (Annex III in BBA, IV in IAS).
    annex, pages = next(((k, v) for k, v in rep.annex_pages.items() if "LIST OF AUTHORS" in k), ("", []))
    annex = " ".join(annex.split()[:2]).title().replace("Iii", "III").replace("Iv", "IV") or "Annex"
    for a in read_annex_people(doc, pages):
        code, i = _annex_role(a["lines"])
        rest = [x for j, x in enumerate(a["lines"]) if j != i]
        # The country is the last line(s); the affiliation comes before.
        ctext, aff = "", rest
        for k in range(1, min(4, len(rest)) + 1):
            cand = " ".join(rest[-k:])
            if countries.resolves(cand):
                ctext, aff = cand, rest[:-k]
        chm = re.match(r"Chapter (\d+)", a["group"])
        chapter = int(chm.group(1)) if chm else 0
        if not chm and not a["group"].lower().startswith("co-chair"):
            continue            # the Secretariat, task forces, technical support unit
        where = f"{annex} p{a['page']}, {a['name']}"
        if not code:
            log.add("people", f"{where}: no role ({' | '.join(a['lines'])})")
        elif code == "la" and not any(r.lower() in ("lead author",) for r in a["lines"]):
            log.add("people", f"{where}: role printed as '{a['lines'][i]}'")
        p = find(a["name"], annex)
        if p is None:
            log.add("people", f"{where} ({ROLE_NAMES.get(code, '?')}, {a['group']}): not on the "
                              f"chapter pages; left out")
            continue
        p.full_name = a["name"]
        if code and chapter and (code, chapter) not in p.roles:
            log.add("people", f"{p.name}: {ROLE_NAMES.get(code)} of chapter {chapter} in {annex}, but "
                              + ", ".join(f"{ROLE_NAMES[r]} of chapter {c}" if c else ROLE_NAMES[r]
                                          for r, c in p.roles) + " on the chapter page")
        ac = countries.codes(ctext, where) if ctext else []
        if ac and not set(ac) <= set(p.countries):
            log.add("countries", f"{p.name}: {'/'.join(ac)} in {annex}, {'/'.join(p.countries) or 'none'} "
                                 f"on the chapter page")
        if code == "cs":
            for c in ac:
                if c not in p.countries:
                    p.countries.append(c)

    # The experts list: ORCID, and names and roles to compare.
    path = ROOT / cfg["experts_csv"] if cfg.get("experts_csv") else None
    if path and path.exists():
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
            log.add("people", f"experts list {path.name}: not UTF-8; read as Windows-1252")
        rows = list(csv.DictReader(text.splitlines(keepends=True)))
        # The shared list (ipbes_experts.csv) covers every assessment.
        if rows and "Assessment" in rows[0]:
            rows = [r for r in rows if r["Assessment"].strip() == cfg["code"]]
            if not rows:
                log.add("people", f"experts list {path.name}: no experts for {cfg['code']}")
        chapters = _chapter_codes(cfg, log)
        for row in rows:
            code = row["Chapter"].strip()
            if chapters and "chapter" in code.lower() and code not in chapters:
                log.add("people", f"experts list: chapter '{code}' ({row['First Name']} {row['Last name']}) "
                                  f"is not a chapter of {cfg['code']} in {Path(cfg['chapters_csv']).name}")
            full = f"{row['First Name'].strip()} {row['Last name'].strip()}"
            role = _CSV_ROLES.get(row["Role"].strip().lower())
            chm = re.search(r"chapter\s*(\d+)", row["Chapter"], re.IGNORECASE)
            p = find(full, "the experts list")
            if p is None:
                if role:
                    log.add("people", f"experts list: {full} ({row['Role']}, {row['Chapter']}) is not "
                                      f"on the chapter pages or the title page")
                continue
            orcid = row["ORCID Identifier"].strip()
            if re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid):
                p.orcid = orcid
            elif orcid:
                log.add("people", f"experts list: {full}: ORCID '{orcid}' is not valid; left out")
            # Names: the experts list splits first and last name; use it when it
            # is the same person and the printed name contains its last name.
            # The experts list splits first and last names: use the split when
            # the printed name ends with its last name.
            # Compare whole words: "Horcea-Milcu" and "O’Brien" are one word each.
            words, n = p.name.split(), len(row["Last name"].split())
            if 0 < n < len(words) and _tokens(" ".join(words[-n:])) == _tokens(row["Last name"]):
                p.first, p.last = " ".join(words[:-n]), " ".join(words[-n:])
            if role and role != "cs" and chm:
                if (role, int(chm.group(1))) not in p.roles:
                    log.add("people", f"{p.name}: {row['Role']} of chapter {chm.group(1)} in the experts list, "
                                      "but " + ", ".join(f"{ROLE_NAMES[r]} of chapter {c}" if c else ROLE_NAMES[r]
                                                         for r, c in p.roles) + " in the report")
    for p in people:
        if not p.countries:
            log.add("countries", f"{p.name}: no country")
        if " ".join(_tokens(p.last)) not in " ".join(_tokens(p.name)):
            log.add("names", f"{p.name}: last name '{p.last}' is not in the name")
    slugs = {}
    for p in people:
        if p.slug in slugs:
            log.add("people", f"{p.name} and {slugs[p.slug]} have the same URI person/{p.slug}")
        slugs[p.slug] = p.name
    return people


# ---------------------------------------------------------------------------
# Knowledge gaps (the SPM table)
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    number: int
    category: str
    text: str
    traceability: str


def gaps_from_box(rep: Report, log: Log) -> list[Gap]:
    """Knowledge gaps listed in an SPM box (TCA Box SPM 9): numbered items
    under sub-headings (the categories). An item is cut into one gap per
    {traceability} mark, as IBOK does; a mark opened with "(" by mistake is
    read too, and reported."""
    box = next((il for il in rep.illustrations if il.number.startswith("SPM") and il.kind == "Box"
                and "knowledge gap" in il.title.lower()), None)
    if box is None:
        log.add("knowledge gaps", "no knowledge-gap table or box in the SPM")
        return []
    gaps, category = [], ""
    for para in box.legend:
        if para.startswith("## "):
            category = para[3:].strip()
            continue
        m = re.match(r"^\d{1,2}\.\s+(.*)$", para)
        if not m:
            continue                # the box's introduction and conclusion
        text, pos = m.group(1), 0
        for mk in re.finditer(r"([{(])([^{}()]*)\}\s*\.?", text):
            if mk.group(1) == "(":
                log.add("knowledge gaps", f"Box {box.number}: '({mk.group(2)}}}' opened with '(' in the PDF")
            seg = text[pos:mk.start()].strip()
            if seg:
                gaps.append(Gap(len(gaps) + 1, category, seg.rstrip(" ."), mk.group(2).strip()))
            pos = mk.end()
        rest = text[pos:].strip()
        if rest and gaps:
            log.add("knowledge gaps", f"Box {box.number}: text after the last traceability mark of an item: {rest[:80]}")
    return gaps


def read_gaps(doc: pymupdf.Document, rep: Report, log: Log) -> list[Gap]:
    """The SPM table of knowledge gaps. Layouts differ (BBA: categories white
    on colour; IAS: dark bold, icons on the right), so the table is read from
    its geometry: the description column is where most text lines start;
    categories are the bold lines left of it, grouped when they touch; each
    gap ends with its {traceability}; a gap belongs to the category last
    printed above it (on this page or an earlier one)."""
    if not rep.gap_table_pages:
        return gaps_from_box(rep, log)
    import collections

    def page_lines(pno: int) -> list[tuple[float, float, str, Span]]:
        out = []
        for b in doc[pno - 1].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                spans = [_span(x) for x in l["spans"] if x["text"].strip()]
                if not spans or abs(l["dir"][0] - 1) > 0.01:
                    continue
                x0, y0 = l["bbox"][0], l["bbox"][1]
                if y0 < 62 or y0 > 790 or x0 < 50:
                    continue        # running header, page number, side tab
                text = re.sub(r"\s+", " ", "".join(x.text for x in spans)).strip()
                out.append((x0, y0, text, max(spans, key=lambda x: len(x.text))))
        return out

    start = rep.gap_table_pages[0]
    first = page_lines(start)
    xs = collections.Counter(round(x0) for x0, _, t, sp in first
                             if not sp.bold and sp.color != WHITE and _near(sp.size, 7.0, 0.3) and len(t) > 30)
    if not xs:
        log.add("knowledge gaps", f"p{start}: knowledge-gap table not understood")
        return []
    desc_x = xs.most_common(1)[0][0]

    gaps: list[Gap] = []
    category, cur = "", ""
    for pno in range(start, min(start + 6, doc.page_count + 1)):
        lines = page_lines(pno)
        desc = sorted((y0, t) for x0, y0, t, sp in lines if abs(x0 - desc_x) <= 4 and not sp.bold
                      and sp.color != WHITE and _near(sp.size, 7.0, 0.3))
        if not desc:
            break
        # Category cells: bold lines left of the descriptions, grouped by position.
        cells: list[list] = []
        for x0, y0, t, sp in sorted((ln for ln in lines if ln[0] < desc_x - 15 and ln[3].bold), key=lambda ln: ln[1]):
            if t.upper() in ("CATEGORY", "CATEGORIES OF KNOWLEDGE GAPS", "DESCRIPTION", "GAP"):
                continue
            if cells and y0 - cells[-1][1] < 12:
                cells[-1][1] = y0
                cells[-1][2] += " " + t
            else:
                cells.append([y0, y0, t])
        cats = [(y, re.sub(r"^\d+\.\s*", "", t).strip()) for y, _, t in cells]
        row_start = None
        for y0, t in desc:
            if not cur:
                row_start = y0
            cur = f"{cur} {t}".strip()
            m = re.search(r"\{([^{}]*)\}\s*\.?$", cur)
            if m:
                above = [c for y, c in cats if y <= (row_start or y0) + 2]
                if above:
                    category = above[-1]
                gaps.append(Gap(len(gaps) + 1, category, cur[:m.start()].strip(), m.group(1).strip()))
                cur = ""
    if cur:
        log.add("knowledge gaps", f"text after the last gap: {cur[:100]}")
    return gaps


# ---------------------------------------------------------------------------
# References: Zotero keys, and the sections that cite them
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # Zotero titles may hold <i>...</i>
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class Ref:
    chapter: int
    text: str
    page: int
    first_author: str = ""
    year: str = ""          # "2019a"
    doi: str = ""
    zotero: str = ""        # item key
    sections: set[str] = field(default_factory=set)


def _ref_fields(r: Ref) -> None:
    m = _REF_START.match(r.text)
    head = m.group(0) if m else r.text[:200]
    y = re.search(r"\(((?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)", head, re.IGNORECASE)
    r.year = y.group(1).replace("n.d.", "n.d").rstrip(".") if y else ""
    authors = head[:y.start()] if y else head
    first = re.split(r",|\.\s|\.$|\s\(", authors.strip(), 1)[0]
    r.first_author = _norm(first)
    r.doi = _doi_in(re.sub(r"(doi\.org/)\s+", r"\1", r.text))


def match_zotero(refs: list[Ref], items: list[dict], log: Log) -> None:
    by_doi: dict[str, list[dict]] = {}
    for it in items:
        if it["doi"]:
            by_doi.setdefault(it["doi"].lower(), []).append(it)
    titles = [(it, _norm(it["title"])) for it in items if len(_norm(it["title"])) >= 8]

    def pick(cands: list[dict], chapter: int) -> dict | None:
        if not cands:
            return None
        own = [c for c in cands if f"Chapter {chapter}" in c["collections"]]
        return (own or cands)[0]

    for r in refs:
        ntext = _norm(r.text)
        year = r.year[:4]
        cands = by_doi.get(r.doi.lower(), []) if r.doi else []
        how = "DOI"
        if not cands:
            how = "title"
            cands = [it for it, t in titles if t in ntext and (not year.isdigit() or year in it["date"])]
            if not cands:
                # Titles differ in small ways ("&" for "and", subtitles): compare the start.
                cands = [it for it, t in titles if len(t) > 25 and t[:40] in ntext
                         and (not year.isdigit() or year in it["date"])]
                how = "title start"
        # The first author must be among the item's creators (Zotero lists a
        # book's editors before a chapter's authors).
        if cands and r.first_author:
            agree = [c for c in cands if not c["creators"] or
                     any(_norm(x).split()[:1] == r.first_author.split()[:1] for x in c["creators"] if _norm(x))]
            if agree:
                cands = agree
            elif how != "DOI":
                cands = []
        it = pick(cands, r.chapter)
        if it is None:
            log.add("references", f"chapter {r.chapter} p{r.page}: no Zotero item for: {r.text[:160]}")
            continue
        r.zotero = it["key"]
        if not r.doi and it["doi"]:
            r.doi = it["doi"]
        if f"Chapter {r.chapter}" not in it["collections"]:
            log.add("references", f"chapter {r.chapter}: {it['key']} matched outside the chapter's "
                                  f"Zotero collection ({', '.join(it['collections'])}): {r.text[:100]}")


# In-text citations: "(Smith et al., 2020a; CBD, 2022)", "Smith and Jones (2019)".
_PAREN = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)*\b(?:1[89]\d\d|20\d\d|n\.d\.?)[a-z]?[^()]*)\)")
_NARRATIVE = re.compile(
    r"\b([A-Z][\w’'\-]+(?:\s(?:[A-Z][\w’'\-]+|de|van|von|der|du|da|zu))*)"
    r"(?:\s(?:et al\.|and|&)\s?(?:[A-Z][\w’'\-]+)?)?\s\(((?:1[89]\d\d|20\d\d|n\.d\.?)[a-z]?(?:,\s?(?:1[89]\d\d|20\d\d)[a-z]?)*)\)")


def citations(text: str) -> set[tuple[str, str]]:
    """{(first author, year)} cited in a text."""
    out = set()
    for m in _PAREN.finditer(text):
        for part in re.split(r";", m.group(1)):
            part = re.sub(r"^\s*(?:e\.g\.,?|see(?: also)?|i\.e\.,?|cf\.|for example,?|as cited in)\s*", "", part.strip())
            ym = list(re.finditer(r"\b((?:1[89]\d\d|20\d\d)[a-z]?|n\.d\.?[a-z]?)\b", part))
            if not ym:
                continue
            name = part[:ym[0].start()].strip(" ,")
            name = re.split(r"\set al\.|\s&\s|\sand\s|,", name)[0].strip()
            for y in ym:
                if name:
                    out.add((_norm(name), y.group(1).replace("n.d.", "n.d").rstrip(".")))
    for m in _NARRATIVE.finditer(text):
        for y in re.split(r",\s?", m.group(2)):
            out.add((_norm(m.group(1)), y.replace("n.d.", "n.d").rstrip(".")))
    return out


def _cites(r: Ref, found: set[tuple[str, str]]) -> bool:
    for name, year in found:
        if r.year != year or not r.first_author or not name:
            continue
        a, b = r.first_author, name
        if a == b or (len(a) >= 3 and (b.startswith(a + " ") or a.startswith(b + " "))) \
                or (len(b) >= 4 and a.split()[0] == b.split()[-1]):
            return True
    return False


def link_citations(rep: Report, refs: list[Ref], doc: pymupdf.Document, log: Log) -> None:
    """Link each reference to the sections (1.1, 1.2 ...) citing it: first
    from the sections' text, boxes and legends; then, for references cited
    only in tables or figures, from the text of the pages a section is
    printed on."""
    by_chapter: dict[int, list[Ref]] = {}
    for r in refs:
        by_chapter.setdefault(r.chapter, []).append(r)
    for ch in rep.chapters:
        chrefs = by_chapter.get(ch.number, [])
        texts = [(s.number, "\n".join(s.paras)) for s in ch.sections]
        texts += [(il.section, il.title + " " + " ".join(il.legend)) for il in rep.illustrations
                  if il.section and il.section.split(".")[0] == str(ch.number)]
        for number, text in texts:
            found = citations(text)
            for r in chrefs:
                if _cites(r, found):
                    r.sections.add(number)

        # Tables and figures: the pages each section is printed on.
        refs_start = min((r.page for r in chrefs), default=10 ** 6)
        spans = []
        for i, sec in enumerate(ch.sections):
            end = ch.sections[i + 1].page if i + 1 < len(ch.sections) else refs_start - 1
            spans.append((sec, sec.page, end))
        for pno in range(ch.sections[0].page if ch.sections else 0, refs_start):
            on_page = [sec for sec, a, b in spans if a <= pno <= b]
            if not on_page:
                continue
            if len(on_page) > 1:
                ills = {il.section for il in rep.illustrations if il.page == pno and il.section}
                on_page = [sec for sec in on_page if sec.number in ills] or on_page
            found = citations(doc[pno - 1].get_text().translate(_LIGATURES))
            for r in chrefs:
                if not r.sections and _cites(r, found):
                    for sec in on_page:
                        r.sections.add(sec.number)
        for r in chrefs:
            if not r.sections:
                log.add("citations", f"chapter {ch.number}: not cited in any section (perhaps only in the "
                                     f"executive summary): {r.text[:120]}")


# ---------------------------------------------------------------------------
# Glossary: the report's Annex I against the IPBES thesaurus
# ---------------------------------------------------------------------------

def check_glossary(paras: list[Para], rep: Report, log: Log, thesaurus: Path) -> tuple[int, int]:
    """Report the glossary terms of Annex I that the IPBES thesaurus does not
    have (as a preferred or alternative English label). (terms, found)"""
    from rdflib import Graph
    from rdflib.namespace import SKOS

    pages = next((v for k, v in rep.annex_pages.items() if "GLOSSARY" in k and "REFERENCES" not in k), [])
    terms = []
    for p in paras:
        if p.page in pages and p.role == "small":
            m = re.match(r"^([^:]{2,90}):\s", p.text)
            if m and not re.search(r"\(\d{4}", m.group(1)):
                terms.append(m.group(1).strip())
    if not terms or not thesaurus.exists():
        return len(terms), 0
    g = Graph()
    g.parse(thesaurus)
    labels = set()
    for prop in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        for _, lab in g.subject_objects(prop):
            if getattr(lab, "language", None) in (None, "en"):
                labels.add(_norm(str(lab)))

    def known(term: str) -> bool:
        variants = {term, re.sub(r"\s*\([^)]*\)", "", term), term.rstrip("s")}
        variants |= {v.split(" or ")[0] for v in variants} | {x.strip() for v in list(variants) for x in v.split("/")}
        return any(_norm(v) in labels for v in variants if v.strip())

    found = 0
    for term in terms:
        if known(term):
            found += 1
        else:
            log.add("glossary", f"'{term}' (Annex I) is not in the IPBES thesaurus")
    return len(terms), found


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

CONFIDENCE = {
    "well established": "hasWellestablished",
    "well-established": "hasWellestablished",
    "established but incomplete": "hasEstablishedIncomplete",
    "unresolved": "hasUnresolved",
    "inconclusive": "hasInconclusive",
}
_CONFIDENCE = re.compile(r"\((well[ -]established|established but incomplete|unresolved|inconclusive)\)",
                         re.IGNORECASE)
_ILL_MENTION = re.compile(r"\b(Figures?|Tables?|Box(?:es)?)\s+(SPM\s?\.?\s?[A-Z]?\d+|\d+\.\d+)", re.IGNORECASE)
_KIND = {"figure": "Figure", "figures": "Figure", "table": "Table", "tables": "Table", "box": "Box", "boxes": "Box"}


def _ill_key(kind: str, number: str) -> str:
    """('Boxes', 'SPM 1') -> 'BOXSPM.1', the key of il_uri."""
    key = (_KIND[kind.lower()] + re.sub(r"\s", "", number)).upper()
    return re.sub(r"SPM\.?", "SPM.", key)


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def resolve_marks(text: str, chapter: int, rep: Report, log: "Log | None" = None, where: str = "") -> str:
    """Superscript numbers were read as footnote marks [^n]. Those that are a
    footnote of the chapter (0: the SPM) stay marks, removed by clean(); the
    others are exponents ("km²") and are written as superscripts."""
    def sub(m):
        n = int(m.group(1))
        if (chapter, n) in rep.footnotes:
            return m.group(0)
        if log:
            log.add("footnotes", f"{where}: superscript {n} is not a footnote of "
                                 f"{'the SPM' if not chapter else 'chapter ' + str(chapter)}; kept as '{m.group(1).translate(_SUPERSCRIPT)}'")
        return m.group(1).translate(_SUPERSCRIPT)
    return FOOTNOTE_MARK.sub(sub, text)


def clean(text: str) -> str:
    """Text for the TTL: footnote marks out, spaces tidied."""
    text = FOOTNOTE_MARK.sub("", text)
    text = _CONTROL.sub("", text.replace("\u00ad", ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:)])", r"\1", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def split_submessages(text: str, where: str = "", log: "Log | None" = None) -> list[tuple[str, str]]:
    """A background message's text cut after each {traceability} mark:
    [(sentences, "2.2.1, 1.1.3"), ...]. A mark closed by ")" by mistake
    ("{5.6.3, 5.7).") is read too, and reported."""
    out, pos = [], 0
    for m in re.finditer(r"\{((?:[^{}()]|\(\d+\))*)([})])\s*\.?", text):
        if m.group(2) == ")" and log:
            log.add("traceability", f"{where}: '{{{m.group(1)})' closed with ')' in the PDF")
        seg = text[pos:m.end()].strip()
        if seg:
            out.append((seg, m.group(1).strip()))
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        if out and len(rest) < 40:
            out[-1] = (out[-1][0] + " " + rest, out[-1][1])
        else:
            out.append((rest, ""))
    return out


def write_ttl(rep: Report, cfg: dict, onto: dict, people: list[Person], gaps: list[Gap],
              refs: list[Ref], log: Log):
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import DCTERMS, FOAF, OWL, RDF, SKOS

    ns = onto["namespace"]
    base = onto["base"]
    rid = cfg["report_id"]
    g = Graph()
    g.bind("ipbes", Namespace(ns))
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("foaf", FOAF)
    g.bind("owl", OWL)

    def P(name: str) -> URIRef:
        return URIRef(ns + name)

    def add(s, p, value) -> None:
        """Leave out empty values (data review L1)."""
        if isinstance(value, str) and not isinstance(value, URIRef):
            value = clean(value)
            if not value:
                return
            value = Literal(value)
        g.add((s, p, value))

    report = URIRef(base + rid)
    ch_uri = {c.number: URIRef(f"{base}ch/{rid}ch{c.number}") for c in rep.chapters}
    sections = {s.number: s for c in rep.chapters for s in c.sections}
    sch_uri = {n: URIRef(f"{base}sch/{rid}-{n}") for n in sections}
    il_uri = {il.ident.upper(): URIRef(f"{base}il/{rid}-{il.ident}") for il in rep.illustrations}
    bgm_uri = {b.ident: URIRef(f"{base}bgm/{rid}-{b.ident}") for b in rep.background_messages}
    person_uri = {p.name: URIRef(f"{base}person/{p.slug}") for p in people}

    def section_links(trace: str, where: str) -> tuple[list[URIRef], list[URIRef]]:
        """'{2.2.1, table 2.2, 4.6}' -> sub-chapters (2.2, 4.6) and illustrations
        (Table2.2). In '{Boxes 5.2, 5.8}' the numbers after a plural are boxes."""
        subs, ills = [], []
        parts = []
        for part in re.split(r"\s*[,;]\s*", trace):
            nums = part.split()
            if len(nums) > 1 and all(re.fullmatch(r"\d{1,2}(?:\.\d+)+", n) for n in nums):
                log.add("traceability", f"{where}: '{part}' has no comma between sections in the PDF")
                parts += nums
            else:
                parts.append(part)
        plural = None
        for part in parts:
            part = part.strip().rstrip(".")
            m = _ILL_MENTION.match(part)
            if m:
                plural = m.group(1) if m.group(1).lower().endswith("s") and m.group(1).lower() != "box" else None
                key = _ill_key(m.group(1), m.group(2))
                if key in il_uri:
                    ills.append(il_uri[key])
                else:
                    log.add("traceability", f"{where}: '{part}' is not a figure, table or box of the report")
                continue
            if plural and re.fullmatch(r"\d{1,2}\.\d{1,2}", part):
                key = _ill_key(plural, part)
                if key in il_uri:
                    ills.append(il_uri[key])
                    continue
            plural = None
            # "5.2.2.1.a": a lettered paragraph of section 5.2.
            m = re.match(r"^(\d{1,2}\.\d{1,2})(?:\.\d+)*(?:\.?[a-z]|\s?\(\d+\))?$", part)
            if m:
                if m.group(1) in sch_uri:
                    if sch_uri[m.group(1)] not in subs:
                        subs.append(sch_uri[m.group(1)])
                else:
                    log.add("traceability", f"{where}: section '{part}' does not exist")
            elif part and not re.fullmatch(r"[A-F]\d{1,2}|KM-?[A-F]?\d{1,2}", part):
                log.add("traceability", f"{where}: '{part}' not understood")
        return subs, ills

    def mentions(text: str) -> list[URIRef]:
        out = []
        for m in _ILL_MENTION.finditer(text):
            key = _ill_key(m.group(1), m.group(2))
            if key in il_uri and il_uri[key] not in out:
                out.append(il_uri[key])
        return out

    def confidence(s, text: str) -> None:
        for term in {m.group(1).lower() for m in _CONFIDENCE.finditer(text)}:
            add(s, P(CONFIDENCE[term]), term.replace("well-established", "well established"))

    # Report
    add(report, RDF.type, P("Report"))
    add(report, SKOS.prefLabel, rep.title)
    add(report, SKOS.altLabel, cfg.get("alt_label", ""))
    add(report, P("hasDoi"), rep.doi.replace("https://doi.org/", ""))
    add(report, P("year"), str(cfg.get("year", "")))

    # Chapters
    for c in rep.chapters:
        s = ch_uri[c.number]
        add(s, RDF.type, P("Chapter"))
        add(s, P("Report"), report)
        add(s, DCTERMS.identifier, f"{c.number:02d}")
        add(s, SKOS.prefLabel, c.title)
        add(s, P("hasDoi"), c.doi.replace("https://doi.org/", ""))

    # Sub-chapters
    for c in rep.chapters:
        for sec in c.sections:
            s = sch_uri[sec.number]
            add(s, RDF.type, P("SubChapter"))
            add(s, P("Report"), report)
            add(s, P("Chapter"), ch_uri[c.number])
            add(s, DCTERMS.identifier, sec.number)
            add(s, SKOS.prefLabel, sec.title)
            body = "\n\n".join(p[3:] if p.startswith("## ") else p for p in sec.paras)
            body = resolve_marks(body, c.number, rep, log, f"section {sec.number}")
            notes = []
            for n in dict.fromkeys(int(x) for x in FOOTNOTE_MARK.findall(body)):
                notes.append(f"{n}. {rep.footnotes[(c.number, n)].text}")
            if notes:
                body += "\n\nNotes\n" + "\n".join(notes)
            add(s, P("hasDescription"), body)
        for u in c.unnumbered:
            log.add("structure", f"chapter {c.number}: unnumbered section '{u.title}' (p{u.page}) is not a "
                                 f"sub-chapter; not in the TTL")
        if c.executive_summary:
            log.add("structure", f"chapter {c.number}: executive summary ({len(c.executive_summary)} findings) "
                                 "not in the TTL: the ontology has no place for it")

    # Illustrations
    for il in rep.illustrations:
        s = il_uri[il.ident.upper()]
        add(s, RDF.type, P("Illustration"))
        ch = 0 if il.number.startswith("SPM") else int(il.number.split(".")[0])
        add(s, DCTERMS.identifier, il.ident)
        add(s, SKOS.prefLabel, resolve_marks(il.title, ch, rep))
        add(s, P("hasDescription"), resolve_marks("\n\n".join(x[3:] if x.startswith("## ") else x
                                                                for x in il.legend), ch, rep))

    # Background messages and their sub-messages
    km_of_sch: dict[URIRef, set[URIRef]] = {}
    bgm_sch: dict[str, set[URIRef]] = {}
    for b in rep.background_messages:
        s = bgm_uri[b.ident]
        add(s, RDF.type, P("BackgroundMessage"))
        add(s, P("Report"), report)
        add(s, DCTERMS.identifier, b.ident)
        add(s, SKOS.prefLabel, resolve_marks(b.headline, 0, rep))
        text = resolve_marks("\n\n".join(b.paras), 0, rep)
        add(s, P("hasDescription"), text)
        confidence(s, b.headline)
        for il in mentions(b.headline + " " + text):
            add(s, P("Illustration"), il)
        bgm_sch[b.ident] = set()
        flat = clean(resolve_marks(" ".join(b.paras), 0, rep))
        segments = split_submessages(flat, b.ident, log)
        if len(segments) == 1 and not segments[0][1]:
            log.add("messages", f"{b.ident}: no traceability marks in its text; one sub-message without sections")
        for i, (seg, trace) in enumerate(segments):
            sm = URIRef(f"{base}subm/{rid}-{b.ident}-SM{i}")
            add(sm, RDF.type, P("SubMessage"))
            add(s, P("SubMessage"), sm)
            add(sm, P("hasDescription"), seg)
            add(sm, DCTERMS.identifier, trace)
            confidence(sm, seg)
            subs, ills = section_links(trace, f"{b.ident}-SM{i}")
            for u in subs:
                add(sm, P("SubChapter"), u)
                bgm_sch[b.ident].add(u)
            for u in dict.fromkeys(ills + mentions(seg)):
                add(sm, P("Illustration"), u)
            if not trace and len(segments) > 1:
                log.add("messages", f"{b.ident}: text after the last traceability mark became "
                                    f"sub-message SM{i} without sections: {seg[:80]}")

    # Key messages
    for k in rep.key_messages:
        s = URIRef(f"{base}key/{rid}-{k.ident}")
        add(s, RDF.type, P("KeyMessage"))
        add(s, P("Report"), report)
        add(s, DCTERMS.identifier, k.ident)
        add(s, SKOS.prefLabel, resolve_marks(k.headline, 0, rep))
        text = resolve_marks("\n\n".join(k.paras), 0, rep)
        add(s, P("hasDescription"), text)
        confidence(s, k.headline)
        # Background messages in braces, at the end (BBA) or anywhere (IAS).
        idents = [i for g_ in re.findall(r"\{([^{}]*)\}", k.headline) for i in re.split(r"\s*,\s*", g_)
                  if re.fullmatch(r"[A-F]\d{1,2}", i.strip())]
        m = bool(idents)
        for ident in idents:
            if ident in bgm_uri:
                add(s, P("BackgroundMessage"), bgm_uri[ident])
                for u in bgm_sch.get(ident, ()):
                    km_of_sch.setdefault(u, set()).add(s)
            else:
                log.add("messages", f"{k.ident}: background message '{ident}' does not exist")
        if not m:
            log.add("messages", f"{k.ident}: no background messages in its headline")
        for il in mentions(k.headline + " " + text):
            add(s, P("Illustration"), il)
    for sch, kms in km_of_sch.items():
        for km in kms:
            add(sch, P("KeyMessage"), km)

    # Knowledge gaps
    for gap in gaps:
        s = URIRef(f"{base}kg/{rid}-{gap.number}")
        add(s, RDF.type, P("KnowledgeGap"))
        add(s, P("Report"), report)
        add(s, P("hasDescription"), gap.text)
        add(s, P("hasCategory"), gap.category)
        add(s, P("traceability"), gap.traceability)
        for part in re.split(r"\s*,\s*", gap.traceability):
            add(s, DCTERMS.identifier, part)
        subs, ills = section_links(gap.traceability, f"knowledge gap {gap.number}")
        for u in subs:
            add(s, P("SubChapter"), u)

    # People
    for p in people:
        s = person_uri[p.name]
        add(s, RDF.type, FOAF.Person)
        add(s, SKOS.prefLabel, p.name)
        add(s, FOAF.firstName, p.first)
        add(s, FOAF.lastName, p.last)
        add(s, P("Report"), report)
        for c in p.countries:
            add(s, P("country"), c)
        for role, chapter in p.roles:
            if chapter:
                add(s, P(role), f"{ROLE_NAMES[role]} in {rid}-ch{chapter}")
                add(s, P("Chapter"), ch_uri[chapter])
                add(ch_uri[chapter], FOAF.Person, s)
            else:
                add(s, P(role), f"{ROLE_NAMES[role]} in {rid}")
                add(report, FOAF.Person, s)
        if p.orcid:
            add(s, OWL.sameAs, URIRef(f"https://orcid.org/{p.orcid}"))

    # References: one per Zotero item, with every chapter and section citing it.
    group = cfg.get("zotero_group")
    seen: dict[tuple[str, int], str] = {}
    for i, r in enumerate(refs):
        if r.zotero:
            s = URIRef(f"{base}ref/{group}/items/{r.zotero}")
            # The same work in two chapters (often with another year letter) is
            # one reference; twice in one chapter's list is worth knowing.
            if (r.zotero, r.chapter) in seen:
                log.add("references", f"chapter {r.chapter}: one Zotero item ({r.zotero}) for two entries: "
                                      f"'{seen[(r.zotero, r.chapter)][:80]}' and '{r.text[:80]}'")
            seen[(r.zotero, r.chapter)] = r.text
        else:
            s = URIRef(f"{base}ref/{rid}-ch{r.chapter}-{i + 1}")
        if (s, RDF.type, P("Reference")) not in g:
            add(s, RDF.type, P("Reference"))
            add(s, P("Report"), report)
            add(s, P("hasDescription"), r.text)
            add(s, P("hasDoi"), r.doi)
            if r.zotero:
                add(s, OWL.sameAs, URIRef(f"https://www.zotero.org/groups/{group}/items/{r.zotero}"))
        add(s, P("Chapter"), ch_uri[r.chapter])
        for n in sorted(r.sections):
            add(s, P("SubChapter"), sch_uri[n])
            add(sch_uri[n], P("Reference"), s)
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path: Path, assessment: str) -> tuple[dict, dict]:
    with path.open("rb") as f:
        conf = tomllib.load(f)
    try:
        cfg = dict(conf["assessments"][assessment])
    except KeyError:
        raise SystemExit(f"{assessment}: not in {path} (have: {', '.join(conf.get('assessments', {}))})")
    # The shared experts and chapters lists, unless the assessment names its own.
    shared = conf.get("experts", {})
    cfg.setdefault("experts_csv", shared.get("csv", ""))
    cfg.setdefault("chapters_csv", shared.get("chapters_csv", ""))
    cfg.setdefault("code", assessment)
    return cfg, conf["ontology"]


def write_log(log: Log, rep: Report, counts: dict, path: Path, name: str) -> None:
    out = [f"# {name}: conversion log", "",
           "What the conversion could not read or match, and where the sources disagree.",
           "Each line names its source; nothing here was guessed silently.", "",
           "| Item | Count |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in counts.items()] + [""]
    topics = {
        "people": "People: roles and presence across the chapter pages, the author annex and the experts list",
        "names": "Names: the same person written differently",
        "countries": "Countries",
        "references": "References and Zotero",
        "citations": "References not found cited in a section",
        "traceability": "Traceability marks",
        "messages": "Messages",
        "knowledge gaps": "Knowledge gaps",
        "footnotes": "Footnotes",
        "glossary": "Glossary terms (Annex I) not in the IPBES thesaurus (ipbes-thesaurus.rdf)",
        "structure": "Structure",
    }
    for key, title in topics.items():
        items = log.items.get(key, [])
        if items:
            out += [f"## {title} ({len(items)})", ""] + [f"- {x}" for x in items] + [""]
    if rep.problems:
        out += ["## Reading the PDF", ""] + [f"- {p}" for p in rep.problems] + [""]
    path.write_text("\n".join(out), encoding="utf-8")


def convert(assessment: str, config: Path = DEFAULT_CONFIG, out: Path | None = None,
            refresh_zotero: bool = False) -> dict:
    cfg, onto = load_config(config, assessment)
    out = out or DEFAULT_OUT / assessment
    pdf = ROOT / cfg["pdf"]
    doc = pymupdf.open(pdf)
    paras = build_paras(read_lines(doc))
    rep = parse(doc, paras)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{assessment}.md").write_text(to_markdown(rep, pdf.name), encoding="utf-8")

    log = Log()
    countries = Countries(log)
    people = build_people(rep, doc, cfg, countries, log)
    gaps = read_gaps(doc, rep, log)
    refs = [Ref(c.number, r.text, r.page) for c in rep.chapters for r in c.references]
    for r in refs:
        _ref_fields(r)
    if cfg.get("zotero_group"):
        items = zotero_items(cfg["zotero_group"], out / f"{assessment}_zotero.json", refresh_zotero)
        match_zotero(refs, items, log)
    link_citations(rep, refs, doc, log)
    n_terms, n_known = check_glossary(paras, rep, log, ROOT / "data" / "rdf" / "ipbes-thesaurus.rdf")
    g = write_ttl(rep, cfg, onto, people, gaps, refs, log)
    ttl = out / f"{assessment}_pdf.ttl"
    g.serialize(ttl, format="turtle", encoding="utf-8")

    counts = {
        "Chapters": len(rep.chapters),
        "Sub-chapters": sum(len(c.sections) for c in rep.chapters),
        "Key messages": len(rep.key_messages),
        "Background messages": len(rep.background_messages),
        "Figures, tables and boxes": len(rep.illustrations),
        "Knowledge gaps": len(gaps),
        "People": len(people),
        "Reference entries in the PDF": len(refs),
        "  matched to Zotero": sum(1 for r in refs if r.zotero),
        "  linked to a sub-chapter": sum(1 for r in refs if r.sections),
        "Glossary terms (Annex I)": n_terms,
        "  in the IPBES thesaurus": n_known,
        "Triples": len(g),
    }
    write_log(log, rep, counts, out / f"{assessment}_conversion.md", assessment)
    return {"ttl": ttl, "counts": counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("assessment", help="as in config/pdf_to_ttl.toml, e.g. BBA")
    ap.add_argument("--out", type=Path, help="output folder (default: data/pdf_compare/<assessment>)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--refresh-zotero", action="store_true", help="download the Zotero library again")
    args = ap.parse_args(argv)
    result = convert(args.assessment, args.config, args.out, args.refresh_zotero)
    ttl = result["ttl"]
    print(f"{args.assessment} -> {ttl.relative_to(ROOT) if ttl.is_relative_to(ROOT) else ttl}")
    for k, v in result["counts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
