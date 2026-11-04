"""Convert an IPBES assessment report (PDF) to RDF/Turtle following the IPBES
report ontology (docs/ontology/IPBES_Ontology_v07.ttl), in the form of the
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
  * Report: title and DOI (citation on page 2; the config's DOI wins, and is the
    reference), management committee (page 2), co-chairs and technical support
    unit (the editors, page 3).
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
def grey(color: int) -> bool:
    """The mid-grey of footnotes and running headers (0x58595B, 0x575756 ...)."""
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return all(0x48 <= c <= 0x70 for c in (r, g, b)) and max(r, g, b) - min(r, g, b) < 0x10
WHITE = 0xFFFFFF
def dark(color: int) -> bool:
    """Black or near-black text (0x231F20, 0x3C3C3B ...), not the mid-grey
    of footnotes and running headers (0x58595B)."""
    return all(((color >> shift) & 0xFF) < 0x48 for shift in (16, 8, 0))
def neutral(color: int) -> bool:
    """A grey from dark (0x303030) to mid (0x707070): footnotes in ECA."""
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return all(0x30 <= c <= 0x70 for c in (r, g, b)) and max(r, g, b) - min(r, g, b) < 0x10


def coloured(color: int) -> bool:
    """A chapter's colour (headings in the reports of 2016-2019): neither
    white, grey nor black."""
    return color != WHITE and not grey(color) and not dark(color)


def _spaced_number(text: str) -> str:
    """ "1. 3. 4 Title" -> "1.3.4 Title": section numbers printed with spaces
    (AFA, SUA chapter 4)."""
    return re.sub(r"^\d{1,2}(?:\.\s?\d{1,2}){1,4}(?=\.?\s)", lambda m: m.group(0).replace(" ", ""),
                  text.strip())

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


# A real link, for footnotes in a colour (IAS): not any "a.b", which figure
# labels such as "1.3 Base conservation on diverse values" (TCA) also are.
_FOOTNOTE_LINK = re.compile(r"https?://|www\.|doi\.org|\b[\w-]+\.(?:org|com|int|net|edu|gov)\b", re.IGNORECASE)


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
    if line.y0 < 62 and m.size <= 7.6 and (grey(m.color) or dark(m.color)):
        return "header"
    if line.y0 > 790 or line.x0 < 45:
        return "folio"
    if m.size >= 20:
        return "title"
    if m.color == WHITE:
        return "other"
    if CAPTION_START.match(line.text) and (
            any(_near(s.size, 10.0) and dark(s.color) and (s.bold or "-Md" in s.font) for s in words)
            # a small label, then the title in 10 pt of any weight (ECA Table SPM 1)
            or (words[0].size <= 8.2 and any(9.5 <= s.size <= 10.6 and dark(s.color) for s in words))):
        return "caption"        # title in medium weight, or a plain note after it (SUA Box 1.1, 4.20)
    # Reports of 2016-2019 (LDR, PPA, SCM, the regional assessments, GA1)
    # print headings larger (17.5-18 pt and 12.7-13 pt), each chapter in its
    # own colour, and the levels below in black (10.7-11 pt, 10 pt).
    accent = m.color == ACCENT or coloured(m.color)
    if (_near(m.size, 15.6) or 17.2 <= m.size <= 18.7) and accent:
        return "h2"
    if 12.4 <= m.size <= 13.2 and accent:
        return "h3"
    if _near(m.size, 10.7) and "Hv" in m.font:
        return "h4"
    heavy = any(k in m.font for k in ("Hv", "Blk", "Black")) or m.bold
    # The deeper levels ("h4b") are only marked in the text: they can sit in
    # one column while a box goes on in the other (IAS Box 6.18), so unlike
    # the other headings they do not end the box.
    if 10.4 <= m.size <= 11.2 and heavy and dark(m.color) and SECTION_NUMBER.match(_spaced_number(line.text)):
        return "h4b"
    if _near(m.size, 10.0) and m.bold and dark(m.color) \
            and re.match(r"^\d{1,2}(?:\.\s?\d{1,2}){3,4}\.?\s", line.text):
        return "h4b"            # 1.3.4.1.1 (AFA, PPA); not a caption
    if _near(m.size, 10.0) and m.bold and dark(m.color):
        return "caption"
    if _near(m.size, 9.0) and m.bold and "DIN" in m.font and dark(m.color):
        return "h5"             # a heading inside a box (TCA Box SPM 9)
    if _near(m.size, 8.5, 0.3) and (dark(m.color) or m.color == ACCENT):
        return "body"
    if _near(m.size, 8.5, 0.3) and coloured(m.color) and heavy and line.x1 - line.x0 > 150:
        return "body"           # a background message's headline in colour (LDR); not a figure label
    if _near(m.size, 7.5) and (dark(m.color) or m.color == ACCENT) and "Roman" not in m.font:
        return "small"
    if _near(m.size, 7.8, 0.15) and (dark(m.color) or grey(m.color)) and "Roman" not in m.font:
        return "small"          # reference lists and captions in PPA (grey-green 0x5D6155)
    if 6.2 <= m.size <= 6.95 and (grey(m.color) or (not dark(m.color) and m.color != WHITE and (
            re.match(r"^\s*\d{1,3}\.\s", line.text) or _FOOTNOTE_LINK.search(line.text)))):
        return "footnote"       # grey, or a numbered or linked footnote in a colour (IAS)
    numbered = re.match(r"^\s*\d{1,3}\.?\s", line.text)
    if 6.2 <= m.size <= 6.95 and numbered and words and grey(words[0].color):
        return "footnote"       # the number grey, the text black (APA: "2. www.worldvaluessurvey.org")
    if 6.2 <= m.size <= 6.95 and neutral(m.color) and numbered and line.y0 > 500:
        return "footnote"       # numbered, in a darker grey (0x3C3C3B, ECA chapter 2)
    # (mid-page too when it holds a link: footnotes at the foot of a column, PPA chapter 5)
    if 6.95 < m.size <= 7.1 and grey(m.color) and (line.y0 > 400 or _FOOTNOTE_LINK.search(line.text)) \
            and re.match(r"^\s*\d{1,3}\.?\s", line.text):
        return "footnote"       # 7 pt at the foot of the page (PPA, 0x706F6F), "15  http://..." without a full stop
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
                            and s["color"] != WHITE and not grey(s["color"]) and not dark(s["color"]):
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
                prev = lines[-1] if lines else None
                if prev and prev.role == "h4b" and line.role != "h4b" and (prev.page, prev.block) == (pno, bno) \
                        and _near(line.main.size, prev.main.size) and line.main.font == prev.main.font:
                    line.role = "h4b"       # the heading's second line
                if prev and prev.role == "body" and line.role == "other" and prev.page == pno \
                        and line.main.color == prev.main.color and _near(line.main.size, prev.main.size) \
                        and line.main.bold == prev.main.bold and 0 <= line.y0 - prev.y1 < 6:
                    line.role = "body"      # a coloured headline's short last line (ECA C4: "incomplete) {4.3.1 ...}.")
                # (in any colour: its link may be in the link colour, TCA "zenodo.14242913).")
                if prev and prev.role == "footnote" and line.role != "footnote" \
                        and (prev.page, prev.block) == (pno, bno) and _near(line.main.size, prev.main.size) \
                        and not re.match(r"^\s*(?:\*|\d{1,3}\.\s)", line.text):
                    line.role = "footnote"  # a footnote's next line (its link, ECA)
                lines.append(line)
    _join_caption_labels(lines)
    return lines


_CAPTION_LABEL = re.compile(r"(Figure|Table|Box)(\s+SPM)?(?:\s*([A-Z]?\d{1,2}(?:\s*[.\s]\s*\d{1,2})?))?")


def _join_caption_labels(lines: list[Line]) -> None:
    """Captions printed as a small label, "Figure SPM" (7.5 pt), a number in a
    circle, and the title in 10 pt to their right, in separate lines or text
    blocks (ECA): joined into one caption line, "Figure SPM 4 Difference
    between ..."; the number's and the title's own lines are dropped."""
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)
    for page_lines in by_page.values():
        for label in page_lines:
            m = _CAPTION_LABEL.fullmatch(label.text.strip())
            if not m or label.main.size > 8.2 or not label.horizontal or label.role == "caption":
                continue
            number, number_line = m.group(3), None
            if not number:
                number_line = next((x for x in page_lines if x is not label and re.fullmatch(r"\d{1,2}", x.text.strip())
                                    and abs(x.y0 - label.y0) < 4 and -2 <= x.x0 - label.x1 < 20), None)
                number = number_line.text.strip() if number_line else ""
            if not number:
                continue
            # The title: 10 pt lines to the right, from just above the label down.
            right = sorted((x for x in page_lines if x is not label and x is not number_line
                            and 9.5 <= x.main.size <= 10.6 and x.x0 > label.x1 - 2 and x.x0 - label.x1 < 60
                            and label.y0 - 12 <= x.y0 <= label.y0 + 80 and x.horizontal), key=lambda x: x.y0)
            title = []
            for x in right:
                if title and x.y0 - title[-1].y0 > 16:
                    break               # the next paragraph
                title.append(x)
            if not title:
                continue
            kind = m.group(1)
            number = re.sub(r"\s+", " ", number)
            head = f"{kind} {'SPM ' if m.group(2) else ''}{number} "
            font = title[0].main.font
            label.spans = [Span(head, font, 10.0, title[0].main.color, True, False, False)] + \
                [s_ for x in title for s_ in (x.spans + [Span(" ", font, 10.0, title[0].main.color,
                                                               False, False, False)])]
            label.x1 = max(x.x1 for x in title)
            label.y1 = max(x.y1 for x in title)
            label.role = "caption"
            for x in title + ([number_line] if number_line else []):
                x.role = "absorbed"


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


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


# What comes before an exponent, not a footnote mark: a unit (km², m³, ha⁻¹,
# yr⁻¹, day⁻¹, t CO2 ...) or a power of ten (10⁶).
_BEFORE_EXPONENT = re.compile(
    r"(?:(?:\b|(?<=\d))(?:km|m|cm|mm|µm|ha|kg|g|Mg|Gg|Tg|Pg|t|Mt|Gt|kt|yr|year|day|d|h|s|l|L|kWh|MWh|GWh|J|W|ppm|"
    r"CO2e?|CO₂e?|CH4|N2O|C|N|P|Gha|people/km|R|r)|\b10|\bx ?10|×10|× 10)\s?$")      # R² (PPA), r² (SCM)


def _span_text(s: Span, role: str, before: str = "") -> str:
    t = s.text
    if s.sup and role in ("body", "small", "caption") and re.fullmatch(r"\s*[\d,–-]+\s*", t):
        # An exponent after a unit or "10": "km²", "ha⁻¹" (its minus kept), "10⁶".
        if _BEFORE_EXPONENT.search(before) or re.match(r"\s*[–-]\s*\d", t):
            return re.sub(r"[–-]", "⁻", t.strip()).translate(_SUPERSCRIPT)
        # Footnote marks: superscript numbers in running text.
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
    """A link broken at the end of a line: in the accent colour. Not a line of
    words in that colour (SUA's key messages), which has spaces inside and
    ends in one."""
    t = s.text.strip()
    words = bool(re.search(r"\s", t)) and s.text != s.text.rstrip() \
        and not re.search(r"per cent\d|https?://|www\.|/", t)     # "%20" printed "per cent20" (NXS)
    return s.color == ACCENT and not s.bold and not words


def _accent(s: Span) -> bool:
    """In the link colour: after a link's first part, its continuation, whatever
    it holds ("FABLEDietBrief_cor per cent20 ...", NXS)."""
    return s.color == ACCENT and not s.bold


def build_paras(lines: list[Line]) -> list[Para]:
    paras: list[Para] = []
    cur: Para | None = None
    fn_number = 0           # the number of the footnote being read
    for line in lines:
        if line.role in ("header", "folio", "absorbed"):
            continue
        same = (cur is not None and cur.page == line.page and cur.lines[-1].block == line.block
                and cur.role == line.role)
        # A footnote starts with its number: "7.\t This categorization...", in a
        # hanging indent, or not indented at all (VA): then it is the next number.
        mnum = re.match(r"^\s*(\d{1,3})\.?\s", line.text) if line.role == "footnote" else None
        if same and mnum and (line.x0 < cur.lines[-1].x0 - 3 or int(mnum.group(1)) == fn_number + 1):
            same = False
        if mnum and (not same or not cur):
            fn_number = int(mnum.group(1))
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
            # The text is a body paragraph, or a background message's headline
            # in 10 pt (SUA).
            target = next((q for q in paras if q.page == p.page and q.role in ("body", "caption")
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
            if prev and prev.role == "caption" and prev.page == p.page and not p.label:
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
            t = _span_text(s, p.role, (text + " " + line_text)[-40:])
            if lead_end is None and t.strip() and not s.bold:
                lead_end = len(_join(text, line_text, prev_link, _accent(parts[0]))) if text else len(line_text)
            line_text += t
        text = _join(text, line_text, prev_link, _accent(parts[0])) if text else line_text
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
    heading_pages: dict[str, int] = field(default_factory=dict)     # "1.1.1" -> page of its heading

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
    group: str = ""     # its section: "A. Understanding the relationship ..."
    label_printed: bool = False     # its number was printed (not counted, as in PPA)

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
    parted: bool = False    # in parts 5.0, 5.1 ... with their own sections 5.1.1 (NXS chapter 5)
    section_words: bool = False     # "Section 1. ...", its sub-sections 1.1 ... (PPA chapter 4)
    part: str = ""          # GA1: the part being read, "2.1"; its sections 2.1.1 ... are the sub-chapters
    part_titles: list[str] = field(default_factory=list)
    printed_title: str = ""         # the title on the chapter's first page, as printed (often in capitals)


@dataclass
class Report:
    title: str = ""
    doi: str = ""
    editors: list[tuple[str, str, str]] = field(default_factory=list)  # (name, "affiliation, country", role)
    editor_surnames: list[str] = field(default_factory=list)      # from the suggested citation
    committee: list[str] = field(default_factory=list)            # the management committee, page 2
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


# A caption's label alone, in white (PPA): "FIGURE 1.1", "TABLE SPM. 2".
_WHITE_CAPTION = re.compile(r"(FIGURE|TABLE|BOX)\s+(SPM\.?\s?\d{1,2}|\d{1,2}\s?\.\s?\d{1,2})")


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
    # A traceability mark ("{3.3.4.1}. Logging ...") never starts a paragraph.
    return first.islower() or first == "{" or not prev.rstrip().endswith(_TERMINAL) or prev.rstrip().endswith(",")


def _add_para(paras: list[str], text: str) -> None:
    if paras and not paras[-1].startswith("## ") and _continues(paras[-1], text):
        paras[-1] = _join(paras[-1], text, False, False) if paras[-1].endswith("-") else paras[-1] + " " + text
    else:
        paras.append(text)


def parse(doc: pymupdf.Document, paras: list[Para]) -> Report:
    rep = Report()
    _front_matter(doc, rep)
    casing = word_casing(paras)
    for level, title, page in doc.get_toc():
        m = re.match(r"^\s*(\d{1,2}\.\d{1,2})\.?\s+(.+)$", re.sub(r"\s+", " ", title))
        if m:
            rep.outline[m.group(1)] = m.group(2).strip()

    # The chapters' reference lists, from the outline: LDR prints no title over them.
    ref_pages = {page for _, title, page in doc.get_toc() if title.strip().upper() == "REFERENCES"}

    mode = "front"          # front, km, bgm, appendix, chapter, exec, refs, annex
    chapter: Chapter | None = None
    section: Section | None = None
    message: Message | None = None
    group = ""
    box: Illustration | None = None     # the box being read (its text is 7.5 pt)
    legend_of: Illustration | None = None   # the figure or table whose legend may follow
    legend_end = (0, 0.0)               # page and bottom of its caption or legend so far
    unnumbered: Section | None = None
    km_heading = ""                     # SUA: the heading of a group of key messages (A, B ...)
    group_km: Message | None = None     # SUA: the key message whose intro may follow its title

    def footnote_chapter() -> int:
        return chapter.number if chapter else 0

    skip: set[int] = set()      # paragraphs already read with the one before
    last_heading = (0, 0.0)     # page and bottom of the last section heading

    for i, p in enumerate(paras):
        if id(p) in skip:
            continue
        text = p.text

        # PPA: "FIGURE 1.1" in white on a colour, its title in the line under it.
        mc = _WHITE_CAPTION.fullmatch(text.strip()) if p.role == "other" and _white(p) else None
        first_line = p.lines[0]
        ml = _WHITE_CAPTION.fullmatch(first_line.text.strip()) if p.role == "other" and len(p.lines) > 1 \
            and all(s_.color == WHITE for s_ in first_line.spans if s_.text.strip()) else None
        if ml:
            # The label and its title read as one paragraph.
            number = re.sub(r"SPM\.\s*", "SPM ", ml.group(2))
            rest = " ".join(line.text.strip() for line in p.lines[1:])
            text = f"{ml.group(1).title()} {number} {rest}"
            p.role = "caption"
        elif mc:
            # The title: the paragraph just under the label, at its left edge.
            below = [q for q in paras[max(0, i - 40):i + 40] if q.page == p.page and q is not p
                     and q.role in ("small", "other", "body") and 0 < q.y0 - p.y0 < 50
                     and abs(q.x0 - p.x0) < 12 and not _white(q) and id(q) not in skip]
            nxt = min(below, key=lambda q: q.y0) if below else None
            if nxt:
                skip.add(id(nxt))
                p.lines = p.lines + nxt.lines
                number = re.sub(r"SPM\.\s*", "SPM ", mc.group(2))
                text = f"{mc.group(1).title()} {number} {nxt.text}"
                p.role = "caption"

        # "Box 2 1" alone: a box, figure or table continued on another page.
        kind, number, title = _caption(text)
        if kind and not title and p.role in ("other", "small", "caption"):
            existing = next((x for x in rep.illustrations if x.kind == kind and x.number == number), None)
            if existing and kind == "Box":
                box, legend_of = existing, None
            continue

        # A part of a chapter ("Chapter 5.1", NXS) starts on a page with its title.
        mpart = re.fullmatch(r"Chapter (\d{1,2})\.(\d{1,2})", text.strip()) if p.role == "other" else None
        if mpart and chapter and int(mpart.group(1)) == chapter.number:
            title_para = next((q for q in paras[i + 1:i + 6] if q.page == p.page and q.role == "title"
                               and q.text.upper() != "TABLE OF CONTENTS"), None)
            number = f"{mpart.group(1)}.{mpart.group(2)}"
            if title_para and not any(s_.number == number for s_ in chapter.sections):
                chapter.parted = True
                title = re.sub(r"\s+", " ", title_para.text).strip()
                title = re.sub(r"(?<=[A-Za-z])\d+$", "", title).capitalize()     # footnote mark "...USE1"
                section = Section(number, rep.outline.get(number, title), p.page)
                chapter.sections.append(section)
                mode, unnumbered, box, legend_of = "chapter", None, None, None
            continue

        if p.role == "other" and re.fullmatch(r"Chapter \d{1,2}", text.strip(), re.IGNORECASE) \
                and _chapter_start(text, p, doc):
            p.role = "title"        # the chapter number, printed as "Chapter 3" ("CHAPTER 3", 2016-2019)

        if p.role == "title":
            t = text.upper()
            box = legend_of = None
            if "TABLE OF KNOWLEDGE GAPS" in t:
                rep.gap_table_pages.append(p.page)      # an appendix of its own (GA1 Appendix 4)
            if t == "KEY MESSAGES":
                mode, message = "km", None
            elif t == "KEY FINDINGS":
                # SCM: high-level messages "1) ...", each over its key findings 1.1, 1.2 ...
                mode, message = "bgm", None
            elif t.replace("-", "") in ("BACKGROUND MESSAGES", "BACKGROUND", "BACKGROUND TO THE KEY MESSAGES"):
                # "BACK-GROUND" on a cover page (PPA, whose section has no other title)
                mode, message = "bgm", None
            elif t.startswith("APPENDIX") and mode in ("km", "bgm", "appendix"):
                mode = "appendix"
            elif mode in ("km", "bgm") and re.match(r"^[A-F]\.\s+\S", text):
                # SUA: "A. Sustainable use of wild species is critical ..." opens
                # the background messages of key message A.
                mode, message, group = "bgm", None, text
                group_km = next((k for k in rep.key_messages if k.ident == text[0]), None)
            elif number := _chapter_start(text, p, doc):
                n = int(number.split(".")[0])
                if rep.chapter(n) is None:
                    chapter = Chapter(n, page=p.page)
                    rep.chapters.append(chapter)
                    _chapter_page(doc, p.page, chapter, rep)
                    # The title printed under the number, in the next title-size
                    # paragraphs on the page ("CREATING AN ENABLING ...").
                    chapter.printed_title = re.sub(r"\s+", " ", " ".join(
                        q.text for q in paras[i + 1:i + 8] if q.page == p.page and q.role == "title"
                        and not re.fullmatch(r"(?:chapter\s*)?\d{1,2}(?:\.\d)?\.?", q.text.strip(), re.I))).strip()
                elif "." in number and number != chapter.part:
                    # GA1: chapter 2 in parts 2.1, 2.2, 2.3, each with its own
                    # first page and authors.
                    part = Chapter(n)
                    _chapter_page(doc, p.page, part, rep)
                    for role, names in part.authors.items():
                        chapter.authors[role] = chapter.authors.get(role, []) + names
                    chapter.part_titles.append(part.title)
                if "." in number:
                    chapter.part = number
                    if not chapter.part_titles:
                        chapter.part_titles.append(chapter.title)
                mode, section, unnumbered = "chapter", None, None
            elif t == "REFERENCES" and chapter:
                mode = "refs"
            elif t.startswith("ANNEX"):
                mode = "annex"
                rep.annex_pages.setdefault(t.split("\n")[0], []).append(p.page)
            continue

        if mode == "bgm" and p.role in ("other", "caption") and not CAPTION_START.match(text) \
                and max(line.main.size for line in p.lines) >= 9.5:
            m = re.match(r"^([A-F])\s?(\d{1,2})\s+(\S.*)$", text)
            first = next((s for line in p.lines for s in line.spans if s.text.strip()), None)
            if re.fullmatch(r"[A-F]\s?\d{1,2}", p.label or "") or (m and first and first.color == WHITE):
                # SUA: a background message's headline, "A1" in white before it.
                ident = p.label.replace(" ", "") if p.label else m.group(1) + m.group(2)
                message = Message(f"{ident[0]}.{ident[1:]}", text if p.label else m.group(3), p.page,
                                  group=group)
                message.label_printed = True
                rep.background_messages.append(message)
                group_km = None
                continue
            if group_km is not None and not group_km.paras and p.role == "other" \
                    and _near(p.lines[0].main.size, 10.7) and p.lines[0].main.color == ACCENT:
                group_km.paras.append(text)     # SUA: the key message's text, under its title
                continue

        if mode == "bgm" and text.isupper() and p.role not in ("title", "footnote"):
            mh = re.match(r"^(\d)\)\s+(.+)$", text)
            if mh:
                # SCM: a high-level message, in capitals: the key message over
                # the key findings numbered after it (1.1, 1.2 ...).
                message = Message(mh.group(1), mh.group(2), p.page)
                rep.key_messages.append(message)
                continue
            if message is not None and message in rep.key_messages and not message.paras:
                message.headline += " " + text      # its next lines
                continue

        if mode == "annex":
            last = list(rep.annex_pages)[-1] if rep.annex_pages else ""
            if re.fullmatch(r"ANNEX [IVX]+", last) and p.role in ("h2", "h3") \
                    and p.page == rep.annex_pages[last][0]:
                # "ANNEX III", then "List of authors and review editors" under it (2016-2019).
                rep.annex_pages[f"{last}: {text.upper()}"] = rep.annex_pages.pop(last)
                continue
            if last and p.page not in rep.annex_pages[last]:
                rep.annex_pages[last].append(p.page)
            continue

        if p.role == "footnote":
            m = re.match(r"^(\d{1,3})\.?\s+(.*)$", text)
            if m:
                key = (footnote_chapter(), int(m.group(1)))
                rep.footnotes[key] = Footnote(int(m.group(1)), m.group(2), p.page)
            elif rep.footnotes and not re.match(r"^[*†‡§]", text) and p.y0 > 650:
                # A footnote continued on the next page or column, in the footnote
                # area at the foot; not a note of its own marked "*" (ECA chapter
                # 2), nor grey figure labels higher up (TCA p142).
                last = list(rep.footnotes)[-1]
                rep.footnotes[last].text += " " + text
            continue

        if p.role == "caption":
            kind, number, title = _caption(text)
            if kind and chapter and chapter.parted and not number.startswith("SPM"):
                m3 = re.match(r"^(\d{1,2})\s+(.*)$", title)       # "Table 5 1 2  Title"
                if m3:
                    number, title = f"{number}.{m3.group(1)}", m3.group(2)
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

        if p.role == "small" and mode in ("chapter", "exec") and chapter and p.page in ref_pages \
                and _REF_START.match(text) and (_AUTHOR_START.match(text) or _ORG_START.match(text)):
            mode = "refs"           # the list starts without a title (LDR)

        if p.role == "small":
            if mode == "refs":
                for piece in _ref_pieces(p):
                    if chapter.references and _ref_continues(chapter.references[-1].text, piece):
                        chapter.references[-1].text = _join(chapter.references[-1].text, piece,
                                                            True, True) if _URLISH.search(chapter.references[-1].text) \
                            else chapter.references[-1].text + " " + piece
                    else:
                        chapter.references.append(RefEntry(piece, p.page))
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

        if p.role in ("h2", "h3", "h4", "h4b"):
            if p.role != "h4b":
                box = legend_of = None
            else:
                p.role = "h4"
            # Numbers printed with spaces: "4. 2. 1 Environmental drivers" (SUA chapter 4).
            text = re.sub(r"^\d{1,2}(?:\.\s?\d{1,2}){1,4}(?=\.?\s)", lambda m_: m_.group(0).replace(" ", ""), text)
            if mode in ("km", "bgm"):
                if mode == "bgm" and p.role == "h2":
                    group = text
                if mode == "km" and p.role == "h3":
                    km_heading = text
                continue
            if mode not in ("chapter", "exec") or chapter is None:
                continue
            # PPA chapter 4: "Section 1. Nature and significance ..." is 4.1, and
            # its sub-sections are printed without the chapter: "1.1" is 4.1.1.
            msec = re.match(r"^Section (\d{1,2})\.?\s+(.+)$", text, re.IGNORECASE)
            if msec and p.role in ("h2", "h3"):
                chapter.section_words = True
                number = f"{chapter.number}.{msec.group(1)}"
                title = next((re.sub(r"^Section \d+\.?\s*", "", re.sub(r"\s+", " ", t_)).strip()
                              for _, t_, pg in doc.get_toc()
                              if re.match(rf"^Section {msec.group(1)}\b", t_.strip(), re.IGNORECASE)), "")
                section = Section(number, title or sentence_case(msec.group(2), casing), p.page)
                chapter.sections.append(section)
                mode, unnumbered = "chapter", None
                last_heading = (p.page, p.y1)
                continue
            if chapter.section_words and section and re.match(r"^\d{1,2}\.\d", text) \
                    and not text.startswith(section.number + "."):
                text = f"{chapter.number}.{text}"
            m = SECTION_NUMBER.match(text)
            if p.role == "h2" and not m and last_heading[0] == p.page and -8 <= p.y0 - last_heading[1] < 25:
                last_heading = (p.page, p.y1)
                continue        # the heading's next lines, in another block (PPA chapter 4)
            if p.role == "h2":
                last_heading = (p.page, p.y1)
            if p.role == "h2" and not m:
                if text.upper() in ("EXECUTIVE SUMMARY", "KEY FINDINGS"):       # SCM: "Key findings"
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
            if chapter.parted and number.count(".") == 2 and not (section and number.startswith(section.number + ".")):
                rep.problems.append(f"p{p.page}: heading {number} outside its part")
                continue
            if chapter.part:
                # GA1 chapter 2: "2.1.1" is a sub-chapter, "2.1.1.1" a heading in it.
                if number.count(".") == 2 and number.startswith(chapter.part + "."):
                    if title.isupper():
                        title = sentence_case(title, casing)
                    section = Section(number, title, p.page)
                    chapter.sections.append(section)
                elif section and number.startswith(section.number + "."):
                    section.paras.append(f"## {number} {title}")
                    section.heading_pages[number] = p.page
                elif number != chapter.part:
                    rep.problems.append(f"p{p.page}: heading {number} outside part {chapter.part}")
                continue
            if number.count(".") == 1:
                if number not in rep.outline and title.isupper():
                    title = sentence_case(title, casing)     # "1.1 INTRODUCTION" (SUA)
                section = Section(number, rep.outline.get(number, title), p.page)
                chapter.sections.append(section)
                if number not in rep.outline and rep.outline:
                    rep.problems.append(f"p{p.page}: section {number} not in the PDF outline; title as printed")
            elif section and number.startswith(section.number + "."):
                section.paras.append(f"## {number} {title}")
                section.heading_pages[number] = p.page
            else:
                rep.problems.append(f"p{p.page}: heading {number} outside its section")
            continue

        if p.role != "body":
            continue

        # A heading of the box being read, in DIN Black at body size: "Evidence
        # relevant in the context of the Aichi Biodiversity Targets" (ECA Box
        # SPM 4), not text of the message the box sits in.
        if box is not None and all("DIN" in s_.font and s_.bold for line in p.lines for s_ in line.spans
                                   if s_.text.strip()):
            box.legend.append("## " + text)
            continue

        # Body text
        legend_of = None

        if mode == "km":
            mk = re.match(r"^([A-F])\.\d{1,2}\s", text)
            if mk and km_heading and not p.label:
                # SUA: key messages A to D, each a heading over its background
                # messages' headlines (A.1, A.2 ...), which are read later.
                if not any(k.ident == mk.group(1) for k in rep.key_messages):
                    rep.key_messages.append(Message(mk.group(1), km_heading, p.page))
                continue
            # KM1 (BBA), KM-A1 (IAS), A1 (LDR, whose background messages are
            # numbered 1, 2 ...) or 1 (PPA)
            group_letter = re.match(r"^([A-F])\.\s", km_heading)
            if re.fullmatch(r"KM\s?-?\s?[A-F]?\d{1,2}|[A-F]\s?\d{1,2}|\d{1,2}", p.label or ""):
                message = Message(re.sub(r"\s", "", p.label), p.lead or text, p.page, group=km_heading)
                message.label_printed = True
                rep.key_messages.append(message)
                rest = p.rest if p.lead else ""
                if rest:
                    message.paras.append(rest)
            elif not p.label and p.lead and group_letter and text.startswith(p.lead) \
                    and not any(k.label_printed for k in rep.key_messages):
                # ECA: no labels. Each group (A. A PRECIOUS ASSET ...) is a key
                # message, its bold statements the text, as the background
                # messages A1, A2 ... follow the groups.
                letter = group_letter.group(1)
                message = next((k for k in rep.key_messages if k.ident == letter), None)
                if message is None:
                    message = Message(letter, km_heading[3:].strip(), p.page)
                    rep.key_messages.append(message)
                _add_para(message.paras, text)
            elif message and not message.paras and p.lead and not message.headline.rstrip().endswith(_TERMINAL):
                # The headline goes on in another column or after a footnote (NXS).
                message.headline = f"{message.headline} {p.lead}"
                if p.rest:
                    message.paras.append(p.rest)
            elif message:
                _add_para(message.paras, text)
        elif mode == "bgm" and (mk := re.match(r"^Key finding (\d{1,2}\.\d{1,2})\s*:\s*", text)):
            # SCM: "Key finding 1.1 : Scenarios and models can ..."
            lead = (p.lead or text)[mk.end():] if (p.lead or text).startswith(mk.group(0)) else p.lead or text
            message = Message(mk.group(1), lead, p.page, group=rep.key_messages[-1].headline
                              if rep.key_messages else "")
            message.label_printed = True
            rep.background_messages.append(message)
            if p.lead and p.rest:
                message.paras.append(p.rest)
        elif mode == "bgm":
            first = next((s_ for line in p.lines for s_ in line.spans if s_.text.strip()), None)
            # Only in a report that prints no labels at all.
            # A headline goes on after a column or page break: it starts in lower case
            # ("recreational benefits ...", PPA); or it is printed twice (PPA 14).
            unlabelled = (not p.label and p.lead and first is not None and coloured(first.color)
                          and first.bold and not any(b.label_printed for b in rep.background_messages)
                          and p.lead[:1].isupper()
                          and not any(b.headline == p.lead for b in rep.background_messages))
            if re.fullmatch(r"[A-F]\s?\d{1,2}|\d{1,2}", p.label or "") or unlabelled:
                # PPA prints no numbers: its messages are numbered in order, 1, 2 ...
                ident = p.label.replace(" ", "") if p.label else str(len(rep.background_messages) + 1)
                message = Message(ident, p.lead or text, p.page, group=group)
                message.label_printed = bool(p.label)
                rep.background_messages.append(message)
                rest = p.rest if p.lead else ""
                if rest:
                    message.paras.append(rest)
            elif message and not message.paras and p.lead and not message.headline.rstrip().endswith(_TERMINAL):
                # The headline goes on in another column or after a footnote (NXS).
                message.headline = f"{message.headline} {p.lead}"
                if p.rest:
                    message.paras.append(p.rest)
            elif message:
                _add_para(message.paras, text)
        elif mode == "exec" and chapter and chapter.parted and section is not None:
            # A part's own executive summary opens its text.
            if not section.paras:
                section.paras.append("## Executive summary")
            _add_para(section.paras, (f"{p.label}. " if re.fullmatch(r"\d{1,2}", p.label or "") else "") + text)
        elif mode == "exec" and chapter:
            if re.fullmatch(r"\d{1,2}", p.label or ""):
                finding = Message(p.label, p.lead or text, p.page)
                chapter.executive_summary.append(finding)
                if p.lead and p.rest:
                    finding.paras.append(p.rest)
            elif chapter.executive_summary and not (p.lead and not chapter.executive_summary[-1].ident):
                _add_para(chapter.executive_summary[-1].paras, text)
            else:
                # Unnumbered findings (2016-2019, NXS), each with its bold lead or as a paragraph.
                finding = Message("", p.lead or text, p.page)
                chapter.executive_summary.append(finding)
                if p.lead and p.rest:
                    finding.paras.append(p.rest)
        elif mode == "chapter":
            target = section or unnumbered
            if target is not None:
                _add_para(target.paras, text)
    # The title printed on the chapter's page wins over its citation's when they
    # differ (BBA chapter 6: "... the financial system and other actors" printed,
    # "... the financial sector, Indigenous Peoples ..." in the citation).
    # A printed title found inside the citation's is only a part of it, as
    # read from the page (GA1 chapter 1: "OF THE IPBES GLOBAL ASSESSMENT ON").
    letters = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    for c in rep.chapters:
        printed = re.sub(r"(?<=[A-Za-z?!.)])\d+(?:,\d+)*$", "", c.printed_title).strip()  # marks: "...?1", "...PEOPLE1,2"
        if printed and c.title and letters(printed) not in letters(c.title) and not c.part_titles:
            rep.problems.append(f"chapter {c.number}: title printed '{printed}', but '{c.title}' in its "
                                f"citation; the printed one kept")
            c.title = sentence_case(printed, casing) if printed.isupper() else printed
    for c in rep.chapters:
        if len(c.part_titles) > 1:
            shared = re.split(r"\s*[–-]\s*", c.part_titles[0])[0]
            if all(t.startswith(shared) for t in c.part_titles):
                c.title = shared       # GA1: "Status and trends – Drivers of change", "... – Nature"
    _split_sections(rep, casing)
    for k in rep.key_messages:
        if k.headline.isupper():
            k.headline = sentence_case(k.headline, casing)      # SCM's high-level messages
    # A chapter whose citation gives no title ("..., 2016: Background to ...",
    # PPA chapter 1): the title from the outline, "CHAPTER 1: BACKGROUND TO ...".
    for c in rep.chapters:
        if not c.title:
            t = next((re.sub(r"\s+", " ", title).strip() for _, title, _ in doc.get_toc()
                      if re.match(rf"^CHAPTER\s+{c.number}\s*[:.]", title, re.IGNORECASE)), "")
            t = re.sub(r"^CHAPTER\s+\d+\s*[:.]\s*", "", t, flags=re.IGNORECASE)
            c.title = sentence_case(t, casing) if t.isupper() else t
    if not rep.outline and any(c.sections for c in rep.chapters):
        rep.problems.append("the PDF outline has no sections: titles as printed, those in capitals "
                            "in sentence case")
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


def _split_sections(rep: Report, casing: dict[str, str]) -> None:
    """Every numbered section a sub-chapter of its own, as the manual
    conversions do: 3.1.1 out of 3.1's text, with the text printed under its
    heading; 3.1 keeps only its own text (none if 3.1.1 follows its heading).
    The numbers give the hierarchy."""
    for c in rep.chapters:
        out: list[Section] = []
        seen: dict[str, Section] = {}
        for sec in c.sections:
            cur = Section(sec.number, sec.title, sec.page)
            out.append(cur)
            seen[cur.number] = cur
            for para in sec.paras:
                m = re.match(r"^## (\d{1,2}(?:\.\d{1,2})+)\s+(.*)$", para)
                if not m:
                    cur.paras.append(para)
                elif m.group(1) in seen:
                    rep.problems.append(f"heading {m.group(1)} printed twice; its text joins the first")
                    cur = seen[m.group(1)]
                else:
                    title = m.group(2).strip()
                    cur = Section(m.group(1), sentence_case(title, casing) if title.isupper() else title,
                                  sec.heading_pages.get(m.group(1), cur.page))
                    out.append(cur)
                    seen[cur.number] = cur
        c.sections = out


def word_casing(paras: list[Para]) -> dict[str, str]:
    """Each word's usual form inside a sentence in the report's text
    ('ipbes' -> 'IPBES', 'goals' -> 'goals'), to write a title printed in
    capitals in sentence case."""
    import collections
    forms: dict[str, collections.Counter] = {}
    phrases: collections.Counter = collections.Counter()
    for p in paras:
        if p.role != "body":
            continue
        # A word before a number ("Chapter 3", "Figure 2") is a name for it.
        for m in re.finditer(r"(?<![.!?:]\s)(?<=\s)([A-Za-z][\w’'-]*)(?!\s+\d)", p.text):
            w = m.group(1)
            forms.setdefault(w.lower(), collections.Counter())[w] += 1
        # Names of several words: "Sustainable Development Goals".
        for m in re.finditer(r"(?<=\s)([A-Z][a-z]+(?:\s(?:of\s|and\s|for\s|on\s)?[A-Z][a-z]+)+)", p.text):
            phrases[m.group(1)] += 1
    casing = {k: c.most_common(1)[0][0] for k, c in forms.items()}
    text = " ".join(p.text for p in paras if p.role == "body")
    for phrase, n in phrases.items():
        if n >= 3 and n > len(re.findall(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE)) - n:
            casing[phrase.lower()] = phrase         # mostly written so
    return casing


def sentence_case(title: str, casing: dict[str, str]) -> str:
    """'ASSESSING SUSTAINABLE USE: A ROADMAP' -> 'Assessing sustainable use: a roadmap'."""
    def word(m: re.Match) -> str:
        w = m.group(0)
        form = casing.get(w.lower(), w.lower())
        return form if form.lower() == w.lower() else w.lower()
    out = re.sub(r"[A-Za-z][\w’'-]*", word, title.strip())
    for key, form in sorted(((k, v) for k, v in casing.items() if " " in k), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(key)}\b", form, out, flags=re.IGNORECASE)
    return out[:1].upper() + out[1:]


def _chapter_start(text: str, p: Para, doc: pymupdf.Document) -> str:
    """The chapter number, '3' (or a part of one, '2.1', GA1), from '3' or
    'Chapter 3' on a chapter's first page, which (or the next) has the
    chapter's citation; '' if this is not one. The number can share its text
    block with the title (GA1: '3 ASSESSING PROGRESS ...'): its first line."""
    pattern = r"(?:Chapter )?(\d{1,2}(?:\.\d)?)\.?"
    m = re.fullmatch(pattern, text.strip(), re.IGNORECASE)
    if not m and p.lines and p.lines[0].main.size >= 100:
        m = re.fullmatch(pattern, p.lines[0].text.strip(), re.IGNORECASE)
    if not m:
        return ""
    # "This chapter should be cited as:" in the reports of 2016-2019.
    cited = any("SHOULD BE CITED AS" in doc[q - 1].get_text().upper()
                for q in (p.page, p.page + 1) if q <= doc.page_count)
    return m.group(1) if cited else ""


# The year in brackets, "(2020a)", or bare: "Nielsen. 2007. Do ..." (PPA chapter 2),
# "Veenman, S., 2013: Coping ..." (SCM).
_BARE_YEAR = r"(?<=[.,]\s)(?:1[6-9]|20)\d\d[a-z]?[.:](?=\s|$)"
_REF_START = re.compile(
    r"^[^\s(](?:[^()]|\([^()]{0,1000}\)){0,1500}?(?:\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)[^)]{0,40}\)"
    r"|" + _BARE_YEAR + ")",
    re.IGNORECASE)


def _bibliography_entry_ends(prev: str, nxt: str) -> bool:
    """Notes-and-bibliography style (GA1 chapter 2): the year comes late, "Nature
    459 (May 1, 2009): 185–92. https://doi.org/...". An entry ending with its
    link, or with a full stop after a year, is complete when the next text
    starts with an author: "Assefa, Engdawork, and ...". Not in author-date
    entries, whose year in brackets comes first ("Smith, J. (2021). ... In J.
    Bebbington, C." / "Larrinaga, B. ... (Eds)")."""
    return bool(_CHICAGO_AUTHOR.match(nxt)) and not re.search(r"\((?:1[6-9]|20)\d\d", prev) \
        and bool(re.search(r"(?:https?://|doi\.org/)\S+$", prev)
                 or (prev.endswith(".") and re.search(r"\b(?:1[6-9]|20)\d\d\b", prev)))


def _ref_pieces(p: Para) -> list[str]:
    """A reference-list paragraph, cut where a new entry starts inside it: GA1
    chapter 2 prints a column of entries as one text block. Most paragraphs
    are one entry, or the part of one."""
    if len(p.lines) < 2:
        return [p.text]
    pieces = [""]
    for line in p.lines:
        t = re.sub(r"\s+", " ", line.text).strip()
        if not t:
            continue
        if pieces[-1] and _bibliography_entry_ends(pieces[-1], t):
            pieces.append(t)
        else:
            pieces[-1] = _join(pieces[-1], t, False, False) if pieces[-1] else t
    if len(pieces) == 1:
        return [p.text]         # as read with its spans (links, hyphens)
    return [x for x in pieces if x]


def _ref_continues(prev: str, nxt: str) -> bool:
    """Whether a reference-list block continues the previous entry. Entries
    break across columns and pages anywhere, even inside the author list."""
    if nxt.startswith(("http", "www.", "doi", "/")) or nxt[:1] in ",;:.&)–" or nxt[:1].isdigit():
        return True
    if _bibliography_entry_ends(prev, nxt):
        return False
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


# "Assefa, Engdawork," / "Armbrust, E. Virginia." / "Food and Agriculture Organization."
_CHICAGO_AUTHOR = re.compile(r"^[A-Z][\w’'-]+(?:\s(?:de|van|von|da|del|der|la|le)?\s?[A-Z][\w’'-]+)?,\s[A-Z][\w’'.-]*(?:\s[A-Z][\w’'.-]*)?[,.]\s")
_YEAR = re.compile(r"\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)|" + _BARE_YEAR, re.IGNORECASE)
# "Bull, J. W., ..." or "IPBES (with ..."
_AUTHOR_START = re.compile(r"^(?:[A-Za-z][^\s,()]*(?:\s[^\s,()]+){0,3},\s(?:[A-Z]\.|[A-Z][a-z]{0,3}\.)|[A-Z][\w-]*\s\(with\b)")
# "FAO. (2018a)." or "European Commission. (2019)."
_ORG_START = re.compile(r"^[A-Z][^()\n]{0,120}?\.\s\((?:(?:1[6-9]|20)\d\d[a-z]?|n\.d\.)")
# "E. J. (2020)": the end of an author list, not a new entry.
_INITIALS_START = re.compile(r"^(?:[A-Z]\.\s?){1,3}(?:,|\(|&)")


# On the page of editors: "Assessment Co-Chair", "Head, Technical Support Unit",
# "Science Officer, Technical Support Unit, IPBES Secretariat, Germany" ...
_EDITOR_ROLE = re.compile(r"(Assessment Co-?chair|Multidisciplinary Expert Panel"
                          r"|(?:[^,]*,\s*)?(?:Head of )?Technical Support Unit)\b[,\s]*(.*)$", re.IGNORECASE)


def _read_editors(block: str, rep: Report) -> None:
    """The people under "Edited by:". Each has a name and a role, with the
    affiliation either before the role (IAS: name, affiliation, role) or after
    it on the role's line (AFA: name, "role, affiliation", which may wrap)."""
    found: list[tuple[str, list[str], str]] = []
    pending: list[str] = []
    for line in (x.strip() for x in block.split("\n")):
        if not line:
            continue
        m = _EDITOR_ROLE.match(line)
        if not m:
            pending.append(line)
            continue
        if not pending:
            continue
        role = "cs" if "chair" in m.group(1).lower() else "ts" if "support" in m.group(1).lower() else ""
        if m.group(2):
            if found:
                found[-1][1].extend(pending[:-1])
            name, aff = pending[-1], [re.sub(r"^at\s+", "", m.group(2))]
        else:
            name, aff = pending[0], pending[1:]
        found.append((re.sub(r"(?<=\w)- (?=\w)", "-", name), aff, role))
        pending = []
    # Others (SCM's Multidisciplinary Expert Panel member) are in the management committee, page 2.
    rep.editors += [(name, " ".join(aff), role) for name, aff, role in found if role]
    # Whatever is left after the last role: SCM's other editors, by initials.
    if any(re.match(r"[A-Z]\.\s?[A-Z]?", x) and "," in x for x in pending):
        rep.problems.append("title page: other editors are listed without a role; no role recorded from this page")


def _read_committee(text: str, rep: Report) -> None:
    """The names after "MEMBERS OF THE MANAGEMENT COMMITTEE WHO PROVIDED GUIDANCE ...
    ASSESSMENT", with "(Multidisciplinary Expert Panel)" and "(Bureau)" in the older reports."""
    m = re.search(r"MANAGEMENT COMMITTEE\s+WHO PROVIDED.*?ASSESSMENT:?\s*\n(.*)", text, re.S)
    if not m:
        return
    lines = []
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or line.isupper() or line.startswith(("This report", "©", "ISBN", "The designations")):
            break
        lines.append(line)
        if line.endswith("."):
            break
    names = re.sub(r"\([^)]*\)", ",", re.sub(r"\s+", " ", " ".join(lines))).rstrip(". ")
    names = [re.sub(r"^and\s+", "", n.strip()) for n in re.split(r"\s*[,;]\s*|\s+and\s+", names)]
    rep.committee = [n for n in names if n]


def _front_matter(doc: pymupdf.Document, rep: Report) -> None:
    for pno in range(min(6, doc.page_count)):
        text = doc[pno].get_text()
        if not rep.committee:
            _read_committee(text, rep)
        m = re.search(r"SUGGESTED CITATION:?\s*\n(.*?)(?:\n\s*\n|$)", text, re.S)
        if m and not rep.title:
            cite = re.sub(r"\s+", " ", m.group(1))
            # "IPBES (2018): The ... report. Archer, E. ...", "IPBES (2019), Global ...,
            # Brondízio, E. S. ...", "IPBES (2016). The ... production. S.G. Potts ..."
            t = re.match(r"IPBES \(\d{4}\)[.,:]\s*(.+?)[.,]\s+(?:[A-Z][\w’'-]+,? [A-Z]\.|[A-Z]\.\s?[A-Z])", cite)
            rep.title = t.group(1) if t else ""
            # The editors' surnames, as they may have several words ("Torre-Marin Rando, A.").
            editors = cite[t.end(1):].split("(eds", 1)[0] if t else ""
            rep.editor_surnames = re.findall(r"([A-Z][\w’'-]*(?:\s[A-Z][\w’'-]*)*),\s*(?:[A-Z]\.\s*)+", editors)
            d = re.search(r"(10\.\d{4,9}/[^\s]+?)\.?(?:\s|$)", cite)
            rep.doi = f"https://doi.org/{d.group(1)}" if d else ""
        if "Edited by:" in text and not rep.editors:
            _read_editors(text.split("Edited by:", 1)[1], rep)


ROLE_HEADINGS = {
    # NXS chapter 5 lists both together; the experts list tells fellows apart.
    "LEAD AUTHORS AND FELLOWS": "la+fl",
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
            if m.size < 6.75 or m.size > 10.5:
                continue            # footnotes (6.5 pt) and titles; the lists are 7-8 pt, their
                                    # headings 10 pt in the reports of 2016-2019
            out.append("".join(x.text for x in spans if not (x.sup and x.text.strip().isdigit())))
    return "\n".join(out)


def _chapter_page(doc: pymupdf.Document, page: int, chapter: Chapter, rep: Report) -> None:
    """Authors, citation and DOI from the chapter's first page (and the next,
    when the lists or the citation continue there)."""
    text = _page_text(doc[page - 1])
    if "SHOULD BE CITED AS" not in text.upper() and page < doc.page_count:
        text += "\n" + _page_text(doc[page])
    text = text.translate(_LIGATURES)
    # The map disclaimer on the second page can sit inside an author list.
    text = re.sub(r"Disclaimer on maps.*?represented therein\.?", "", text, flags=re.S)
    heads = "|".join(sorted(map(re.escape, ROLE_HEADINGS), key=len, reverse=True))
    # "REVIEW EDITORS 2:": a footnote number may follow the heading.
    # "Coordinating Lead Authors:" in the reports of 2016-2019, without the colon in AFA.
    parts = re.split(rf"\n\s*({heads})\s*\d*\s*:?\s*\n", "\n" + text, flags=re.IGNORECASE)
    for head, body in zip(parts[1::2], parts[2::2]):
        body = re.split(r"THIS CHAPTER SHOULD BE CITED AS|\nChapter \d+\n", body, flags=re.IGNORECASE)[0]
        body = re.sub(r"\n\s*\d+\s*\n", "\n", body)        # stray footnote numbers
        role = ROLE_HEADINGS[re.sub(r"\s+", " ", head).upper()]
        chapter.authors[role] = _split_people(re.sub(r"\s+", " ", body).strip())
    # The citation ends with the DOI, which may be broken after a "/".
    m = re.search(r"THIS CHAPTER SHOULD BE CITED AS:\s*(.*?(?:DOI:\s*)?https?://doi\.org/(?:\S*/[ \t]*\n\s*\S+|\S+))",
                  text, re.S | re.IGNORECASE)
    if not m:
        # Without a DOI (2016-2019): the citation ends with its pages, "pp.1-51."
        m = re.search(r"THIS CHAPTER SHOULD BE CITED AS:\s*(.*?pp\.\s*[\dxvi]+\s*[-–]\s*[\dxvi]+\.?)",
                      text, re.S | re.IGNORECASE)
    if not m:
        # Without pages (SCM): the citation ends with the publisher's place.
        m = re.search(r"THIS CHAPTER SHOULD BE CITED AS:\s*(.*?Bonn,\s+Germany)", text, re.S | re.IGNORECASE)
    if m:
        cite = re.sub(r"(doi\.org/\S*/)\s+", r"\1", m.group(1))
        cite = re.sub(r"\s+", " ", cite).strip()
        chapter.cite_as = cite
        d = re.search(r"(10\.\d{4,9}/\S+?)\.?(?:\s|$)", cite)
        chapter.doi = f"https://doi.org/{d.group(1)}" if d else ""
        # "Chapter 6:" or "Chapter 6.", then "In:" (or "In IPBES (2018)", LDR)
        # SCM has no chapter number: "..., 2016: Overview and vision. In IPBES (2016):"
        t = re.search(r"Chapter \d+(?:\.\d+)?[:.]\s*(.+?)\s+In:?\s", cite) or re.search(r"\b\d{4}[a-z]?:\s*(.+?)\s+In:?\s", cite)
        # PPA: "..., 2016: Chapter 1 Background to ...": the number goes.
        chapter.title = re.sub(r"^Chapter\s+\d+[:.]?\s*", "", t.group(1).strip().rstrip(".")) if t else ""
    if not chapter.doi:
        rep.problems.append(f"chapter {chapter.number}: no DOI on its first pages (p{page})")


def _split_people(text: str) -> list[str]:
    """'A (X / Y), B (Z).' -> ['A (X / Y)', 'B (Z)']; names without countries
    (the technical support unit) are split on commas."""
    text = text.rstrip(". ")
    if "(" not in text:
        return [n.strip() for n in re.split(r"[,;]", text) if n.strip()]
    # A name may hold a nickname: "Margaret (Peggy) Smith (Canada)". Names are
    # separated by commas, now and then by semicolons (SUA chapter 4).
    # A country can hold brackets itself: "(Iran (Islamic Republic of))" (APA).
    entries = [m.group(0).strip(" ,;") for m in
               re.finditer(r"[^,;()]+(?:\([^()]*\)[^,;()]+)?\((?:[^()]|\([^()]*\))*\)", text)]
    # A first bracket holding a country is not a nickname: a comma is missing
    # between two people, "Philip Ivey (South Africa) Patrice Bigombe Logo
    # (Cameroon)" (AFA chapter 4).
    import geo
    out = []
    for e in entries:
        m = re.match(r"^([^()]+\(([^()]*)\))\s+([^()]+\(.*\))$", e)
        if m and all(geo.country_code_for_label(x.strip()) for x in re.split(r"[/,;]", m.group(2)) if x.strip()):
            out += [m.group(1).strip(), m.group(3).strip()]
        else:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Markdown: the report as read, for checking
# ---------------------------------------------------------------------------

def to_markdown(rep: Report, name: str) -> str:
    out = [f"# {rep.title}", "", f"Read from {name}. DOI: {rep.doi}", ""]
    out += [f"- {ROLE_NAMES[r]}: {n} ({a})" for n, a, r in rep.editors] + [""]

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
            out += [(f"{f.ident}. " if f.ident else "") + f"**{f.headline}** {f.text}", ""]
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
    import time

    api = f"https://api.zotero.org/groups/{group}"

    def get(client, url, params):
        """GET with retries: the API sometimes answers 502 or times out."""
        for attempt in range(6):
            try:
                r = client.get(url, params=params)
                if r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()
                    return r
                wait = int(r.headers.get("Retry-After", 0)) or 2 ** attempt
            except httpx.TransportError:
                wait = 2 ** attempt
            time.sleep(wait)
        r.raise_for_status()
        return r

    with httpx.Client(timeout=60, headers={"Zotero-API-Version": "3"}) as client:
        collections = {}
        start = 0
        while True:
            r = get(client, f"{api}/collections", {"limit": 100, "start": start})
            batch = r.json()
            collections.update({c["key"]: c["data"]["name"] for c in batch})
            start += len(batch)
            if not batch or start >= int(r.headers.get("Total-Results", start)):
                break
        items, start = [], 0
        while True:
            r = get(client, f"{api}/items/top", {"limit": 100, "start": start, "format": "json"})
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

    def _rdf_name(self, name: str, code: str) -> bool:
        """Whether `name` is one of the RDF's labels for `code`, brackets aside."""
        _, by_code = self.geo._country_index()
        concept = by_code.get(code)
        keys = {self.geo._key(lab) for lab in self.geo._labels(concept)} if concept is not None else set()
        return self.geo._key(name) in keys

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
            # A name that is only an alias ("UK") is reported below; the RDF's own
            # short names ("Iran" for "Iran (Islamic Republic of)") are not.
            if code and not (alias and code == alias and self.geo._COUNTRY_ALIASES.get(t.lower())
                             and not self._rdf_name(t, code)):
                return code
        if alias:
            self.log.add("countries", f"{where}: '{name}' is not a country name in ipbes-geo.rdf; "
                                      f"read as {alias}")
            return alias
        if re.fullmatch(r"[A-Z]{2,}(?:[\s-][A-Z]{2,}){0,3}", name):      # "FAO", "GEO BON"
            self.log.add("countries", f"{where}: '{name}' is an organization, not a country; "
                                      f"not used as a country (the person is kept)")
        else:
            self.log.add("countries", f"{where}: country '{name}' not found in ipbes-geo.rdf; "
                                      f"not used as a country (the person is kept)")
        return ""

    def resolves(self, text: str) -> bool:
        """Whether every part of 'A, B / C' is a country (no logging)."""
        quiet = Countries.__new__(Countries)
        quiet.geo, quiet.log = self.geo, Log()
        parts = [x for x in re.split(r"\s*[/,;]\s*", text) if x.strip()]
        return bool(parts) and all(quiet.code(x, "") and not quiet.log.items for x in parts)

    def codes(self, text: str, where: str) -> list[str]:
        """'Plurinational State of Bolivia / Germany', 'India, United States of America',
        'Bolivia and USA' (AMA; but 'Trinidad and Tobago' is one country)."""
        out = []
        for part in re.split(r"\s*[/,;]\s*", text):
            part = re.sub(r"^the\s+", "", part.strip(), flags=re.IGNORECASE)
            pieces = re.split(r"\s+and\s+", part)
            if len(pieces) > 1 and not self.geo.country_code_for_label(part) \
                    and all(self.geo.country_code_for_label(x) for x in pieces):
                parts = pieces
            else:
                parts = [part]
            for piece in parts:
                if piece:
                    code = self.code(piece, where)
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
    "ts": "Technical Support Unit",
    "mc": "Management Committee",
    "cl": "Coordinating Lead Author",
    "la": "Lead Author",
    "fl": "Fellow",
    "ca": "Contributing Author",
    "re": "Review Editor",
}
_CSV_ROLES = {"chair": "cs", "chair - cla": "cs", "coordinating lead author": "cl", "lead author": "la", "fellow": "fl",
              "review editor": "re", "contributing author": "ca",
              "mep task force/expert group member": "mc", "bureau task force/expert group member": "mc"}
_PARTICLES = {"de", "da", "das", "do", "dos", "del", "della", "der", "den", "van", "von", "zu", "di",
              "du", "la", "le", "bin", "binti", "al", "el", "ten", "ter", "mc"}


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(?:dr|prof|mr|mrs|ms)\b\.?", " ", s.lower())
    return re.sub(r"[^a-z ]+", " ", s.replace("-", " ")).split()


# Short forms of first names whose first three letters differ from the full
# name's (Steve / Stephen are matched by those): "Mike Murray-Hudson" is
# "Michael Alan Murray-Hudson" (GA1).
_SHORT_FORMS = {
    "mike": "michael", "mick": "michael", "bill": "william", "will": "william", "bob": "robert",
    "rob": "robert", "bert": "robert", "dick": "richard", "rick": "richard", "ted": "edward",
    "ned": "edward", "jim": "james", "jack": "john", "peggy": "margaret", "meg": "margaret",
    "maggie": "margaret", "liz": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "sandy": "alexander", "sasha": "alexander", "tony": "anthony", "hank": "henry",
    "harry": "henry", "chuck": "charles", "polly": "mary", "molly": "mary", "kate": "katherine",
    "kathy": "katherine", "sally": "sarah", "jenny": "jennifer", "nancy": "anne",
}


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

    def covered(x: list[str], y: list[str]) -> bool:
        """Each word of x in y, or an initial of a word of y."""
        return all(t in y or (len(t) == 1 and any(u.startswith(t) for u in y)) for t in x)
    if last_same and (covered(ta, tb) or covered(tb, ta)):
        return "close"              # "William W. L. Cheung" / "Wai Lung (William) Cheung"
    if last_same and _SHORT_FORMS.get(ta[0], ta[0]) == _SHORT_FORMS.get(tb[0], tb[0]):
        return "close"              # Mike / Michael
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
    seen: list[str] = field(default_factory=list)   # the name as printed in each source

    def __post_init__(self):
        self.seen.append(self.name)

    @property
    def slug(self) -> str:
        def clean(s):
            return re.sub(r"[^\w-]", "", s.replace(" ", ""))
        return f"{clean(self.last)}_{clean(self.first)}"


def fix_typo(p: "Person", log: "Log") -> None:
    """A name printed with a typo where it was first read ('Lusine Margayan' on
    one chapter page, 'Margaryan' on the others, in the annex and the experts
    list): the spelling most sources agree on is kept."""
    import collections
    import difflib

    words = p.name.split()
    counts = collections.Counter(re.sub(r"\s+", " ", n).strip() for n in p.seen)
    for variant, n in counts.most_common():
        other = variant.split()
        if n <= counts[p.name] or len(other) != len(words):
            continue
        diff = [(a, b) for a, b in zip(words, other) if a != b]
        if len(diff) == 1 and difflib.SequenceMatcher(None, *diff[0]).ratio() >= 0.8:
            old, new = diff[0]
            log.add("names", f"'{p.name}' written '{variant}' in {n} of {len(p.seen)} sources; "
                             f"'{variant}' kept")
            p.name = variant
            p.first = re.sub(rf"\b{re.escape(old)}\b", new, p.first)
            p.last = re.sub(rf"\b{re.escape(old)}\b", new, p.last)
            return


def _split_name(name: str, surnames: list[str]) -> tuple[str, str]:
    """(first, last). A surname from the chapter citation wins; otherwise the
    last word, with particles (de, van ...) kept with it."""
    toks = name.split()
    for s in sorted(surnames, key=len, reverse=True):
        if _tokens(name)[-len(_tokens(s)):] == _tokens(s) and len(_tokens(s)) < len(_tokens(name)):
            # Particles stay with it, if the citation leaves them out ("de Aguiar").
            i = len(toks) - len(s.split())
            while i > 1 and toks[i - 1].lower() in _PARTICLES:
                i -= 1
            return " ".join(toks[:i]), " ".join(toks[i:])
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
                elif _near(m.size, 7.5) and m.bold and cur is not None and not cur["lines"] \
                        and cur["page"] == pno:
                    # The name goes on in the next line: "Ramón de la Concepción
                    # Pichs-" / "Madruga" (GA1 Annex III).
                    cur["name"] = _join(cur["name"], text, False, False)
                elif _near(m.size, 7.5) and m.bold:
                    cur = {"name": text, "lines": [], "group": group, "page": pno}
                    people.append(cur)
                elif cur is not None and _near(m.size, 7.5):
                    cur["lines"].append(text)
    return people


ANNEX_ROLES = {"coordinating lead author": "cl", "lead author": "la", "fellow": "fl",
               "contributing author": "ca", "review editor": "re", "chair": "cs", "co-chair": "cs"}


def _annex_role(lines: list[str]) -> tuple[list[str], int]:
    """The role(s) among a person's lines: 'Lead Author', a cut-off 'Lead Autho',
    'Coordinating Lead Author / Lead Author' (NXS), or a role run into the
    affiliation ('Review EditorETH Zurich'). ([codes], line index)"""
    for i, line in enumerate(lines):
        codes = []
        for part in line.split("/"):
            low = part.lower().strip()
            for text, code in sorted(ANNEX_ROLES.items(), key=lambda kv: -len(kv[0])):
                if low == text or (len(low) >= 6 and text.startswith(low) and len(low) >= len(text) - 2) \
                        or (low.startswith(text) and part.strip()[len(text):len(text) + 1].isupper()):
                    codes.append(code)
                    break
        if codes and (len(codes) == len(line.split("/")) or len(codes) == 1 and "/" not in line):
            return codes, i
    return [], -1


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


def _orcid(value: str) -> str:
    """The ORCID iD in a spreadsheet value ("0000-0002-1825-0097", or as a
    link), if its form and its check digit (ISO 7064 MOD 11-2) are right;
    else ''."""
    m = re.fullmatch(r"(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", value.strip())
    if not m:
        return ""
    digits = m.group(1).replace("-", "")
    total = 0
    for d in digits[:15]:
        total = (total + int(d)) * 2
    check = (12 - total % 11) % 11
    return m.group(1) if digits[15] == ("X" if check == 10 else str(check)) else ""


def build_people(rep: Report, doc: pymupdf.Document, cfg: dict, countries: Countries,
                 log: Log) -> list[Person]:
    import csv

    people: list[Person] = []
    la_or_fl: set[tuple[str, int]] = set()      # (name, chapter) under "LEAD AUTHORS AND FELLOWS"

    def find(name: str, source: str = "") -> Person | None:
        found = [(p, name_match(p.name, name)) for p in people]
        for p, how in found:
            if how == "same":
                p.seen.append(name)
                return p
        for p, how in found:
            if how == "close":
                if source:
                    log.add("names", f"'{p.name}' (chapter or title page) is '{name}' in {source}")
                p.seen.append(name)
                return p
        return None

    # Chapter pages: the authors of each chapter, by role.
    for ch in rep.chapters:
        surnames = _citation_surnames(ch.cite_as)
        for role, entries in ch.authors.items():
            if role == "la+fl":
                role = "la"
                for entry in entries:
                    la_or_fl.add((re.sub(r"\s*\(.*$", "", entry).strip(), ch.number))
            if role not in ROLE_NAMES:
                continue
            for entry in entries:
                # The last (...) holds the countries, which may hold brackets
                # themselves: "Mahdi Kolahi (Iran (Islamic Republic of))" (APA).
                m = re.match(r"^(.*?)\s*\(((?:[^()]|\([^()]*\))*)\)\s*$", entry)
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

    # Co-chairs and the technical support unit: the title page.
    for name, aff, role in rep.editors:
        p = find(name)
        if p is None:
            first, last = _split_name(name, rep.editor_surnames)
            p = Person(name, first, last)
            people.append(p)
        country = aff.rsplit(",", 1)[-1] if "," in aff else ""
        for c in countries.codes(country, f"title page, {name}"):
            if c not in p.countries:
                p.countries.append(c)
        if (role, 0) not in p.roles:
            p.roles.append((role, 0))

    # The management committee: page 2, which gives names only.
    for name in rep.committee:
        p = find(name)
        if p is None:
            first, last = _split_name(name, [])
            p = Person(name, first, last)
            people.append(p)
        if ("mc", 0) not in p.roles:
            p.roles.append(("mc", 0))

    # The author annex: checks names, roles and countries; adds full names.
    # The annex listing authors (Annex III in BBA, IV in IAS).
    annex, pages = next(((k, v) for k, v in rep.annex_pages.items() if "LIST OF AUTHORS" in k), ("", []))
    words = annex.split()[:2]
    annex = f"Annex {words[1]}" if len(words) == 2 else "Annex"        # "Annex III", "Annex II"
    for a in read_annex_people(doc, pages):
        codes, i = _annex_role(a["lines"])
        code = codes[0] if codes else ""
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
        elif codes == ["la"] and a["lines"][i].strip().lower() != "lead author":
            log.add("people", f"{where}: role printed as '{a['lines'][i]}'")
        p = find(a["name"], annex)
        if p is None:
            log.add("people", f"{where} ({ROLE_NAMES.get(code, '?')}, {a['group']}): not on the "
                              f"chapter pages; left out")
            continue
        p.full_name = a["name"]
        if codes == ["fl"] and chapter and ("la", chapter) in p.roles and \
                any(name_match(n, p.name) for n, c in la_or_fl if c == chapter):
            # Under "Lead authors and fellows" on the chapter page; a fellow here.
            p.roles[p.roles.index(("la", chapter))] = ("fl", chapter)
            log.add("people", f"{p.name}: listed under 'Lead authors and fellows' in chapter {chapter}; "
                              f"a fellow in {annex}")
        if codes and chapter and not any((c, chapter) in p.roles for c in codes):
            log.add("people", f"{p.name}: {' / '.join(ROLE_NAMES[c] for c in codes)} of chapter {chapter} "
                              f"in {annex}, but "
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
        # The two lists name chapters differently ("regional_assessments_chapter 3"
        # in the experts list, "africa_chapter3" in the chapters list): compare the numbers.
        numbers = {m.group(1) for c in chapters if (m := re.search(r"chapter\s?(\d+)$", c))}
        # An ORCID iD given to two different people (not two spellings of one
        # name) is wrong for at least one of them: left out for both.
        holders: dict[str, list[str]] = {}
        for row in rows:
            iid = _orcid(row["ORCID Identifier"])
            name = f"{row['First Name'].strip()} {row['Last name'].strip()}"
            if iid and not any(name_match(name, h) for h in holders.get(iid, [])):
                holders.setdefault(iid, []).append(name)
        shared = {iid: names for iid, names in holders.items() if len(names) > 1}
        for iid, names in shared.items():
            log.add("people", f"experts list: ORCID {iid} is given to {' and '.join(names)}; "
                              f"left out for both, the experts are kept")
        committee_rows: list[str] = []
        for row in rows:
            code = row["Chapter"].strip()
            n = re.search(r"chapter\s?(\d+)$", code)
            if chapters and "chapter" in code.lower() and code not in chapters \
                    and not (n and n.group(1) in numbers):
                log.add("people", f"experts list: chapter '{code}' ({row['First Name']} {row['Last name']}) "
                                  f"is not a chapter of {cfg['code']} in {Path(cfg['chapters_csv']).name}")
            full = f"{row['First Name'].strip()} {row['Last name'].strip()}"
            role = _CSV_ROLES.get(row["Role"].strip().lower())
            chm = re.search(r"chapter\s*(\d+)", row["Chapter"], re.IGNORECASE)
            p = find(full, "the experts list")
            if role == "mc":
                # The management committee: page 2 and the experts list together.
                committee_rows.append(full)
                if p is None:
                    p = Person(full, row["First Name"].strip(), row["Last name"].strip())
                    people.append(p)
                if not any(name_match(n, full) for n in rep.committee):
                    log.add("people", f"experts list: {full} ({row['Role']}) is not on page 2 of the report; kept")
                if re.match(r"[A-Z]\.", p.name):       # initials only on page 2 (PPA)
                    p.name, p.first, p.last = full, row["First Name"].strip(), row["Last name"].strip()
                if ("mc", 0) not in p.roles:
                    p.roles.append(("mc", 0))
                role = None
            if p is None:
                if role:
                    log.add("people", f"experts list: {full} ({row['Role']}, {row['Chapter']}) is not "
                                      f"on the chapter pages or the title page")
                continue
            orcid = row["ORCID Identifier"].strip()
            iid = _orcid(orcid)
            if iid and iid not in shared:
                p.orcid = iid
            elif orcid and not iid:
                log.add("people", f"experts list: {full}: ORCID '{orcid}' is not valid; the expert is kept "
                                  f"without an ORCID iD")
            # What the report does not give, from the experts list; the
            # report's own values always win.
            if not p.countries:
                for nat in (row.get("Nationality 1", ""), row.get("Nationality 2", "")):
                    for c in countries.codes(nat, f"experts list, {full}") if nat.strip() else []:
                        if c not in p.countries:
                            p.countries.append(c)
                if p.countries:
                    log.add("countries", f"{p.name}: no country in the report; {', '.join(p.countries)} "
                                         f"from the experts list")
            if not p.roles and role:
                chapter_no = 0 if role == "cs" else int(chm.group(1)) if chm else None
                if chapter_no is not None and (role == "cs" or rep.chapter(chapter_no)):
                    p.roles.append((role, chapter_no))
                    log.add("people", f"{p.name}: no role in the report; {row['Role']}"
                                      + (f" of chapter {chapter_no}" if chapter_no else "") + " from the experts list")
            # Names: the experts list splits first and last name; use it when it
            # is the same person and the printed name contains its last name.
            # The experts list splits first and last names: use the split when
            # the printed name ends with its last name.
            # Compare whole words: "Horcea-Milcu" and "O’Brien" are one word each.
            words, n = p.name.split(), len(row["Last name"].split())
            if 0 < n < len(words) and _tokens(" ".join(words[-n:])) == _tokens(row["Last name"]):
                p.first, p.last = " ".join(words[:-n]), " ".join(words[-n:])
            if role == "fl" and chm and ("la", int(chm.group(1))) in p.roles and \
                    any(name_match(n, p.name) for n, c in la_or_fl if c == int(chm.group(1))):
                p.roles[p.roles.index(("la", int(chm.group(1))))] = ("fl", int(chm.group(1)))
                log.add("people", f"{p.name}: listed under 'Lead authors and fellows' in chapter "
                                  f"{chm.group(1)}; a fellow in the experts list")
            if role and role != "cs" and chm:
                if (role, int(chm.group(1))) not in p.roles:
                    log.add("people", f"{p.name}: {row['Role']} of chapter {chm.group(1)} in the experts list, "
                                      "but " + ", ".join(f"{ROLE_NAMES[r]} of chapter {c}" if c else ROLE_NAMES[r]
                                                         for r, c in p.roles) + " in the report")
        for name in rep.committee:
            if not any(name_match(name, n) for n in committee_rows):
                log.add("people", f"{name}: on page 2 of the report as management committee, not in the experts list "
                                  f"with that role; kept")
    for p in people:
        fix_typo(p, log)
        if not p.countries:
            log.add("countries", f"{p.name}: no country in the report or the experts list; kept without one")
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
    note: str = ""      # how it was obtained, if the report does not list it as a gap


# Sections a paragraph cites: "(see 5.2.2.3.1)", "see sections 3.2 and 4.1", {4.5.1},
# or a number of three levels or more; not "Table 6.3" or "1.5 million".
_SEE_SECTIONS = re.compile(r"\bsee\s+(?:[Ss]ections?\s+)?(\d{1,2}(?:\.\d{1,2})+(?:(?:\s*[,;]\s*|\s+and\s+)\d{1,2}(?:\.\d{1,2})+)*)")
_DEEP_SECTION = re.compile(r"(?<![\w.])(?<!Table )(?<!Figure )(?<!Box )\d{1,2}(?:\.\d{1,2}){2,4}(?![\w]|\.\d)")


def gaps_from_sections(gaps: list[Gap], rep: Report, cfg: dict, log: Log) -> list[Gap]:
    """Knowledge gaps the report discusses in running text, not as a list (config:
    gap_sections; VA 6.4.4): one per paragraph of each sub-section under the given
    section, after those it lists. The sub-section's title is the category, its
    number and the sections the paragraph cites the traceability; a note says how
    the gap was obtained."""
    root = cfg.get("gap_sections")
    if not root:
        return gaps
    listed = len(gaps)
    chapter = rep.chapter(int(root.split(".")[0]))
    subs = [sec for sec in chapter.sections if sec.number.startswith(root + ".")] if chapter else []
    for sec in subs:
        for para in sec.paras:
            text = clean(resolve_marks(para, chapter.number, rep)).strip()
            if not text:
                continue
            cited = [x for m in _SEE_SECTIONS.finditer(text) for x in re.split(r"\s*[,;]\s*|\s+and\s+", m.group(1))]
            cited += _DEEP_SECTION.findall(text)
            trace = ", ".join(dict.fromkeys([sec.number] + cited))
            gaps.append(Gap(len(gaps) + 1, sec.title, text, trace,
                            note=f"Extracted from section {sec.number} of the report, one gap per paragraph; "
                                 f"the report discusses it in running text and does not list it as a knowledge gap."))
    log.add("knowledge gaps", f"{listed} gaps from the report's list, and {len(gaps) - listed} extracted from "
                              f"sections {root}.x, one per paragraph (each with a note saying so)")
    if not subs:
        log.add("knowledge gaps", f"gap_sections: no sub-sections under {root}")
    return gaps


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
    bullets = False             # the items are bullets (ECA), not numbered (TCA)
    for para in box.legend:
        if para.startswith("## "):
            category = para[3:].strip()
            continue
        # ECA Box SPM 5: one gap per bullet, "• Gaps in our understanding of
        # X: ...", its lead the category, its {marks} the traceability. The box
        # stays an illustration as well.
        mb = re.match(r"^[•▪●]\s*(.*)$", para)
        if mb:
            lead, _, rest = mb.group(1).partition(":")
            if not rest:
                lead, rest = category, mb.group(1)
            marks = re.findall(r"\{([^{}]*)\}", rest)
            desc = re.sub(r"\s*\{[^{}]*\}", "", rest).strip()
            gaps.append(Gap(len(gaps) + 1, lead.strip(), desc, ", ".join(dict.fromkeys(m_.strip() for m_ in marks))))
            bullets = True
            continue
        if bullets and gaps[-1].text and not gaps[-1].text.endswith(_TERMINAL) and not para[:1].isdigit():
            # A bullet going on in the next column.
            marks = re.findall(r"\{([^{}]*)\}", para)
            more = re.sub(r"\s*\{[^{}]*\}", "", para).strip()
            gaps[-1].text = f"{gaps[-1].text} {more}"
            if marks:
                gaps[-1].traceability = ", ".join(dict.fromkeys(x for x in [gaps[-1].traceability] + marks if x))
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
    # Pages with another table or figure: the gap table has ended (SUA: the
    # definitions table of Appendix 3 follows in the same layout).
    others = {il.page for il in rep.illustrations if il.page not in rep.gap_table_pages}
    for pno in range(start, min(start + 6, doc.page_count + 1)):
        if pno > start and pno in others:
            break
        lines = page_lines(pno)
        # Lines continuing a bulleted gap are indented (SUA).
        # Not a sub-heading repeating the white label at the side (BBA: "Gaps
        # characterized by low uptake ...").
        side_labels = [t for x0, y0, t, sp in lines if sp.color == WHITE and x0 < desc_x - 15 and len(t) > 15]
        desc_lines = [(y0, t) for x0, y0, t, sp in lines
                      if desc_x - 4 <= x0 <= desc_x + 22 and not sp.bold
                      and sp.color != WHITE and _near(sp.size, 7.0, 0.3)
                      and not any(t.startswith(label) for label in side_labels)]
        desc = sorted((y0, re.sub(r"^[•▪]\s*", "", t)) for y0, t in desc_lines)
        # Where a bullet or a lettered item starts (GA1: "•", then "a)" ... "g)").
        items = {y0 for y0, t in desc_lines if re.match(r"^(?:[•▪]|[a-z]\)\s)", t)}
        if not desc:
            break
        # Category cells: bold lines left of the descriptions, grouped by position.
        cells: list[list] = []
        for x0, y0, t, sp in sorted((ln for ln in lines if ln[0] < desc_x - 15 and ln[3].bold
                                     and ln[3].color != WHITE), key=lambda ln: ln[1]):
            if t.upper() in ("CATEGORY", "CATEGORIES OF KNOWLEDGE GAPS", "DESCRIPTION", "GAP"):
                continue
            if cells and y0 - cells[-1][1] < 12:
                cells[-1][1] = y0
                cells[-1][2] += " " + t
            else:
                cells.append([y0, y0, t])
        cats = [(y, re.sub(r"^\d+\.\s*", "", t).strip()) for y, _, t in cells]
        # A traceability column to the right (NXS): each "{" starts a row.
        trace = sorted((y0, t) for x0, y0, t, sp in lines if x0 > desc_x + 150 and not sp.bold
                       and _near(sp.size, 7.0, 0.3) and sp.color != WHITE)
        if any(t.startswith("{") for _, t in trace):
            rows: list[list] = []          # [y, traceability, description]
            for y0, t in trace:
                if t.startswith("{") or not rows:
                    rows.append([y0, t, ""])
                else:
                    rows[-1][1] += " " + t
            for y0, t in desc:
                row = [r for r in rows if r[0] <= y0 + 2]
                if row:
                    row[-1][2] = f"{row[-1][2]} {t}".strip()
                elif cur or gaps:
                    cur = f"{cur} {t}".strip()       # a row continued from the previous page
            for y, tr, text in rows:
                above = [c for yc, c in cats if yc <= y + 2]
                if above:
                    category = above[-1]
                if cur and gaps:
                    gaps[-1].text = f"{gaps[-1].text} {cur}".strip()
                    cur = ""
                tm = re.search(r"\{([^{}]*)\}", tr)
                gaps.append(Gap(len(gaps) + 1, category, text.strip(), tm.group(1).strip() if tm else tr.strip("{} ")))
            if cur and gaps:
                gaps[-1].text = f"{gaps[-1].text} {cur}".strip()
                cur = ""
            continue
        if not any("{" in t for _, t in desc):
            # No traceability marks (LDR Table SPM 3): a gap ends where the
            # next line is further down than the spacing of a gap's lines.
            rows: list[list] = []          # [y of first line, y of last line, text]
            for y0, t in desc:
                if rows and (y0 not in items if items else y0 - rows[-1][1] < 13):
                    rows[-1][1] = y0
                    rows[-1][2] = _join(rows[-1][2], t, False, False)
                else:
                    rows.append([y0, y0, t])
            for y, _, text in rows:
                above = [c for yc, c in cats if yc <= y + 2]
                if above:
                    category = above[-1]
                gaps.append(Gap(len(gaps) + 1, category, text.strip(), ""))
            continue
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
    y = re.search(r"\(((?:1[6-9]|20)\d\d[a-z]?|n\.d\.?[a-z]?|in press|forthcoming)"
                  r"|(?<=[.,]\s)((?:1[6-9]|20)\d\d[a-z]?)[.:](?=\s|$)", head, re.IGNORECASE)
    r.year = (y.group(1) or y.group(2)).replace("n.d.", "n.d").rstrip(".") if y else ""
    authors = head[:y.start()] if y else head
    first = re.split(r",|\.\s|\.$|\s\(", authors.strip(), 1)[0]
    r.first_author = _norm(first)
    r.doi = _doi_in(re.sub(r"(doi\.org/)\s+", r"\1", r.text))


def match_zotero(refs: list[Ref], items: list[dict], log: Log) -> None:
    by_doi: dict[str, list[dict]] = {}
    for it in items:
        if it["doi"]:
            by_doi.setdefault(it["doi"].lower(), []).append(it)
    # Titles compared without spaces: older PDFs print "30years", "doi. org" (LDR).
    titles = [(it, _norm(it["title"]).replace(" ", "")) for it in items if len(_norm(it["title"])) >= 8]

    def author_agrees(c: dict, first: str) -> bool:
        """The first author among the item's creators, also as the second word
        of a name with a particle ("Aarde, R. Van" is "van Aarde")."""
        return not c["creators"] or any(first.split()[:1] and first.split()[0] in _norm(x).split()
                                        for x in c["creators"] if _norm(x))

    def pick(cands: list[dict], chapter: int) -> dict | None:
        if not cands:
            return None
        own = [c for c in cands if f"Chapter {chapter}" in c["collections"]]
        return (own or cands)[0]

    for r in refs:
        ntext = _norm(r.text).replace(" ", "")
        year = r.year[:4]
        cands = by_doi.get(r.doi.lower(), []) if r.doi else []
        how = "DOI"
        if not cands:
            how = "title"
            cands = [it for it, t in titles if t in ntext and (not year.isdigit() or year in it["date"])]
            if not cands:
                # Titles differ in small ways ("&" for "and", subtitles): compare the start.
                # (34 letters without spaces: what 40 characters were with them)
                cands = [it for it, t in titles if len(t) > 22 and t[:34] in ntext
                         and (not year.isdigit() or year in it["date"])]
                how = "title start"
            if not cands and year.isdigit() and r.first_author:
                # The year one off between the PDF and Zotero (LDR: 2013 / 2014),
                # with the whole title and the first author the same.
                near = {str(int(year) - 1), str(int(year) + 1)}
                cands = [it for it, t in titles if len(t) >= 30 and t in ntext
                         and any(y in it["date"] for y in near) and author_agrees(it, r.first_author)]
                how = "title, year one off"
                if cands:
                    log.add("references", f"chapter {r.chapter} p{r.page}: year {r.year} in the PDF, "
                                          f"{cands[0]['date']} in Zotero ({cands[0]['key']}): {r.text[:100]}")
        # The first author must be among the item's creators (Zotero lists a
        # book's editors before a chapter's authors).
        if cands and r.first_author:
            agree = [c for c in cands if author_agrees(c, r.first_author)]
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
        refs_start = min((r.page for r in chrefs), default=doc.page_count + 1)
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

def glossary_entries(paras: list[Para], rep: Report) -> list[dict]:
    """The glossary of Annex I: [{"term", "definition", "page"}], line by line, as
    the older reports print terms on their own lines with no space between entries.
    A term is the bold at the start of a line, and may wrap; the rest is its definition."""
    pages = next((v for k, v in rep.annex_pages.items() if "GLOSSARY" in k and "REFERENCES" not in k), [])
    entries: list[dict] = []
    in_term = False         # the entry so far is only a term: a bold next line goes on with it
    for p in paras:
        if p.page in pages and p.role == "title" and "REFERENCES" in p.text.upper():
            break           # the glossary's own reference list (NXS)
        if not (p.page in pages and p.role == "small"):
            continue
        for line in p.lines:
            spans = [s for s in line.spans if s.text and not s.sup]
            text = re.sub(r"\s+", " ", "".join(s.text for s in spans)).strip()
            if len(text) < 2:
                continue    # the letters of the alphabet that head the groups
            lead = ""
            for s in spans:
                if not s.bold and s.text.strip():
                    break
                lead += s.text
            lead = re.sub(r"\s+", " ", lead).strip()
            rest = text[len(lead):].strip() if text.startswith(lead) else text
            if text[0] in "•▪·":
                lead, rest = "", text        # a bulleted item inside a definition
            last = entries[-1] if entries else None
            if lead and re.fullmatch(r"(?:Glossary )?References:?", lead, re.IGNORECASE):
                return _clean_glossary(entries)     # AFA's "Glossary References"
            opens = bool(lead) and (in_term or lead[0].isupper() or lead[0].isdigit() or not last
                                    or last["definition"].rstrip().endswith((".", ")", "]")))
            if not opens or len(lead) > 120:
                if last:
                    last["definition"] = _join(last["definition"], text, False, False) if last["definition"] else text
                in_term = False
                continue
            if in_term and last:
                last["term"] = _join(last["term"], lead, False, False)
            else:
                entries.append({"term": lead, "definition": "", "page": p.page})
            # Bold that stops short of the end of the term: "Land sparin" + "g", "(TEK" + ")".
            short = re.match(r"^([a-z)]{1,2})(?=[\s:]|$)\s*:?\s*", rest)
            if short and entries[-1]["term"][-1:].isalpha() or (short and short.group(1) == ")"):
                entries[-1]["term"] += short.group(1)
                rest = rest[short.end():]
            entries[-1]["definition"] = rest.lstrip(": ").strip()
            in_term = not rest
    return _clean_glossary(entries)


def _clean_glossary(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        term = e["term"].strip(" :•·-–").strip()
        if term.count("(") > term.count(")"):
            term += ")"
        if not re.search(r"[A-Za-z]{2}", term):
            continue
        out.append({"term": term, "definition": e["definition"].strip(" :.").strip() + ".", "page": e["page"]})
    return out


def term_key(text: str) -> str:
    """A term for lookup: spaces and hyphens, plurals and -isation/-ization
    aside ("World views" = "Worldviews", "Synergies" = "Synergy",
    "Minimisation" = "Minimization")."""
    def singular(w: str) -> str:
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 4 and w.endswith(("ses", "xes", "ches", "shes")):
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            return w[:-1]
        return w
    words = [singular(w) for w in _norm(text).split()]
    return re.sub(r"is(ation|e|ed|ing)\b", r"iz\1", "".join(words))


def thesaurus_labels(thesaurus: Path) -> set[str]:
    """The English labels (preferred, alternative, hidden) of the IPBES thesaurus, normalized."""
    from rdflib import Graph
    from rdflib.namespace import SKOS

    g = Graph()
    g.parse(thesaurus)
    labels = set()
    for prop in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        for _, lab in g.subject_objects(prop):
            if getattr(lab, "language", None) in (None, "en"):
                labels.add(term_key(str(lab)))
                # The labels were rationalised: "Biomass (ecology)" is also "Biomass".
                labels.add(term_key(re.sub(r"\s*\([^)]*\)", "", str(lab))))
    return labels


def definition_key(text: str) -> str:
    """A definition for comparison: its letters and digits, without its source
    ("(Source: ga1)", "(IPBES core glossary, 2021)") or other closing brackets."""
    text = re.sub(r"(?:\s*\([^()]*\)[\s.]*)+$", "", text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def thesaurus_definitions(thesaurus: Path) -> dict[str, str]:
    """The English definitions of the IPBES thesaurus, by definition_key(): the
    concept's preferred label. A definition is a node with its text in rdf:value."""
    from rdflib import Graph, RDF, URIRef
    from rdflib.namespace import SKOS

    g = Graph()
    g.parse(thesaurus)
    out: dict[str, str] = {}
    for concept, node in g.subject_objects(SKOS.definition):
        values = [node] if not isinstance(node, URIRef) else list(g.objects(node, RDF.value))
        label = next((str(x) for x in g.objects(concept, SKOS.prefLabel)
                      if getattr(x, "language", None) in (None, "en")), "")
        for v in values:
            key = definition_key(str(v))
            if getattr(v, "language", None) in (None, "en") and len(key) >= 20:
                out.setdefault(key, label)
    return out


def in_thesaurus(term: str, labels: set[str]) -> bool:
    """A term, or a plain form of it (no brackets, either of "X or Y", either
    side of "/"), is a label of the thesaurus, as term_key() compares them."""
    variants = {term, re.sub(r"\s*\([^)]*\)", "", term), term.rstrip("s")}
    variants |= {x.strip() for v in list(variants) for x in v.split(" or ")}
    variants |= {x.strip() for v in list(variants) for x in v.split("/")}
    return any(term_key(v) in labels for v in variants if v.strip())


def check_glossary(paras: list[Para], rep: Report, log: Log, thesaurus: Path) -> tuple[int, int]:
    """Report the glossary terms of Annex I that the IPBES thesaurus does not
    have, as an English label or with the same definition. (terms, found)"""
    entries = glossary_entries(paras, rep)
    if not entries or not thesaurus.exists():
        return len(entries), 0
    labels, definitions = thesaurus_labels(thesaurus), thesaurus_definitions(thesaurus)
    found = 0
    for e in entries:
        if in_thesaurus(e["term"], labels) or definition_key(e["definition"]) in definitions:
            found += 1
        else:
            log.add("glossary", f"'{e['term']}' (Annex I) is not in the IPBES thesaurus")
    return len(entries), found


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

_CONFIDENCE = re.compile(r"\((well[ -]established|established but incomplete|unresolved|inconclusive)\)",
                         re.IGNORECASE)
_ILL_MENTION = re.compile(r"\b(Figures?|Tables?|Box(?:es)?)\s+(SPM\s?\.?\s?[A-Z]?\d+|\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_KIND = {"figure": "Figure", "figures": "Figure", "table": "Table", "tables": "Table", "box": "Box", "boxes": "Box"}


def _expand_range(part: str) -> list[str]:
    """A range in a traceability mark: "3.3.4.1–7" and "3.3.4.1–3.3.4.7" are
    3.3.4.1, 3.3.4.2 ... 3.3.4.7 (ECA, also LDR, GA1, AMA). A range over
    different parents ("3.2.1–3.4") is left as printed."""
    # Of figures, tables or boxes too: "Boxes 4.21 – 4.24" (AMA).
    mi = re.fullmatch(r"(Figures?|Tables?|Box(?:es)?)\s+(\d{1,2})\.(\d+)\s*[–-]\s*(?:(\d{1,2})\.)?(\d+)",
                      part.strip(), re.IGNORECASE)
    if mi and (not mi.group(4) or mi.group(4) == mi.group(2)) and 0 < int(mi.group(5)) - int(mi.group(3)) < 40:
        kind = {"f": "Figure", "t": "Table", "b": "Box"}[mi.group(1)[0].lower()]
        return [f"{kind} {mi.group(2)}.{n}" for n in range(int(mi.group(3)), int(mi.group(5)) + 1)]
    m = re.fullmatch(r"(\d{1,2}(?:\.\d+)*)\.(\d+)\s*[–-]\s*(?:(\d{1,2}(?:\.\d+)*)\.)?(\d+)", part.strip())
    if not m:
        return [part]
    head, first, head2, last = m.groups()
    if head2 and head2 != head or not 0 < int(last) - int(first) < 40:
        return [part]
    return [f"{head}.{n}" for n in range(int(first), int(last) + 1)]


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


_NOT_SENTENCE_END = re.compile(r"(?:\b(?:al|e\.g|i\.e|cf|Fig|Figs|No|Nos|vs|pp|Dr|Mr|Ms|Mrs)|\b[A-Z])\.$")


def _sentences(text: str) -> list[str]:
    """A paragraph's full sentences: cut after ".", "!", "?" or a {mark} before a
    capital, not inside brackets or marks, nor after "et al." or an initial."""
    out, pos = [], 0
    for m in re.finditer(r"(?:[.!?][”\"’\]]*|\})\s+(?=[A-Z“\"‘\[])", text):
        head = text[pos:m.end()]
        if head.count("(") > head.count(")") or head.count("{") > head.count("}") \
                or _NOT_SENTENCE_END.search(head.rstrip()):
            continue
        out.append(head.strip())
        pos = m.end()
    if text[pos:].strip():
        out.append(text[pos:].strip())
    return out


# A mark at a sentence's end printed in round brackets by mistake: "... (4.5.5)." (ECA C1).
_ROUND_MARK = re.compile(r"\((\d{1,2}(?:\.\d+)+(?:\s*[,;]\s*\d{1,2}(?:\.\d+)+)*)\)[.\s]*$")


def split_submessages(paragraphs: list[str], where: str = "", log: "Log | None" = None) -> list[tuple[str, str]]:
    """A background message's sub-messages, made of full sentences: each states a
    level of confidence, "(well established)", the minimum, and carries its
    {traceability} marks if it has any. Within a paragraph, a new sub-message starts
    at each sentence with a level; sentences with a mark only, or neither, join the
    current one, or lead into the first level. In a paragraph without levels the
    marks decide. Returns [(sentences, "2.2.1, 1.1.3"), ...]. A mark closed
    by ")" ("{5.6.3, 5.7).") or in round brackets is read too, and reported."""
    out: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        sentences = []                  # (text, has a level, its mark's parts)
        for sentence in _sentences(paragraph):
            parts: list[str] = []
            for m in re.finditer(r"\{((?:[^{}()]|\(\d+\))*)([})])", sentence):
                if m.group(2) == ")" and log:
                    log.add("traceability", f"{where}: '{{{m.group(1)})' closed with ')' in the PDF")
                parts += re.split(r"\s*[,;]\s*", m.group(1))
            round_mark = _ROUND_MARK.search(sentence) if not parts else None
            if round_mark:
                if log:
                    log.add("traceability", f"{where}: '({round_mark.group(1)})' in round brackets in the PDF")
                parts = re.split(r"\s*[,;]\s*", round_mark.group(1))
            sentences.append((sentence, bool(_CONFIDENCE.search(sentence)), [x.strip() for x in parts if x.strip()]))
        has_levels = any(level for _, level, _ in sentences)
        groups: list[list] = []         # [text, has a level, parts]
        for sentence, level, parts in sentences:
            starts = level if has_levels else bool(parts)
            complete = groups and (groups[-1][1] if has_levels else bool(groups[-1][2]))
            if groups and not (starts and complete):
                groups[-1][0] = f"{groups[-1][0]} {sentence}"
                groups[-1][1] = groups[-1][1] or level
                groups[-1][2] += parts
            else:
                groups.append([sentence, level, parts])
        out += [(text, ", ".join(dict.fromkeys(parts))) for text, _, parts in groups]
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
        for part in (x for piece in re.split(r"\s*[,;]\s*", trace) for x in _expand_range(piece)):
            nums = [n.rstrip(".") for n in part.split()]
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
            if plural and re.fullmatch(r"\d{1,2}\.\d{1,2}(?:\.\d{1,2})?", part):
                key = _ill_key(plural, part)
                if key in il_uri:
                    ills.append(il_uri[key])
                    continue
            plural = None
            # "5.2.2.1.a": a lettered paragraph of section 5.2.2.1. Linked to
            # the most specific sub-chapter there is: 5.2.2.1, else 5.2.2 ...
            m = re.match(r"^(\d{1,2}(?:\.\d+)+)(?:\.?[a-z]|\s?\(\d+\))?$", part)
            if m:
                levels = m.group(1).split(".")
                hit = next((".".join(levels[:n]) for n in range(len(levels), 1, -1)
                            if ".".join(levels[:n]) in sch_uri), None)
                if hit:
                    if sch_uri[hit] not in subs:
                        subs.append(sch_uri[hit])
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
            add(s, P("hasConfidenceLevel"), term.replace("well-established", "well established"))

    # Report
    add(report, RDF.type, P("Report"))
    add(report, SKOS.prefLabel, rep.title)
    add(report, SKOS.altLabel, cfg.get("alt_label", ""))
    # The config is the reference for DOIs; what the report prints is only a fallback.
    add(report, P("hasDoi"), (cfg.get("doi") or rep.doi).replace("https://doi.org/", ""))
    add(report, P("year"), str(cfg.get("year", "")))

    # Chapters
    for c in rep.chapters:
        s = ch_uri[c.number]
        add(s, RDF.type, P("Chapter"))
        add(s, P("Report"), report)
        add(s, DCTERMS.identifier, f"{c.number:02d}")
        add(s, SKOS.prefLabel, c.title)
        for doi in cfg.get("chapter_dois", {}).get(str(c.number)) or [c.doi]:
            add(s, P("hasDoi"), doi.replace("https://doi.org/", ""))

    # Sub-chapters
    for c in rep.chapters:
        for sec in c.sections:
            s = sch_uri[sec.number]
            add(s, RDF.type, P("SubChapter"))
            add(s, P("Report"), report)
            add(s, P("Chapter"), ch_uri[c.number])
            # The parent: the nearest section above it that exists. A top
            # section (3.1, or GA1's 2.1.1 under part 2.1) has only its chapter.
            levels = sec.number.split(".")
            parent = next((".".join(levels[:n]) for n in range(len(levels) - 1, 0, -1)
                           if ".".join(levels[:n]) in sch_uri), None)
            if parent:
                add(s, P("SubChapter"), sch_uri[parent])
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
        # The executive summary is the chapter's description, finding by finding.
        if c.executive_summary:
            body = "\n\n".join(" ".join([f"{f.ident}. {f.headline}" if f.ident else f.headline, *f.paras])
                                 for f in c.executive_summary)
            body = resolve_marks(body, c.number, rep, log, f"chapter {c.number} executive summary")
            notes = [f"{n}. {rep.footnotes[(c.number, n)].text}"
                     for n in dict.fromkeys(int(x) for x in FOOTNOTE_MARK.findall(body))]
            if notes:
                body += "\n\nNotes\n" + "\n".join(notes)
            add(ch_uri[c.number], P("hasDescription"), body)

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
        segments = split_submessages([clean(resolve_marks(x, 0, rep)) for x in b.paras], b.ident, log)
        if not any(trace for _, trace in segments):
            log.add("messages", f"{b.ident}: no traceability marks in its text; sub-messages without sections")
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

    def section_letter(message: Message) -> str:
        m = re.match(r"([A-F])\.?\d*$", message.ident) or re.match(r"([A-F])\.\s", message.group)
        return m.group(1) if m else ""

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
        if not idents:
            # SUA: key message A has background messages A.1, A.2 ...
            # (ECA: key message A has A1, A2 ...; SCM: 1 has 1.1, 1.2 ...,
            # whose dot is needed, as PPA numbers both kinds 1, 2 ...)
            own = (rf"{re.escape(k.ident)}\.\d{{1,2}}" if k.ident.isdigit()
                   else rf"{re.escape(k.ident)}\.?\d{{1,2}}")
            idents = [b.ident for b in rep.background_messages if re.fullmatch(own, b.ident)]
        if not idents:
            # The rest print no link: a key message goes with the background
            # messages of its section (A1, or 1 under "A. VALUES OF ...").
            letter = section_letter(k)
            idents = [b.ident for b in rep.background_messages if letter and section_letter(b) == letter]
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
    gaps_linked = 0
    for gap in gaps:
        s = URIRef(f"{base}kg/{rid}-{gap.number}")
        add(s, RDF.type, P("KnowledgeGap"))
        add(s, P("Report"), report)
        add(s, P("hasDescription"), gap.text)
        add(s, P("hasCategory"), gap.category)
        add(s, SKOS.note, gap.note)
        add(s, P("traceability"), gap.traceability)
        # The gap's own identifier, as its URI and the manual conversions
        # (BBA_v01, GA1_v11, NXS_v01): "BBA26-1"; its sections are in traceability.
        add(s, DCTERMS.identifier, f"{rid}-{gap.number}")
        subs, ills = section_links(gap.traceability, f"knowledge gap {gap.number}")
        for u in subs:
            add(s, P("SubChapter"), u)
        # The reports do not relate gaps to background messages: a gap is linked
        # to those with a sub-message tracing to one of the gap's own sections.
        related = [b.ident for b in rep.background_messages if bgm_sch[b.ident].intersection(subs)]
        for ident in related:
            add(s, P("BackgroundMessage"), bgm_uri[ident])
        gaps_linked += bool(related)
    if gaps:
        log.add("messages", f"knowledge gaps: {gaps_linked} of {len(gaps)} linked to background messages "
                            f"through a shared sub-chapter")

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


# The log's lines, by kind: (topic, pattern, group heading, line without what
# the heading says). Lines whose pattern names a chapter are grouped by chapter
# too. Lines matching no pattern go under "Other".
_LOG_GROUPS = [
    ("references", r"^chapter (?P<ch>\d+) (?P<p>p\d+): no Zotero item for: (?P<t>.*)$",
     "No Zotero item", "{p}: {t}"),
    ("references", r"^chapter (?P<ch>\d+) (?P<p>p\d+): year (?P<t>.*)$",
     "Year differs between the PDF and Zotero (matched)", "{p}: year {t}"),
    ("references", r"^chapter (?P<ch>\d+): (?P<t>\S+) matched outside the chapter's Zotero collection (?P<u>.*)$",
     "Matched outside the chapter's Zotero collection", "{t} {u}"),
    ("references", r"^chapter (?P<ch>\d+): one Zotero item (?P<t>.*)$",
     "One Zotero item for two entries", "{t}"),
    ("citations", r"^chapter (?P<ch>\d+): not cited in any section \(perhaps only in the executive summary\): (?P<t>.*)$",
     "Not cited in any section (perhaps only in the executive summary, a table or a figure)", "{t}"),
    ("people", r"^experts list: (?P<t>.*) is not on the chapter pages or the title page$",
     "In the experts list, not in the report", "{t}"),
    ("people", r"^experts list: (?P<t>.*ORCID.*)$", "ORCID iDs in the experts list", "{t}"),
    ("people", r"^experts list: chapter (?P<t>.*)$", "Chapter codes in the experts list", "chapter {t}"),
    ("people", r"^(?P<t>Annex [IVX]+ p\d+, .*): not on the chapter pages; left out$",
     "In the author annex, not on the chapter pages (left out)", "{t}"),
    ("people", r"^(?P<t>Annex [IVX]+ p\d+, .*: no role .*)$", "Author annex: no role", "{t}"),
    ("people", r"^(?P<t>Annex [IVX]+ p\d+, .*: role printed as .*)$", "Author annex: role misprinted", "{t}"),
    ("people", r"^(?P<t>.* in the experts list, but .* in the report)$",
     "Roles: the experts list and the report differ", "{t}"),
    ("people", r"^(?P<t>.* in Annex [IVX]+, but .*)$", "Roles: the author annex and the chapter pages differ", "{t}"),
    ("people", r"^(?P<t>.*): no role in the report; (?P<u>.*) from the experts list$",
     "Roles taken from the experts list", "{t}: {u}"),
    ("people", r"^(?P<t>.*listed under 'Lead authors and fellows'.*)$", "Fellows among the lead authors", "{t}"),
    ("countries", r"^(?P<t>.*): '(?P<n>.*)' is not a country name in ipbes-geo\.rdf; read as (?P<c>\w+)$",
     "Country names not as in ipbes-geo.rdf (read anyway)", "{t}: '{n}' read as {c}"),
    ("countries", r"^(?P<t>.*): '(?P<n>.*)' is an organization, not a country; .*$",
     "Organizations given as countries (the person is kept)", "{t}: '{n}'"),
    ("countries", r"^(?P<t>.*): country '(?P<n>.*)' not found in ipbes-geo\.rdf; .*$",
     "Countries not found in ipbes-geo.rdf (the person is kept)", "{t}: '{n}'"),
    ("countries", r"^(?P<t>.* in Annex [IVX]+, .* on the chapter page)$",
     "Countries: the author annex and the chapter page differ", "{t}"),
    ("countries", r"^(?P<t>.*): no country in the report; (?P<u>.*) from the experts list$",
     "Countries taken from the experts list", "{t}: {u}"),
    ("countries", r"^(?P<t>.*): no country in the report or the experts list; kept without one$",
     "No country in the report or the experts list (the person is kept)", "{t}"),
    ("names", r"^(?P<t>.*\(chapter or title page\) is .*)$", "Name variants across sources", "{t}"),
    ("names", r"^(?P<t>.* written .* sources; .* kept)$", "Spelling corrected by majority", "{t}"),
    ("names", r"^(?P<t>.*: last name .* is not in the name)$", "Last names", "{t}"),
    ("traceability", r"^(?P<t>.*) closed with '\)' in the PDF$", "Marks closed with ')'", "{t}"),
    ("traceability", r"^(?P<t>.*) in round brackets in the PDF$", "Marks in round brackets", "{t}"),
    ("traceability", r"^(?P<t>.*) has no comma between sections in the PDF$", "Sections without a comma", "{t}"),
    ("traceability", r"^(?P<t>.*) is not a figure, table or box of the report$", "Figures, tables or boxes that do not exist", "{t}"),
    ("traceability", r"^(?P<t>.*): section '(?P<n>.*)' does not exist$", "Sections that do not exist", "{t}: {n}"),
    ("traceability", r"^(?P<t>.*) not understood$", "Not understood", "{t}"),
    ("messages", r"^(?P<t>.*): no background messages in its headline$",
     "Key messages without background messages", "{t}"),
    ("messages", r"^(?P<t>.*): text after the last traceability mark became (?P<u>.*)$",
     "Text after the last traceability mark", "{t}: {u}"),
    ("messages", r"^(?P<t>.*): no traceability marks in its text; .*$",
     "Background messages without traceability marks", "{t}"),
    ("structure", r"^chapter (?P<ch>\d+): unnumbered section (?P<t>.*)$", "Unnumbered sections (not in the TTL)",
     "{t}"),
]
_LOG_GROUPS = [(topic, re.compile(rx), head, fmt) for topic, rx, head, fmt in _LOG_GROUPS]


def _log_lines(key: str, items: list[str]) -> list[str]:
    """A topic's lines under "### kind" (and "#### Chapter n") headings."""
    groups: dict[str, dict[str, list[str]]] = {}
    for item in items:
        for topic, rx, head, fmt in _LOG_GROUPS:
            m = rx.match(item) if topic == key else None
            if m:
                fields = m.groupdict()
                chapter = f"Chapter {fields['ch']}" if fields.get("ch") else ""
                groups.setdefault(head, {}).setdefault(chapter, []).append(fmt.format(**fields))
                break
        else:
            groups.setdefault("Other", {}).setdefault("", []).append(item)
    if list(groups) == ["Other"]:
        return [f"- {x}" for x in items] + [""]
    out = []
    for head, by_chapter in sorted(groups.items(), key=lambda kv: kv[0] == "Other"):
        out += [f"### {head} ({sum(len(v) for v in by_chapter.values())})", ""]
        for chapter, lines in sorted(by_chapter.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
            if chapter and len(by_chapter) > 1:
                out += [f"#### {chapter} ({len(lines)})", ""]
            elif chapter:
                lines = [f"{chapter.lower()}: {x}" for x in lines]
            out += [f"- {x}" for x in lines] + [""]
    return out


def write_log(log: Log, rep: Report, counts: dict, path: Path, name: str) -> None:
    out = [f"# {name}: conversion log", "",
           "What the conversion could not read or match, and where the sources disagree.",
           "Each line names its source; nothing here was guessed silently.", "",
           "| Item | Count |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in counts.items()] + [""]
    # The references last: their lists are by far the longest.
    topics = {
        "people": "People: roles and presence across the chapter pages, the author annex and the experts list",
        "names": "Names: the same person written differently",
        "countries": "Countries",
        "traceability": "Traceability marks",
        "messages": "Messages",
        "knowledge gaps": "Knowledge gaps",
        "footnotes": "Footnotes",
        "glossary": "Glossary terms (Annex I) not in the IPBES thesaurus (ipbes-thesaurus.rdf)",
        "structure": "Structure",
    }
    references = {
        "citations": "References not found cited in a section",
        "references": "References and Zotero",
    }

    def add_topics(topics: dict[str, str]) -> None:
        for key, title in topics.items():
            items = log.items.get(key, [])
            if items:
                out.extend([f"## {title} ({len(items)})", ""] + _log_lines(key, items))
    add_topics(topics)
    if rep.problems:
        out += ["## Reading the PDF", ""] + [f"- {p}" for p in rep.problems] + [""]
    add_topics(references)
    path.write_text("\n".join(out), encoding="utf-8")


def convert(assessment: str, config: Path = DEFAULT_CONFIG, out: Path | None = None,
            refresh_zotero: bool = False) -> dict:
    cfg, onto = load_config(config, assessment)
    out = out or DEFAULT_OUT / assessment
    pdf = ROOT / cfg["pdf"]
    doc = pymupdf.open(pdf)
    paras = build_paras(read_lines(doc))
    rep = parse(doc, paras)
    # Titles corrected by hand, where the report misprints them.
    for number, title in cfg.get("chapter_titles", {}).items():
        rep.chapter(int(number)).title = title
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{assessment}.md").write_text(to_markdown(rep, pdf.name), encoding="utf-8")

    log = Log()
    for number in cfg.get("chapter_titles", {}):
        log.add("structure", f"chapter {number}: title corrected by hand in the config")
    countries = Countries(log)
    people = build_people(rep, doc, cfg, countries, log)
    gaps = gaps_from_sections(read_gaps(doc, rep, log), rep, cfg, log)
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
