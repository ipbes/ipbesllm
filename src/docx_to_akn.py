"""Convert IPBES Plenary session reports from Word (.docx) to Akoma Ntoso
XML (AKN4UN) in data/xml, the files the app indexes, in the form of the
Indigo reports in data/xml_original.

    PYTHONPATH=src python src/docx_to_akn.py                    # data/docx/*.docx
    PYTHONPATH=src python src/docx_to_akn.py data/docx/2017-5.docx
    PYTHONPATH=src python src/docx_to_akn.py --out <folder>

Files are named <year>-<session>.docx; the XML keeps the name. Old Word
files (.doc) are skipped: open them in Word and save them as .docx first.

Translations are named with the language appended (2024-11_fr.docx: ar,
es, fr, ru, zh, as in [languages] in the settings) and go to
data/xml/<language>/2024-11.xml, outside the folder the app indexes. A
translation is another expression of the English report's work: its date,
session and symbol come from the English report, its expression IRI and
<FRBRlanguage> use the language's code (fra), and <FRBRtranslation> points
to the English expression. Its structure is built on the English form of
the numbers (I., A., 1., (a)), so it has the English report's hierarchy and
element identifiers; <num> shows the number as the translation does
("一、", "أولاً-", "(أ)"). Decisions, annexes, appendices and tables are
recognised from each language's words in the settings. Compare a
translation with the English report using xml_compare_lang.py.

The model follows the AKN4UN guidelines for meeting reports
(https://unsceb-hlcm.github.io/part2/, section 7) and naming convention
(https://unsceb-hlcm.github.io/part1/, section 6), as the Indigo files in
data/xml_original do:

    <akomaNtoso><debateReport name="debateReport">
      <meta>        FRBR identification: /akn/un-ipbes/debatereport/report/plenary/<first day>/<session>
      <preface>     title, session, place and dates, document symbol
      <mainBody>    division (I., II.) > subdivision (A., B.) > paragraph (1., 2.) > subparagraph ((a), (i))
      <attachments> Annex > decision IPBES-5/1 > Annex I to decision IPBES-5/1 > Appendix
                    each attachment: <heading>, <subheading>, <doc name="annexure|decision|appendix">

The values that belong to IPBES (FRBR country and IRIs, the organization in
<references>, the acronym in decision symbols "IPBES-5/1", the document
symbol, Word style names) are in config/akn4un_ipbes.toml. To convert
another body's reports, copy it, change the values and pass --config.

The Word reports use the UN templates: headings are styled CH1-CH4 with the
number typed before a tab ("\\tI.\\tOpening of the session"), paragraphs are
numbered by Word ("Normal_number"), and the annexes start with the
ZZ_Anx_header style. Word's automatic numbers are not in the text; they are
recomputed from numbering.xml (see Numbering).

Compare the result with data/xml_original using xml_compare.py.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import docx
from lxml import etree

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "akn4un_ipbes.toml"

# The organization's settings (config/akn4un_ipbes.toml): metadata, the
# acronym in decision symbols, Word styles. Set by load_config().
CONFIG: dict = {}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def akn(tag: str) -> str:
    return f"{{{AKN_NS}}}{tag}"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

PAGE_BREAK = "\f"
# [text] double_quotes = "straight": “The Future We Want” -> "The Future We Want"
_STRAIGHT_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"'})

# Non-breaking hyphens (U+2011) become "-", so "IPBES-5/1" matches searches.
_CHAR_MAP = str.maketrans({"‑": "-", "­": "", "​": ""})


@dataclass
class Span:
    """A run of text with its formatting, or a footnote (note is set)."""
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    vert: str = ""                      # "sup", "sub" or ""
    note: list | None = None            # footnote: list of Span lists (its paragraphs)


def _on(rpr, tag: str) -> bool:
    """A run property such as <w:b/> or <w:b w:val="0"/>."""
    el = rpr.find(w(tag)) if rpr is not None else None
    return el is not None and el.get(w("val"), "true") not in ("0", "false", "none")


def _inside(el, tags: set[str], stop) -> bool:
    parent = el.getparent()
    while parent is not None and parent is not stop:
        if parent.tag in tags:
            return True
        parent = parent.getparent()
    return False


_SKIP_RUNS = {w("del"), w("moveFrom"), w("txbxContent")}


def spans_of(p_el, footnotes: dict | None = None) -> list[Span]:
    """The text of a Word paragraph, as formatted spans, with its footnotes."""
    spans: list[Span] = []
    for r in p_el.iter(w("r")):
        if _inside(r, _SKIP_RUNS, p_el):
            continue
        rpr = r.find(w("rPr"))
        fmt = dict(bold=_on(rpr, "b"), italic=_on(rpr, "i"), underline=_on(rpr, "u"))
        va = rpr.find(w("vertAlign")) if rpr is not None else None
        if va is not None:
            fmt["vert"] = {"superscript": "sup", "subscript": "sub"}.get(va.get(w("val")), "")
        for child in r:
            tag = child.tag
            if tag == w("t"):
                spans.append(Span(child.text or "", **fmt))
            elif tag == w("tab"):
                spans.append(Span("\t", **fmt))
            elif tag in (w("br"), w("cr")):
                # A page break inside a paragraph starts a new one ("…gaps.
                # [page] Annex III to decision IPBES-4/1"): see read_blocks.
                spans.append(Span(PAGE_BREAK if child.get(w("type")) == "page" else "\n", **fmt))
            elif tag == w("noBreakHyphen"):
                spans.append(Span("-", **fmt))
            elif tag == w("sym"):
                spans.append(Span(chr(int(child.get(w("char"), "20"), 16)) if child.get(w("char"), "").isalnum() else "", **fmt))
            elif tag == w("footnoteReference") and footnotes is not None:
                note = footnotes.get(child.get(w("id")))
                if note:
                    spans.append(Span("", note=note))
    quotes = _STRAIGHT_QUOTES if CONFIG.get("text", {}).get("double_quotes") == "straight" else {}
    for s in spans:
        s.text = s.text.translate(_CHAR_MAP).translate(quotes)
    return spans


def plain(spans: list[Span]) -> str:
    return "".join(s.text for s in spans)


def strip_spans(spans: list[Span], prefix_len: int) -> list[Span]:
    """Drop the first `prefix_len` characters (a typed number), then leading
    and trailing white space."""
    out, skip = [], prefix_len
    for s in spans:
        if skip and s.note is None:
            cut = min(skip, len(s.text))
            skip -= cut
            s = Span(s.text[cut:], s.bold, s.italic, s.underline, s.vert)
        out.append(s)
    # leading / trailing white space
    while out and out[0].note is None and not out[0].text.strip():
        out.pop(0)
    if out and out[0].note is None:
        out[0] = Span(out[0].text.lstrip(), out[0].bold, out[0].italic, out[0].underline, out[0].vert)
    while out and out[-1].note is None and not out[-1].text.strip():
        out.pop()
    if out and out[-1].note is None:
        out[-1] = Span(out[-1].text.rstrip(), out[-1].bold, out[-1].italic, out[-1].underline, out[-1].vert)
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Numbering: Word's automatic paragraph and list numbers
# ---------------------------------------------------------------------------

def _roman(n: int) -> str:
    out = ""
    for value, letters in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
                           (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
                           (5, "v"), (4, "iv"), (1, "i")):
        while n >= value:
            out, n = out + letters, n - value
    return out


def _letter(n: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa (Word repeats the letter)."""
    return chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)


def format_number(n: int, fmt: str) -> str:
    return {
        "lowerLetter": _letter(n), "upperLetter": _letter(n).upper(),
        "lowerRoman": _roman(n), "upperRoman": _roman(n).upper(),
        "bullet": "", "none": "",
    }.get(fmt, str(n))


# Numbers in the translations. The structure is built on the English form
# of a number (I., A., 1., (a)), so that a translation has the hierarchy and
# element identifiers of the English report; <num> shows the number as the
# translation does ("一、", "أولاً-", "(أ)").
_CHINESE_DIGITS = "一二三四五六七八九"
# Abjad order (Word's arabicAbjad): a, b, c, d ... as أ، ب، ج، د ...
_ARABIC_ABJAD = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن",
                 "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
_ARABIC_ALPHA = ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص",
                 "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "هـ", "و", "ي"]
# Section numbers I., II. ... and A., B. ... as Arabic words.
_ARABIC_ORDINALS = ["أولاً", "ثانياً", "ثالثاً", "رابعاً", "خامساً", "سادساً", "سابعاً", "ثامناً",
                    "تاسعاً", "عاشراً", "حادي عشر", "ثاني عشر", "ثالث عشر", "رابع عشر",
                    "خامس عشر", "سادس عشر", "سابع عشر", "ثامن عشر", "تاسع عشر", "عشرون"]
_ARABIC_LETTER_NAMES = ["ألف", "باء", "جيم", "دال", "هاء", "واو", "زاي", "حاء", "طاء", "ياء",
                        "كاف", "لام", "ميم", "نون", "سين", "عين", "فاء", "صاد", "قاف", "راء",
                        "شين", "تاء", "ثاء", "خاء", "ذال", "ضاد"]
_RUSSIAN_LETTERS = "абвгдежзиклмнопрстуфхцчшщэюя"
# Cyrillic letters typed for Latin ones in numbers ("А.", "В.", "С." for A., B., C.).
_HOMOGLYPHS = str.maketrans("АВСЕНКМОРТХаеорсху", "ABCEHKMOPTXaeopcxy")
# Word formats of the translations and the English format they stand for.
_NATIVE_FORMATS = {
    "chineseCounting": "upperRoman", "chineseCountingThousand": "upperRoman",
    "japaneseCounting": "upperRoman", "ideographTraditional": "upperRoman",
    "taiwaneseCounting": "upperRoman", "taiwaneseCountingThousand": "upperRoman",
    "chineseLegalSimplified": "upperRoman",
    "arabicAbjad": "lowerLetter", "arabicAlpha": "lowerLetter",
    "russianLower": "lowerLetter", "russianUpper": "upperLetter",
    "decimalFullWidth": "decimal", "decimalHalfWidth": "decimal",
}
# Punctuation of the translations' numbers, in the English form.
_NUMBER_PUNCT = str.maketrans({"、": ".", "．": ".", "（": "(", "）": ")", "-": ".", "–": "."})


def _chinese(n: int) -> str:
    """1 -> 一, 10 -> 十, 12 -> 十二, 20 -> 二十, 21 -> 二十一 (up to 99)."""
    if n < 10:
        return _CHINESE_DIGITS[n - 1] if n > 0 else "〇"
    tens, ones = divmod(n, 10)
    return (_CHINESE_DIGITS[tens - 1] if tens > 1 else "") + "十" + \
        (_CHINESE_DIGITS[ones - 1] if ones else "")


def _from_chinese(text: str) -> int | None:
    if not text or any(c not in _CHINESE_DIGITS + "十" for c in text):
        return None
    if "十" not in text:
        return _CHINESE_DIGITS.index(text) + 1 if len(text) == 1 else None
    tens, _, ones = text.partition("十")
    t = _CHINESE_DIGITS.index(tens) + 1 if tens else 1
    o = _CHINESE_DIGITS.index(ones) + 1 if ones else 0
    return t * 10 + o


def format_native(n: int, fmt: str) -> str:
    """The number as Word shows it in a translation."""
    if _NATIVE_FORMATS.get(fmt) == "upperRoman":
        return _chinese(n)
    if fmt == "arabicAbjad":
        return _ARABIC_ABJAD[(n - 1) % len(_ARABIC_ABJAD)]
    if fmt == "arabicAlpha":
        return _ARABIC_ALPHA[(n - 1) % len(_ARABIC_ALPHA)]
    if fmt == "russianLower":
        return _RUSSIAN_LETTERS[(n - 1) % len(_RUSSIAN_LETTERS)]
    if fmt == "russianUpper":
        return _RUSSIAN_LETTERS[(n - 1) % len(_RUSSIAN_LETTERS)].upper()
    return format_number(n, fmt)


def _arabic_word(word: str) -> str:
    """A pattern for an Arabic word with its tanwin optional and any alef."""
    out = ""
    for c in word:
        if c == "ً":
            out += "ً?"
        elif c in "أإآا":
            out += "[أإآا]"
        else:
            out += re.escape(c)
    return out


_SEP = r"\s*[-–]\s*"
_AR_ORDINAL = re.compile(
    r"^[\t ]*(" + "|".join(_arabic_word(w) for w in sorted(_ARABIC_ORDINALS, key=len, reverse=True))
    + r")(?:" + _SEP + r"|$)")
_AR_LETTER_NAME = re.compile(
    r"^[\t ]*(" + "|".join(_arabic_word(w) for w in _ARABIC_LETTER_NAMES) + r")(?:" + _SEP + r"|$)")
# (أ), (ب) ...: also typed (ه), (ﻫ), (ى), with zero-width non-joiners.
_AR_LETTER_FORMS = {"هـ": "هـ|ه|ﻫ", "ي": "ي|ى", "أ": "أ|ا"}
_ZW = "[‌‍‎‏]*"
_AR_LIST_LETTER = re.compile(
    r"^[\t ]*(\(" + _ZW + "(" + "|".join(_AR_LETTER_FORMS.get(c, re.escape(c)) for c in _ARABIC_ABJAD)
    + ")" + _ZW + r"\))\s*")


def _abjad_value(letter: str) -> int:
    for n, c in enumerate(_ARABIC_ABJAD, 1):
        if letter == c or letter in _AR_LETTER_FORMS.get(c, "").split("|"):
            return n
    return 0
# Paragraph and list numbers typed as "1- ", "١- " (Arabic-Indic digits), "(1)".
_AR_DECIMAL = re.compile(r"^[\t ]*(([0-9٠-٩]{1,3})\s*[-–])\s*(?=\S)")
_AR_PAREN_DECIMAL = re.compile(r"^[\t ]*(\(([0-9٠-٩]{1,2})\))\s*(?=\S)")
_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ZH_NUMBER = re.compile(r"^[\t ]*([一二三四五六七八九十]{1,3})(?:\s*[、.．]\s*|$)")
_ZH_LIST_NUMBER = re.compile(r"^[\t ]*([(（]([一二三四五六七八九十]{1,3})[)）])\s*(?=\S)")


def _arabic_value(found: str, words: list[str]) -> int:
    key = re.sub("[أإآا]", "ا", found.replace("ً", ""))
    for n, w in enumerate(words, 1):
        if re.sub("[أإآا]", "ا", w.replace("ً", "")) == key:
            return n
    return 0


def native_number(raw: str, alone: bool = False):
    """A number typed in a translation's own form at the start of `raw`:
    (English number, format, number as typed, length of the prefix), else
    None. With `alone`, the text must be the number alone (a section number
    on its own line); otherwise it must be followed by its separator and text."""
    lang = LANG.get("_key", "en")
    if lang == "ar":
        for pattern, words, fmt in ((_AR_ORDINAL, _ARABIC_ORDINALS, "upperRoman"),
                                    (_AR_LETTER_NAME, _ARABIC_LETTER_NAMES, "upperLetter")):
            m = pattern.match(raw)
            if m and (m.end() == len(raw.rstrip())) == alone:
                n = _arabic_value(m.group(1), words)
                if n:
                    num = format_number(n, fmt) + "."
                    return num, fmt, raw[:m.end()].strip(), m.end()
        if alone:
            return None
        m = _AR_LIST_LETTER.match(raw)
        if m and (n := _abjad_value(m.group(2))):
            return f"({_letter(n)})", "lowerLetter", re.sub(_ZW, "", m.group(1)), m.end()
        m = _AR_DECIMAL.match(raw)
        if m:
            return f"{int(m.group(2).translate(_INDIC_DIGITS))}.", "decimal", m.group(1), m.end()
        m = _AR_PAREN_DECIMAL.match(raw)
        if m:                           # (1), (2): the (i), (ii) of the English
            n = int(m.group(2).translate(_INDIC_DIGITS))
            return f"({_roman(n)})", "lowerRoman", m.group(1), m.end()
    elif lang == "zh":
        m = _ZH_NUMBER.match(raw)
        if m and (m.end() == len(raw.rstrip())) == alone:
            n = _from_chinese(m.group(1))
            if n:
                return _roman(n).upper() + ".", "upperRoman", raw[:m.end()].strip(), m.end()
        m = _ZH_LIST_NUMBER.match(raw) if not alone else None
        if m and (n := _from_chinese(m.group(2))):        # (一): the (i) of a list
            return f"({_roman(n)})", "lowerRoman", m.group(1), m.end()
    return None


@dataclass
class Level:
    start: int = 1
    fmt: str = "decimal"
    text: str = "%1."


class Numbering:
    """Recomputes the numbers Word shows for numbered paragraphs.

    Each list (w:num) points to a list definition (w:abstractNum) with a
    format per level. Lists sharing a definition share its counters; a list
    with a w:startOverride restarts them where it is first used. A number
    at one level restarts the levels below it.
    """

    def __init__(self, document):
        self.abstract: dict[str, dict[int, Level]] = {}
        self.nums: dict[str, tuple[str, dict[int, int], dict[int, Level]]] = {}
        self.style_num: dict[str, tuple[str | None, int | None]] = {}
        self.style_parent: dict[str, str] = {}
        self.counters: dict = {}
        self.used: set = set()
        self._read_styles(document)
        try:
            root = document.part.numbering_part.element
        except (NotImplementedError, KeyError, AttributeError):
            return
        self.style_links: dict[str, str] = {}      # abstractNum -> list style it uses
        for a in root.findall(w("abstractNum")):
            levels = {}
            for lvl in a.findall(w("lvl")):
                levels[int(lvl.get(w("ilvl")))] = self._level(lvl)
            self.abstract[a.get(w("abstractNumId"))] = levels
            link = a.find(w("numStyleLink"))
            if link is not None:
                self.style_links[a.get(w("abstractNumId"))] = link.get(w("val"))
        for n in root.findall(w("num")):
            starts, overrides = {}, {}
            for o in n.findall(w("lvlOverride")):
                ilvl = int(o.get(w("ilvl")))
                so = o.find(w("startOverride"))
                if so is not None:
                    starts[ilvl] = int(so.get(w("val")))
                lvl = o.find(w("lvl"))
                if lvl is not None:
                    overrides[ilvl] = self._level(lvl)
            abs_id = n.find(w("abstractNumId")).get(w("val"))
            self.nums[n.get(w("numId"))] = (abs_id, starts, overrides)

    @staticmethod
    def _level(lvl) -> Level:
        def val(tag, default):
            el = lvl.find(w(tag))
            return el.get(w("val")) if el is not None and el.get(w("val")) is not None else default
        return Level(int(val("start", "1")), val("numFmt", "decimal"), val("lvlText", ""))

    def _read_styles(self, document):
        for style in document.styles.element.findall(w("style")):
            sid = style.get(w("styleId"))
            based = style.find(w("basedOn"))
            if based is not None:
                self.style_parent[sid] = based.get(w("val"))
            numpr = style.find(f"{w('pPr')}/{w('numPr')}")
            if numpr is not None:
                num_id = numpr.find(w("numId"))
                ilvl = numpr.find(w("ilvl"))
                self.style_num[sid] = (num_id.get(w("val")) if num_id is not None else None,
                                       int(ilvl.get(w("val"))) if ilvl is not None else None)

    def _resolve_link(self, abs_id: str) -> str:
        """A list definition that only points to a list style
        (w:numStyleLink) takes its levels from the list style's numbering."""
        seen = set()
        while abs_id in self.style_links and abs_id not in seen:
            seen.add(abs_id)
            num_id, _ = self.style_num.get(self.style_links[abs_id], (None, None))
            if num_id not in self.nums:
                break
            abs_id = self.nums[num_id][0]
        return abs_id

    def _style_numpr(self, style_id: str | None):
        seen = set()
        while style_id and style_id not in seen:
            seen.add(style_id)
            if style_id in self.style_num:
                return self.style_num[style_id]
            style_id = self.style_parent.get(style_id)
        return None, None

    def number(self, p_el) -> tuple[str, str, int, str] | None:
        """(number text, format, level, as shown) for a numbered paragraph,
        else None. The text and format are the English ones (I., (a)); "as
        shown" is the number in a translation's own form (一、, (أ)) when it
        differs, else "". Advances the list's counters, so call it once per
        paragraph, in document order."""
        ppr = p_el.find(w("pPr"))
        num_id = ilvl = None
        style_id = None
        if ppr is not None:
            ps = ppr.find(w("pStyle"))
            style_id = ps.get(w("val")) if ps is not None else None
            numpr = ppr.find(w("numPr"))
            if numpr is not None:
                n = numpr.find(w("numId"))
                i = numpr.find(w("ilvl"))
                num_id = n.get(w("val")) if n is not None else None
                ilvl = int(i.get(w("val"))) if i is not None else None
        if num_id is None:
            s_num, s_ilvl = self._style_numpr(style_id)
            num_id = s_num
            ilvl = ilvl if ilvl is not None else s_ilvl
        if not num_id or num_id == "0" or num_id not in self.nums:
            return None
        ilvl = ilvl or 0
        abs_id, starts, overrides = self.nums[num_id]
        abs_id = self._resolve_link(abs_id)
        levels = dict(self.abstract.get(abs_id, {}))
        levels.update(overrides)
        if ilvl not in levels:
            return None
        counts = self.counters.setdefault(abs_id, {})
        first_use = (num_id, ilvl) not in self.used
        self.used.add((num_id, ilvl))
        if ilvl in starts and first_use:
            counts[ilvl] = starts[ilvl]         # the list restarts the numbering
        else:
            counts[ilvl] = counts[ilvl] + 1 if ilvl in counts else levels[ilvl].start
        for deeper in [k for k in counts if k > ilvl]:
            del counts[deeper]
        level = levels[ilvl]
        if level.fmt in ("bullet", "none"):
            return "", level.fmt, ilvl, ""

        def english(lv: Level) -> str:
            """The English format of a level: Chinese counting is I. in
            "一、" but (i) in "(一)" (a list)."""
            fmt = _NATIVE_FORMATS.get(lv.fmt, lv.fmt)
            if fmt == "upperRoman" and lv.fmt != "upperRoman" and re.search(r"[(（]%\d", lv.text):
                return "lowerRoman"
            return fmt

        def filler(native: bool):
            def fill(m):
                k = int(m.group(1)) - 1
                lv = levels.get(k, Level())
                value = counts.get(k, starts.get(k, lv.start))
                if native:
                    return format_native(value, lv.fmt)
                return format_number(value, english(lv))
            return fill
        text = re.sub(r"%(\d)", filler(False), level.text).strip()
        fmt = english(level)
        if LANG.get("_key", "en") == "en":
            return text, fmt, ilvl, ""
        shown = re.sub(r"%(\d)", filler(True), level.text).strip()
        text = text.translate(_NUMBER_PUNCT)
        return text, fmt, ilvl, (shown if shown != text else "")


# ---------------------------------------------------------------------------
# Reading the Word file into a flat list of blocks
# ---------------------------------------------------------------------------

# Word styles, from the settings ([styles]); see load_config().
HEADING_STYLES: dict[str, int] = {}
TITLE_STYLES: set[str] = set()
ANNEX_HEADER_STYLES: set[str] = set()
ANNEX_TITLE_STYLES: set[str] = set()
SKIP_STYLES: set[str] = set()

# A number typed before the text: "I.", "A.", "1.", "(a)", "(i)", "a)".
_TYPED_NUMBER = re.compile(
    r"^[\t ]*(\(?[0-9]{1,3}[.)]|\([a-zA-Z]{1,4}\)|[a-z]{1,2}\)|[IVXLC]{1,6}\.|[A-Z]\.|[0-9]{1,2}\.[0-9]{1,2}\.?)"
    r"(?:\t+| {2,})")
# Draft alternatives in negotiated texts: "[2bis.", "[2bis alt", "3ter."
# (IPBES-1 rules of procedure), followed by a single space.
_BIS_NUMBER = re.compile(r"^[\t ]*(\[?\d{1,3}\s?(?:bis|ter|quater)(?:\s+alt)?\.?)(?=\s)")
_HEADING_NUMBER = re.compile(r"^[\t ]*([IVXLC]{1,6}\.?|[A-Z]\.|[0-9]{1,2}(?:\.[0-9]{1,2})*\.?)\t+")
_ROMAN_ONLY = re.compile(r"^[IVXL]{1,6}\.?$")
# Translations: a section number followed by a space in a heading style.
_HEADING_NUMBER_SPACE = re.compile(r"^[\t ]*([IVXLC]{1,6}\.|[A-Z]\.)[ \u00a0\u3000]+(?=\S)")


@dataclass
class Block:
    kind: str                       # heading, para, table, title, annex_header, annex_title
    spans: list = field(default_factory=list)
    style: str = ""
    level: float = 0                # heading level
    num: str = ""                   # number (Word's or typed), in the English form
    display: str = ""               # the number as a translation shows it, when different
    fmt: str = ""                   # decimal, lowerLetter, lowerRoman, upperRoman, bullet ...
    ilvl: int = 0
    auto: bool = False              # number from Word's numbering
    rows: list = field(default_factory=list)   # table: rows of cells (dict)

    @property
    def text(self) -> str:
        return norm(plain(self.spans))

    @property
    def bold_line(self) -> bool:
        """A short, unnumbered paragraph in bold: a heading typed as text."""
        return (self.kind == "para" and not self.num and len(self.text) < 300
                and any(s.text.strip() for s in self.spans)
                and all(s.bold for s in self.spans if s.note is None and s.text.strip()))


def _footnotes(document, numbering) -> dict:
    """footnote id -> its paragraphs (lists of spans)."""
    notes = {}
    for rel in document.part.rels.values():
        if rel.reltype.endswith("/footnotes"):
            root = etree.fromstring(rel.target_part.blob)
            for fn in root.findall(w("footnote")):
                if fn.get(w("type")) in ("separator", "continuationSeparator", "continuationNotice"):
                    continue
                paras = [strip_spans(spans_of(p), 0) for p in fn.iter(w("p"))]
                paras = [p for p in paras if plain(p).strip()]
                if paras:
                    notes[fn.get(w("id"))] = paras
    return notes


def _cell_blocks(tc, footnotes) -> list[list[Span]]:
    return [s for s in (strip_spans(spans_of(p, footnotes), 0) for p in tc.iter(w("p"))) if plain(s).strip()]


def _table(tbl, footnotes) -> list[list[dict]]:
    """Rows of cells {paras, colspan, rowspan}; vertically merged cells
    become one cell with a rowspan."""
    rows, origin = [], {}           # origin: grid column -> cell that started a vertical merge
    for tr in tbl.findall(w("tr")):
        col, row = 0, []
        trpr = tr.find(w("trPr"))
        gb = trpr.find(w("gridBefore")) if trpr is not None else None
        if gb is not None:
            col += int(gb.get(w("val"), "0"))
        for tc in tr.findall(w("tc")):
            tcpr = tc.find(w("tcPr"))
            span_el = tcpr.find(w("gridSpan")) if tcpr is not None else None
            span = int(span_el.get(w("val"), "1")) if span_el is not None else 1
            vm = tcpr.find(w("vMerge")) if tcpr is not None else None
            if vm is not None and vm.get(w("val"), "continue") == "continue" and col in origin:
                origin[col]["rowspan"] += 1
                col += span
                continue
            cell = {"paras": _cell_blocks(tc, footnotes), "colspan": span, "rowspan": 1}
            if vm is not None:
                origin[col] = cell
            else:
                origin.pop(col, None)
            row.append(cell)
            col += span
        if row:
            rows.append(row)
    return rows


def _body_elements(parent):
    """Paragraphs and tables of the body in order, looking inside content
    controls (w:sdt)."""
    for el in parent:
        if el.tag in (w("p"), w("tbl")):
            yield el
        elif el.tag == w("sdt"):
            content = el.find(w("sdtContent"))
            if content is not None:
                yield from _body_elements(content)


def read_blocks(path: Path) -> list[Block]:
    document = docx.Document(str(path))
    numbering = Numbering(document)
    footnotes = _footnotes(document, numbering)
    styles = {s.style_id: s.name for s in document.styles}
    blocks: list[Block] = []
    for el in _body_elements(document.element.body):
        if el.tag == w("tbl"):
            blocks.append(Block("table", rows=_table(el, footnotes)))
            continue
        ppr = el.find(w("pPr"))
        ps = ppr.find(w("pStyle")) if ppr is not None else None
        style = styles.get(ps.get(w("val")) if ps is not None else "Normal", "Normal")
        number = numbering.number(el)
        if style in SKIP_STYLES:
            continue
        pieces = _split_pages(spans_of(el, footnotes))
        for k, spans in enumerate(pieces):
            b = _paragraph_block(spans, style, number if k == 0 else None)
            if b is not None:
                blocks.append(b)
    return blocks


def _split_pages(spans: list[Span]) -> list[list[Span]]:
    """A paragraph's spans, split where it has a page break."""
    pieces: list[list[Span]] = [[]]
    for s in spans:
        if s.note is None and PAGE_BREAK in s.text:
            parts = s.text.split(PAGE_BREAK)
            for n, part in enumerate(parts):
                if n:
                    pieces.append([])
                if part:
                    pieces[-1].append(Span(part, s.bold, s.italic, s.underline, s.vert))
        else:
            pieces[-1].append(s)
    return [p for p in pieces if plain(p).strip() or any(s.note for s in p)]


def _paragraph_block(spans: list[Span], style: str, number) -> "Block | None":
    """A block for one paragraph (or the part of one after a page break)."""
    raw = plain(spans)
    if not raw.strip() and not any(s.note for s in spans):
        return None
    b = Block("para", style=style)
    if style in HEADING_STYLES:
        b.kind, b.level = "heading", HEADING_STYLES[style]
    elif style in TITLE_STYLES:
        b.kind = "title"
    elif style in ANNEX_HEADER_STYLES:
        b.kind = "annex_header"
    elif style in ANNEX_TITLE_STYLES:
        b.kind = "annex_title"
    prefix = 0
    if number is not None:
        b.num, b.fmt, b.ilvl, b.display = number
        b.auto = True
    elif (native := native_number(raw)) is not None:
        b.num, b.fmt, b.display, prefix = native
    else:
        # Russian reports type Cyrillic look-alikes in numbers ("В." for B.):
        # matched in Latin, shown as typed. The length does not change.
        latin = raw.translate(_HOMOGLYPHS) if LANG.get("_key", "en") == "ru" else raw
        m = (_HEADING_NUMBER if b.kind == "heading" else _TYPED_NUMBER).match(latin) \
            or (_BIS_NUMBER.match(latin) if b.kind == "para" else None)
        if m is None and b.kind == "heading" and LANG.get("_key", "en") != "en":
            m = _HEADING_NUMBER_SPACE.match(latin)  # "C. 选举..." in a heading style
        if m:
            b.num, prefix = m.group(1), m.end()
            b.fmt = _typed_format(b.num)
            typed = raw[m.start(1):m.end(1)]
            b.display = typed if typed != b.num else ""
    b.spans = strip_spans(spans, prefix)
    if b.kind == "annex_header" and not _HEADER.match(b.text):
        b.kind = "para"                 # a table note in the annex header style
    return b if b.text or any(s.note for s in b.spans) else None


def _typed_format(num: str) -> str:
    if re.search(r"\d\s?(?:bis|ter|quater)", num):
        return "decimal"
    core = num.strip("().")
    if core.isdigit() or re.fullmatch(r"[0-9.]+", core):
        return "decimal"
    if re.fullmatch(r"[IVXLC]+", core) and core not in ("C", "L"):
        return "upperRoman"
    if re.fullmatch(r"[ivxlc]+", core) and len(core) > 1:
        return "lowerRoman"
    if re.fullmatch(r"[A-Z]", core):
        return "upperLetter"
    if re.fullmatch(r"[ivxlc]", core):
        return "lowerRoman?"            # (i) or the letter i: decided in context
    return "lowerLetter"


# ---------------------------------------------------------------------------
# Building the document tree
# ---------------------------------------------------------------------------

# Decision symbols ("IPBES-5/1"; the acronym is [document] body in the
# settings), compiled by load_config():
_BODY = "IPBES"
_DECISION: re.Pattern            # "IPBES-5/1: ...", "Decision IPBES-7/1", "Decision IPBES/1/2"
_MISTYPED_DECISION: re.Pattern   # typed differently in a heading: "IPBES 4-4" (IPBES-4)
_ANNEX_TO_DECISION: re.Pattern   # "Annex I to decision IPBES-5/1"
_OLD_DECISION: re.Pattern        # "Decision IPBES/1/2 Next steps ..." (IPBES-1)
_SYMBOL: re.Pattern              # document symbol on the cover: IPBES/5/15
_ANNEX: re.Pattern               # "Annex", "Annex I to decision ..."
_APPENDIX: re.Pattern            # "Appendix"
_TABLE_TITLE: re.Pattern         # "Table 1" (group 1: as typed, group 2: the number)
_HEADER: re.Pattern              # a real header in an annex header style
_DECISION_SYMBOL: re.Pattern     # a decision symbol in any acronym, to normalise it

# The language being converted: its settings ([languages.<key>]) and key.
LANG: dict = {}

# Settings for a configuration without [languages]: English, as before.
_DEFAULT_LANGUAGE = {
    "code": "", "bodies": [], "decision": r"(?:Decision\s+)?{symbol}",
    "decision_old": r"Decision\s+({old_symbol})\s*:?\s*",
    "decision_typo": r"((?:Decision\s+)?)({body})(?:[\s-]?(\d+)[-‑]|\s+(\d+)/)(\d+)\b",
    "annex": r"Annex(?:es)?\b", "appendix": r"Appendi(?:x|ces)\b",
    "annex_to_decision": r"\bto\s+decision\s+({symbol})",
    "header": "Annex|Appendi|Decision|{body}", "table": "Table", "report": "Report of",
    "not_headings": ["Rule ", "Table ", "Figure "], "session": "session", "rule": r"Rule\s+\d+",
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read the organization's settings and set the styles and patterns
    that depend on them (for English; see set_language)."""
    with open(path, "rb") as f:
        CONFIG.clear()
        CONFIG.update(tomllib.load(f))
    styles = CONFIG["styles"]
    HEADING_STYLES.clear()
    HEADING_STYLES.update(styles["headings"])
    for target, key in ((TITLE_STYLES, "titles"), (ANNEX_HEADER_STYLES, "annex_headers"),
                        (ANNEX_TITLE_STYLES, "annex_titles"), (SKIP_STYLES, "skip")):
        target.clear()
        target.update(styles.get(key, []))
    set_language("en")
    return CONFIG


def set_language(key: str):
    """Compile the patterns of a language ([languages.<key>] in the settings)."""
    global _BODY, _DECISION, _MISTYPED_DECISION, _ANNEX_TO_DECISION, _OLD_DECISION, _SYMBOL
    global _ANNEX, _APPENDIX, _TABLE_TITLE, _HEADER, _DECISION_SYMBOL
    lang = dict(_DEFAULT_LANGUAGE)
    lang.update(CONFIG.get("languages", {}).get(key, {}))
    lang["code"] = lang["code"] or CONFIG["identification"]["language"]
    lang["_key"] = key
    LANG.clear()
    LANG.update(lang)
    body = CONFIG["document"]["body"]
    # The settings' acronym, and the translations' (regular expressions:
    # Arabic writes م.ح.د or م ح د).
    _BODY = "(?:" + "|".join([re.escape(body)] + [b for b in lang["bodies"] if b != body]) + ")"
    # Translations also space the hyphen: "م ح د - 3/1".
    dash = r"[-‑/]" if key == "en" else r"\s?[-‑/]\s?"
    sub = {"symbol": rf"{_BODY}{dash}\d+/\d+(?!\d)", "old_symbol": rf"{_BODY}/\d+/\d+",
           "body": _BODY}

    def compile_(name, anchor=True, flags=re.IGNORECASE):
        pattern = re.sub(r"\{(symbol|old_symbol|body)\}", lambda m: sub[m.group(1)], lang[name])
        return re.compile(("^" if anchor else "") + pattern, flags)
    _DECISION = compile_("decision")
    _MISTYPED_DECISION = compile_("decision_typo")
    _ANNEX_TO_DECISION = compile_("annex_to_decision", anchor=False)
    _OLD_DECISION = compile_("decision_old", flags=0)
    _ANNEX = compile_("annex")
    _APPENDIX = compile_("appendix")
    _HEADER = re.compile(r"\[?\s*(?:" + re.sub(r"\{body\}", lambda m: _BODY, lang["header"]) + ")",
                         re.IGNORECASE)
    table = lang["table"]
    _TABLE_TITLE = re.compile(rf"^((?:{table})\s*([0-9]+[A-Za-z]?))\.?(?:\s|$)")
    _DECISION_SYMBOL = re.compile(rf"({_BODY}){dash}(\d+)/(\d+)")
    _SYMBOL = re.compile(CONFIG["document"]["symbol"])


def _is_report_title(text: str) -> bool:
    """The report's title on the cover ("Report of ...", [languages] report)."""
    return bool(re.match(LANG["report"], text))


def decision_symbol(text: str) -> str:
    """The first decision symbol in `text`, in the settings' acronym:
    "Решение МПБЭУ-11/1" -> "IPBES-11/1"; the old form stays (IPBES/1/2)."""
    m = _DECISION_SYMBOL.search(text)
    if not m:
        return ""
    sep = "/" if m.group(0)[len(m.group(1)):].lstrip()[:1] == "/" else "-"
    return f"{CONFIG['document']['body']}{sep}{m.group(2)}/{m.group(3)}"

# Rank of a list format: lower ranks contain higher ones.
_FMT_RANK = {"decimal": 1, "upperLetter": 1, "lowerLetter": 2, "lowerRoman": 3, "upperRoman": 1, "bullet": 4}


@dataclass
class Node:
    tag: str                        # division, subdivision, paragraph, subparagraph, attachment, root
    num: str = ""                   # in the English form (I., (a)): eId and structure
    display: str = ""               # <num> as the translation shows it, when different
    heading: list = field(default_factory=list)       # spans
    subheading: list = field(default_factory=list)
    level: float = 0                # heading level (hierarchy) or list rank (paragraphs)
    intro: list = field(default_factory=list)         # blocks before children
    children: list = field(default_factory=list)      # Nodes
    wrap: list = field(default_factory=list)          # blocks after children
    kind: str = ""                  # attachment: top, decision, annex, appendix
    decision: str = ""              # attachment: decision symbol (IPBES-5/1)
    report_annex: bool = False      # attachment: a decision that is also an annex of the report (IPBES-1)
    body: "Node | None" = None      # attachment: its document root

    def add_block(self, block):
        (self.wrap if self.children else self.intro).append(block)


class Builder:
    """Places blocks into the tree: attachments > headings > paragraphs >
    subparagraphs."""

    def __init__(self):
        self.root = Node("root")
        self.attachments: list[Node] = []          # open attachments, outermost first
        self.preface: list[Block] = []
        self.started = False
        self._reset_doc(self.root)
        self.pending_subheading = False

    def _reset_doc(self, doc_root: Node):
        self.doc_root = doc_root
        self.hier: list[Node] = []                  # open headings
        self.para: Node | None = None               # open paragraph
        self.items: list[Node] = []                 # open subparagraphs
        self.last_para_num: int | None = None       # number of the last paragraph

    def container(self) -> Node:
        return self.hier[-1] if self.hier else self.doc_root

    # attachments -----------------------------------------------------------
    def open_attachment(self, kind: str, heading: list):
        text = norm(plain(heading)).lstrip("[ ")
        decision = ""
        if kind == "decision":
            decision = decision_symbol(text)
        stack = self.attachments
        # A decision with nothing in it, straight before another: an entry
        # of the list of decisions in a heading style (a translation styles
        # the last "IPBES-2/9: Communication ..." as an annex title). It
        # goes back into the text.
        prev = stack[-1] if stack else None
        if kind == "decision" and prev is not None and prev.kind == "decision" \
                and not prev.subheading and not prev.body.children and not prev.body.intro:
            stack.pop()
            parent = stack[-1].body if stack else self.root
            parent.children.remove(prev)
            parent.add_block(Block("para", spans=prev.heading))
        if kind == "top":
            stack.clear()
        elif kind == "decision":
            while stack and stack[-1].kind != "top":
                stack.pop()
        elif kind == "annex":
            target = _ANNEX_TO_DECISION.search(text)
            # "Annex II" after "Annex I - Decision IPBES/1/1" is the next
            # annex of the report, not an annex of that decision.
            in_decision = any(a.kind == "decision" and not a.report_annex for a in stack)
            if target or in_decision:
                while stack and stack[-1].kind != "decision":
                    stack.pop()
            else:
                stack.clear()
        elif kind == "appendix":
            while stack and stack[-1].kind not in ("annex", "decision"):
                stack.pop()
        att = Node("attachment", heading=heading, kind=kind, decision=decision)
        att.body = Node("root")
        (stack[-1].body if stack else self.root).children.append(att) if stack else self.root.children.append(att)
        stack.append(att)
        self._reset_doc(att.body)
        self.pending_subheading = True

    # headings --------------------------------------------------------------
    def open_heading(self, level: float, num: str, heading: list, display: str = ""):
        self.close_table_section()
        while self.hier and self.hier[-1].level >= level:
            self.hier.pop()
        # Top-level headings are divisions, the levels below subdivisions,
        # as in most of the reference files.
        depth = len(self.hier)
        node = Node("division" if depth == 0 else "subdivision", num=num,
                    heading=heading, level=level, display=display)
        self.container().children.append(node)
        self.hier.append(node)
        self.para, self.items = None, []

    def open_table_section(self, num: str, heading: list, display: str = ""):
        """A table title ("Table 1 / Status of cash contributions ...") is a
        subdivision holding the table, as in the reference files: num
        "Table 1", eId ...__subdvs_Table1__table_1."""
        self.close_table_section()
        top = self.hier[-1] if self.hier else None
        node = Node("subdivision", num=num, heading=heading, kind="table",
                    level=(top.level if top is not None else 0) + 0.25, display=display)
        self.container().children.append(node)
        self.hier.append(node)
        self.para, self.items = None, []

    def close_table_section(self):
        if self.hier and self.hier[-1].kind == "table":
            self.hier.pop()

    # paragraphs ------------------------------------------------------------
    def add_paragraph(self, b: Block):
        self.close_table_section()
        m = re.match(r"\(?(\d+)", b.num)
        self.last_para_num = int(m.group(1)) if m else None
        node = Node("paragraph", num=b.num, intro=[b], display=b.display)
        self.container().children.append(node)
        self.para, self.items = node, []

    def add_item(self, b: Block, rank: float):
        if self.para is None:
            self.close_table_section()
        # A list without a numbered paragraph: its items are paragraphs
        # ("(a)", "(b)" after a heading), siblings at the same level.
        if self.para is not None and 0 < rank <= self.para.level:
            self.para, self.items = None, []
        if self.para is None:
            node = Node("paragraph", num=b.num, intro=[b], level=rank, display=b.display)
            self.container().children.append(node)
            self.para, self.items = node, []
            return
        while self.items and self.items[-1].level >= rank:
            self.items.pop()
        parent = self.items[-1] if self.items else self.para
        node = Node("subparagraph", num=b.num, intro=[b], level=rank, display=b.display)
        parent.children.append(node)
        self.items.append(node)

    def add_text(self, b: Block):
        """An unnumbered block: it continues the open paragraph."""
        if self.para is not None:
            target = self.para
            if self.items:
                target = self.para          # text after a list closes the list
                self.items = []
            target.add_block(b)
        else:
            self.container().add_block(b)


def _list_rank(b: Block, previous_letter: str | None) -> float:
    fmt = b.fmt
    if fmt == "lowerRoman?":
        core = b.num.strip("().")
        fmt = "lowerLetter" if previous_letter and ord(core) == ord(previous_letter) + 1 else "lowerRoman"
        b.fmt = fmt
    rank = _FMT_RANK.get(fmt, 2)
    # The format decides: 1. contains (a) contains (i). Word's list level
    # only orders items of the same format (1. then 1.1), since items from
    # different lists have unrelated levels (agenda "2." at level 3, its
    # "(a)" at level 1).
    return rank + 0.1 * b.ilvl if b.auto else rank


def build(blocks: list[Block]) -> Builder:
    bld = Builder()
    i = 0
    letter_before: str | None = None
    last_letter: str | None = None              # letter of the last "A." heading
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        text = b.text
        title = text.lstrip("[ ")               # "[Appendix", "[[Annex IV": bracketed drafts

        # Front matter: everything before the first heading.
        if not bld.started:
            if b.kind == "heading" and not b.num and _is_report_title(text):
                b.kind, b.style = "title", "BB_Title"     # the title in a heading style
            # The cover's lines in heading styles before the title
            # (translations of IPBES-1: "Première session").
            elif b.kind == "heading" and not b.num and any(
                    x.kind == "heading" and not x.num and _is_report_title(x.text) for x in blocks[i + 1:i + 4]):
                b.kind = "para"
            # Reports without heading styles (Arabic): the first section
            # number typed in bold ("أولاً- افتتاح الدورة", I. Opening ...).
            typed_section = b.kind == "para" and not b.auto and b.fmt == "upperRoman" \
                and _bold_heading(b) is not None
            if b.kind == "heading" or b.kind == "annex_header" or typed_section:
                bld.started = True
            else:
                bld.preface.append(b)
                i += 1
                continue

        if b.kind == "title":                   # a title style after the cover: a heading
            b.kind, b.level = "heading", 1
        heading_like = b.kind in ("heading", "annex_title", "annex_header") or b.bold_line \
            or _annex_to_decision_line(b, title) or _decision_line(b, title)
        # Attachments: annex header, decisions, annexes, appendices.
        m = _MISTYPED_DECISION.match(title) if heading_like and not b.num else None
        if m:                                   # "IPBES 4-4: ..." -> "IPBES-4/4: ..."
            body = CONFIG["document"]["body"]
            typed = m.group(2) if m.group(2).casefold() != body.casefold() else body
            session = next(g for g in m.groups()[2:-1] if g)
            fixed = f"{m.group(1)}{typed}-{session}/{m.group(m.lastindex)}"
            b.spans = [Span(fixed)] + strip_spans(b.spans, len(plain(b.spans)) - len(plain(b.spans).lstrip()) + m.end())
            text = title = b.text
        if b.kind == "annex_header" and _DECISION.match(title):
            b.kind = "heading"                  # a decision in the annex header style
        if b.kind == "annex_header":
            # Inside a decision, an annex or appendix in this style belongs to
            # it (IPBES-11: "Appendix to the scoping report ..." in Annex I);
            # anything else starts a new part of the report.
            in_decision = any(a.kind == "decision" for a in bld.attachments)
            kind = ("appendix" if _APPENDIX.match(title) else
                    "annex" if _ANNEX.match(title) else "top") if in_decision else "top"
            bld.open_attachment(kind, b.spans)
            i += 1
            continue
        # "Annex I" then "Decision IPBES/1/1 Rules of procedure ..." (IPBES-1):
        # the decision is the annex's subheading (see _decision_in_subheading).
        if heading_like and not b.num and bld.pending_subheading and _OLD_DECISION.match(title):
            att = bld.attachments[-1] if bld.attachments else None
            if att is not None and att.kind == "annex" and not att.subheading:
                att.subheading = b.spans
                _decision_in_subheading(att)
                i += 1
                continue
        if heading_like and not b.num and _starts_attachment(title):
            kind = ("decision" if _DECISION.match(title) else
                    "appendix" if _APPENDIX.match(title) else "annex")
            # "Annex" alone, in bold rather than the annex header style
            # (translations without styles): the annex of decisions.
            word = _ANNEX.match(title)
            if kind == "annex" and word and word.end() == len(title.rstrip(" ]")) \
                    and not any(a.kind == "decision" for a in bld.attachments):
                kind = "top"
            bld.open_attachment(kind, b.spans)
            i += 1
            continue
        if heading_like and not b.num and bld.pending_subheading:
            att = bld.attachments[-1] if bld.attachments else None
            if att is not None and not att.subheading:
                att.subheading = b.spans
                _decision_in_subheading(att)
                i += 1
                continue
        bld.pending_subheading = False

        # Headings. An unnumbered heading straight before another heading
        # contains it ("Executive summary" over "A. Rationale").
        # A table title: "Table 1" (num) and its title (heading), in a
        # heading or table title style, in bold, or just before a table.
        tm = _TABLE_TITLE.match(text) if b.kind in ("para", "heading") and not b.num else None
        if tm and len(text) < 250 and not text.endswith(".") and (
                b.kind == "heading" or b.style == "Title_table" or b.bold_line
                or any(x.kind == "table" for x in blocks[i + 1:i + 3])):
            raw = plain(b.spans)
            number = r"\s+\S+?\.?(?=\s|$)" if LANG["_key"] == "en" else r"\s*[0-9]+[A-Za-z]?\.?(?=\s|$|[^\w.])"
            cut = len(raw) - len(raw.lstrip()) + \
                len(re.match(rf"\s*(?:{LANG['table']}){number}", raw).group(0))
            if LANG["_key"] == "en":
                bld.open_table_section(norm(tm.group(1)), strip_spans(b.spans, cut))
            else:                               # "Tableau 1": num "Table 1", as in English
                bld.open_table_section(f"Table {tm.group(2)}", strip_spans(b.spans, cut), norm(tm.group(1)))
            i += 1
            continue

        # A section number alone on its line in a heading style ("VIII"),
        # then the title: one heading, as in decisions.
        alone = _section_alone(text) if not b.num else None
        if b.kind == "heading" and alone and nxt is not None \
                and nxt.kind in ("heading", "para") and not nxt.num and len(nxt.text) < 200:
            bld.open_heading(1, alone[0], nxt.spans, alone[1])
            i += 2
            continue
        if b.kind in ("heading", "annex_title"):
            # The number decides the level when it has one (some reports
            # style "B." as CH3 next to "A." in CH2); the style otherwise.
            core = b.num.strip("().")
            if b.fmt == "upperRoman" and last_letter and len(core) == 1 \
                    and ord(core) == ord(last_letter) + 1:
                b.fmt = "upperLetter"           # "I." after "H." is a letter
            level = {"upperRoman": 1, "upperLetter": 2, "decimal": 3}.get(b.fmt) if b.num else None
            level = level or b.level or 1
            if b.fmt == "upperLetter":
                last_letter = core[-1:]
            elif b.fmt == "upperRoman":
                last_letter = None
            if not b.num and nxt is not None and nxt.kind == "heading":
                level -= 0.5
            bld.open_heading(level, b.num, b.spans, b.display)
            i += 1
            continue
        # A section number alone on its line ("II"), then the title: a
        # heading in decisions.
        if not b.auto and alone and nxt is not None and nxt.kind == "para" \
                and not nxt.num and len(nxt.text) < 200:
            bld.open_heading(1, alone[0], nxt.spans, alone[1])
            i += 2
            continue
        # Headings typed as bold paragraphs, in older reports: numbered
        # ("1.1 Governance structures", "II. Initial ...") or not ("Rule 25").
        bold_heading = _bold_heading(b)
        if bold_heading is not None:
            level, num, spans, display = bold_heading
            if level is None:                   # unnumbered: under the open heading
                top = bld.hier[-1] if bld.hier else None
                level = top.level if top is not None and top.kind == "bold" else \
                    (top.level + 0.5 if top is not None else 1)
            bld.open_heading(level, num, spans, display)
            if num == "" and bld.hier:
                bld.hier[-1].kind = "bold"
            i += 1
            continue

        if b.kind == "table":
            bld.add_text(b)
        elif b.num and b.fmt != "bullet" and _is_paragraph_number(b, bld):
            bld.add_paragraph(b)
        elif b.num or b.fmt == "bullet":
            rank = _list_rank(b, letter_before)
            bld.add_item(b, rank)
        else:
            bld.add_text(b)
        # The last letter of an (a), (b) list: "(i)" after "(h)" is a letter.
        # Text between the items keeps it ("(b) ... [note] (c)").
        if b.fmt == "lowerLetter":
            letter_before = b.num.strip("().")[-1:]
        elif b.num and b.fmt not in ("lowerRoman", "lowerRoman?"):
            letter_before = None
        i += 1
    return bld


_BOLD_TYPED_NUMBER = re.compile(r"^([IVXL]{1,5}|[A-Z]|\d{1,2}(?:\.\d{1,2})*)\.?\s+(?=\S)")


def _bold_heading(b: Block):
    """(level, number, spans, number as shown) when a bold paragraph is a
    heading, else None. Level from the number: I. 1, A. 2, 1. 3, 1.1 4,
    1.1.1 5; None for an unnumbered one."""
    if b.kind != "para" or b.fmt == "bullet":
        return None
    texts = [s for s in b.spans if s.note is None and s.text.strip()]
    # A section number typed in Arabic words ("أولاً -", "ألف -") makes a
    # heading even when the line is not all bold.
    arabic_section = LANG["_key"] == "ar" and not b.auto and b.display \
        and b.fmt in ("upperRoman", "upperLetter")
    # "Rule 25" alone on its line, bold or not.
    rule = not b.num and LANG.get("rule") and re.fullmatch(LANG["rule"], b.text)
    if not texts or not (arabic_section or rule or all(s.bold for s in texts)):
        return None
    text = b.text
    # A section number (I., A.) allows a longer title: translations are
    # longer than English, and some have no heading styles.
    longest = 300 if b.fmt in ("upperRoman", "upperLetter") else 160
    if len(text) > longest or text.endswith((".", ":", ";", ",", "。", "：", "؛", "،")):
        return None
    num, spans, display = b.num, b.spans, b.display
    if not num:
        latin = text.translate(_HOMOGLYPHS) if LANG["_key"] == "ru" else text
        m = _BOLD_TYPED_NUMBER.match(latin)
        if m and not text.startswith(tuple(LANG["not_headings"])):
            num = m.group(1) + ("." if not m.group(1)[-1:].isdigit() or "." not in m.group(1) else "")
            typed = text[m.start(1):m.end(1)]
            display = typed + num[len(m.group(1)):] if typed != m.group(1) else ""
            spans = strip_spans(spans, len(plain(spans)) - len(plain(spans).lstrip()) + m.end())
        else:
            return None, "", spans, ""
    core = num.strip("().")
    if re.fullmatch(r"[IVXL]+", core) and core not in ("L",):
        return 1, num, spans, display
    if re.fullmatch(r"[A-Z]", core):
        return 2, num, spans, display
    if re.fullmatch(r"\d+(?:\.\d+)*", core):
        return 2 + core.count(".") + 1, num, spans, display
    return None


def _section_alone(text: str):
    """A section number alone on its line ("VIII", "七", "سابعاً"):
    (number in the English form, number as shown or ""), else None."""
    if _ROMAN_ONLY.match(text):
        return text, ""
    native = native_number(text, alone=True)
    if native is not None and native[1] == "upperRoman":
        return native[0].rstrip("."), native[2]
    return None


def _decision_in_subheading(att: Node):
    """IPBES-1: "Annex III" with the subheading "Decision IPBES/1/2 Next
    steps ...". The decision goes into the heading ("Annex III - Decision
    IPBES/1/2"), as in the Indigo files, so it is read as a decision."""
    text = norm(plain(att.subheading))
    m = _OLD_DECISION.match(text)
    if att.kind == "decision" or not m:
        return
    att.kind, att.decision = "decision", m.group(1)
    att.report_annex = True             # an annex of the report, not of a decision
    # "Decision IPBES/1/2" as typed (Décision, Решение ...).
    att.heading = [Span(f"{norm(plain(att.heading))} - {text[:m.end(1)]}")]
    att.subheading = strip_spans(att.subheading, len(plain(att.subheading)) - len(plain(att.subheading).lstrip()) + m.end())


def _annex_to_decision_line(b: Block, title: str) -> bool:
    """A line that is only "Annex I to decision IPBES-5/1", in any style (a
    translation has one in a table title style)."""
    if b.kind != "para" or b.num or len(title) > 100 or not _ANNEX.match(title):
        return False
    m = _ANNEX_TO_DECISION.search(title)
    return bool(m and m.end() >= len(title.rstrip(" .]")))


def _decision_line(b: Block, title: str) -> bool:
    """"Decision IPBES-2/2: Multidisciplinary Expert Panel" in any style (a
    translation has decisions in a custom style, bold by the style): the
    word "Decision", the symbol, a colon and a short title. The list of
    decisions at the start of the annex has no "Decision" word, except in
    IPBES-1, whose symbols have the old form (IPBES/1/1): left out."""
    if b.kind != "para" or b.num or len(title) > 250:
        return False
    m = _DECISION.match(title)
    s = _DECISION_SYMBOL.search(title)
    return bool(m and s and s.start() > 0 and "/" not in s.group(0)[len(s.group(1)):].lstrip()[:1]
                and title[s.end():].lstrip().startswith((":", "：")))


def _starts_attachment(text: str) -> bool:
    return bool(_DECISION.match(text) or _ANNEX.match(text) or _APPENDIX.match(text))


def _is_paragraph_number(b: Block, bld: Builder) -> bool:
    """A numbered paragraph ("12."), as opposed to an item of a list: the
    next number of the paragraphs, or the first number after a heading. A
    "1." inside a paragraph starts a list ("The Plenary adopted the
    following agenda: 1. Opening of the session ...")."""
    if b.fmt != "decimal" or b.ilvl > 0:
        return False
    m = re.match(r"[(\[]?(\d+)", b.num)
    if not m:
        return False
    n = int(m.group(1))
    if re.search(r"bis|ter|quater", b.num):
        return True                     # "[2bis." after paragraph 2
    if bld.para is None:
        return True
    # The next item of an open list ("10." after item "9." of an agenda
    # in paragraph 9).
    if bld.items:
        top = re.match(r"\(?(\d+)", bld.items[0].num)
        if top and int(top.group(1)) + 1 == n:
            return False
    if bld.last_para_num is not None and n == bld.last_para_num + 1:
        return True
    return not (n == 1 or bld.items or _introduces_list(bld))


def _introduces_list(bld: Builder) -> bool:
    if bld.para is None or bld.items or bld.para.children:
        return False
    last = bld.para.wrap[-1] if bld.para.wrap else bld.para.intro[-1] if bld.para.intro else None
    return bool(last is not None and getattr(last, "kind", "") == "para" and last.text.endswith(":"))


# ---------------------------------------------------------------------------
# Metadata from the front matter
# ---------------------------------------------------------------------------

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_M = "(" + "|".join(m.capitalize() for m in MONTHS) + ")"
_DATE_RANGE = re.compile(
    rf"(\d{{1,2}})(?:\s+{_M})?(?:\s+(\d{{4}}))?\s*(?:–|-|—|to)\s*\d{{1,2}}\s+{_M}\s+(\d{{4}})")
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth"]


@dataclass
class Info:
    session: str
    date: str               # first day of the session, YYYY-MM-DD
    title: str
    place: str
    symbol: str
    session_label: str = ""         # "Onzième session", as on a translation's cover
    translation: bool = False       # a translation of the English report


def translation_info(bld: Builder, english: Info) -> Info:
    """A translation's metadata: the work's (session, date, symbol) from the
    English report; the title, session line and place from its own cover."""
    lines = []
    for b in bld.preface:
        if b.kind == "table":
            for row in b.rows:
                for cell in row:
                    for p in cell["paras"]:
                        lines += [norm(x) for x in plain(p).split("\n") if norm(x)]
        else:
            lines.append(b.text)
    title = next((b.text for b in bld.preface if b.kind != "table" and b.style == "BB_Title"), "") \
        or next((x for x in lines if _is_report_title(x)), "") or english.title
    label = place = ""
    year = english.date[:4]
    for k, line in enumerate(lines):
        if not (re.search(LANG["session"], line, re.IGNORECASE) and len(line) < 60):
            continue
        # The place and dates, on one or two lines ("... 16 de diciembre" / "de 2024").
        for after in (lines[k + 1:k + 2], lines[k + 1:k + 3]):
            if after and year in " ".join(after):
                label, place = line, " ".join(after)
                break
        if label:
            break
    return Info(english.session, english.date, title, place, english.symbol,
                session_label=label, translation=True)


def document_info(path: Path, bld: Builder, all_blocks: list[Block]) -> Info:
    front = [b for b in bld.preface if b.kind != "table"]
    front_text = " ".join(b.text for b in front)
    tables = " ".join(" ".join(" ".join(plain(p) for p in c["paras"]) for r in b.rows for c in r)
                      for b in bld.preface if b.kind == "table")
    session = ""
    m = re.match(CONFIG["document"]["file_name"], report_name(path))
    if m:
        session = m.group(2)
    else:
        for n, word in enumerate(_ORDINALS, 1):
            if re.search(rf"\b{word} session\b", front_text, re.IGNORECASE):
                session = str(n)
                break
    title = next((b.text for b in front if b.style == "BB_Title"), "")
    if not title:
        title = next((b.text for b in front if _is_report_title(b.text)), "")
    if not title:
        title = CONFIG["document"]["title"].format(ordinal=_ORDINALS[int(session) - 1])
    date, place = "", ""
    early = front_text + " " + " ".join(b.text for b in all_blocks[:60] if b.kind == "para")
    dm = _DATE_RANGE.search(early)
    if dm:
        day, month1, year1, month2, year2 = dm.groups()
        month = month1 or month2
        year = year1 or year2
        date = f"{year}-{MONTHS.index(month.lower()) + 1:02d}-{int(day):02d}"
    elif m:
        date = f"{m.group(1)}-01-01"
    for b in front:
        if b.style == "AA_Title" and _DATE_RANGE.search(b.text):
            place = b.text
    symbol_match = _SYMBOL.search(tables + " " + front_text)
    return Info(session, date, title, place, symbol_match.group(0) if symbol_match else "")


# ---------------------------------------------------------------------------
# Writing Akoma Ntoso
# ---------------------------------------------------------------------------

class Writer:
    def __init__(self, info: Info):
        self.info = info
        ident = CONFIG["identification"]
        self.work = (f"/akn/{ident['country']}/{ident['doc_type']}/{ident['subtype']}/"
                     f"{ident['actor']}/{info.date}/{info.session}")
        self.expr = f"{self.work}/{LANG['code']}@{info.date}"
        # The English expression a translation is made from (AKN4UN 5.1.11).
        self.source = f"{self.work}/{ident['language']}@{info.date}" if info.translation else ""
        self.used_ids: set[str] = set()
        self.note_number = 0

    # identifiers -----------------------------------------------------------
    def eid(self, prefix: str, kind: str, num: str, index: int) -> str:
        """AKN4UN element identifier: <parent>__<kind>_<num>, the number
        without punctuation ("(a)" -> "a", "I." -> "I"), or a running number."""
        token = re.sub(r"[^0-9A-Za-z]", "", num) if num else ""
        base = f"{kind}_{token or ('nn_' + str(index) if kind in ('dvs', 'subdvs') else index)}"
        value = f"{prefix}__{base}" if prefix else base
        candidate, n = value, 2
        while candidate in self.used_ids:
            candidate, n = f"{value}_{n}", n + 1
        self.used_ids.add(candidate)
        return candidate

    # meta ------------------------------------------------------------------
    def meta(self, component: str, title: str, top: bool) -> etree._Element:
        info = self.info
        today = datetime.date.today().isoformat()
        c, org = CONFIG["identification"], CONFIG["organization"]
        author, ext = c["author_href"], c["manifestation_extension"]
        meta = etree.Element(akn("meta"))
        ident = etree.SubElement(meta, akn("identification"), source=c["source"])
        work = etree.SubElement(ident, akn("FRBRWork"))
        etree.SubElement(work, akn("FRBRthis"), value=f"{self.work}/!{component}")
        etree.SubElement(work, akn("FRBRuri"), value=self.work)
        etree.SubElement(work, akn("FRBRalias"), value=title, name="title")
        if top and info.symbol and c.get("symbol_alias", True):
            etree.SubElement(work, akn("FRBRalias"), value=info.symbol, name="symbol")
        etree.SubElement(work, akn("FRBRdate"), date=info.date, name="Generation")
        etree.SubElement(work, akn("FRBRauthor"), href=author)
        etree.SubElement(work, akn("FRBRcountry"), value=c["country"])
        etree.SubElement(work, akn("FRBRsubtype"), value=c["subtype"])
        etree.SubElement(work, akn("FRBRnumber"), value=info.session)
        expr = etree.SubElement(ident, akn("FRBRExpression"))
        etree.SubElement(expr, akn("FRBRthis"), value=f"{self.expr}/!{component}")
        etree.SubElement(expr, akn("FRBRuri"), value=self.expr)
        etree.SubElement(expr, akn("FRBRdate"), date=info.date, name="Generation")
        etree.SubElement(expr, akn("FRBRauthor"), href=author)
        etree.SubElement(expr, akn("FRBRlanguage"), language=LANG["code"])
        if self.source:
            # "by" is required by the schema: the organization in <references>.
            etree.SubElement(expr, akn("FRBRtranslation"), href=f"{self.source}/!{component}",
                             fromLanguage=c["language"], by=c["source"])
        man = etree.SubElement(ident, akn("FRBRManifestation"))
        etree.SubElement(man, akn("FRBRthis"), value=f"{self.expr}/!{component}{ext}")
        etree.SubElement(man, akn("FRBRuri"), value=f"{self.expr}{ext}")
        etree.SubElement(man, akn("FRBRdate"), date=today, name="Generation")
        etree.SubElement(man, akn("FRBRauthor"), href=author)
        if top:
            refs = etree.SubElement(meta, akn("references"), source=c["source"])
            etree.SubElement(refs, akn("TLCOrganization"), eId=org["eId"], href=org["href"],
                             showAs=org["showAs"])
        return meta

    # inline ----------------------------------------------------------------
    def inline(self, parent: etree._Element, spans: list[Span], eid: str):
        """Formatted text and footnotes, appended to `parent`."""
        last = None

        def add_text(text):
            nonlocal last
            text = text.replace("\t", " ").replace("\n", " ")
            if last is None:
                parent.text = (parent.text or "") + text
            else:
                last.tail = (last.tail or "") + text

        for s in _merge(spans):
            if s.note is not None:
                self.note_number += 1
                n = sum(1 for _ in parent.iter(akn("authorialNote"))) + 1
                note_id = f"{eid}__authorialNote_{n}"
                # In <sup>, as in every reference file.
                sup = etree.SubElement(parent, akn("sup"))
                note = etree.SubElement(sup, akn("authorialNote"), marker=str(self.note_number),
                                        placement="bottom", eId=note_id)
                for k, para in enumerate(s.note, 1):
                    p = etree.SubElement(note, akn("p"), eId=f"{note_id}__p_{k}")
                    self.inline(p, para, f"{note_id}__p_{k}")
                last = sup
                continue
            if not (s.bold or s.italic or s.underline or s.vert):
                add_text(s.text)
                continue
            outer = inner = None
            for flag, tag in ((s.bold, "b"), (s.italic, "i"), (s.underline, "u"),
                              (bool(s.vert), s.vert)):
                if flag:
                    el = etree.Element(akn(tag))
                    if inner is None:
                        outer = el
                    else:
                        inner.append(el)
                    inner = el
            inner.text = s.text.replace("\t", " ").replace("\n", " ")
            parent.append(outer)
            last = outer

    def p(self, parent, spans, prefix: str, index: int):
        eid = f"{prefix}__p_{index}" if prefix else f"p_{index}"
        el = etree.SubElement(parent, akn("p"), eId=eid)
        self.inline(el, spans, eid)
        return el

    def blocks(self, parent, blocks: list[Block], prefix: str):
        """Paragraph text and tables into `parent`."""
        p_count = t_count = 0
        for b in blocks:
            if b.kind == "table":
                t_count += 1
                self.table(parent, b, f"{prefix}__table_{t_count}" if prefix else f"table_{t_count}")
            else:
                p_count += 1
                spans = b.spans
                if b.style in ("Title_table", "Title_figure"):
                    spans = [Span(s.text, True, s.italic, s.underline, s.vert, s.note) for s in spans]
                self.p(parent, spans, prefix, p_count)

    def table(self, parent, b: Block, eid: str):
        table = etree.SubElement(parent, akn("table"), eId=eid)
        k = 0
        for row in b.rows:
            tr = etree.SubElement(table, akn("tr"))
            for cell in row:
                attrs = {}
                if cell["colspan"] > 1:
                    attrs["colspan"] = str(cell["colspan"])
                if cell["rowspan"] > 1:
                    attrs["rowspan"] = str(cell["rowspan"])
                td = etree.SubElement(tr, akn("td"), **attrs)
                for para in cell["paras"] or [[]]:
                    k += 1
                    self.p(td, para, eid, k)
            if len(tr) == 0:
                table.remove(tr)

    # structure -------------------------------------------------------------
    def node(self, parent, node: Node, prefix: str, index: int):
        abbr = {"division": "dvs", "subdivision": "subdvs", "paragraph": "para",
                "subparagraph": "subpara"}[node.tag]
        eid = self.eid(prefix, abbr, node.num, index)
        el = etree.SubElement(parent, akn(node.tag), eId=eid)
        if node.num:
            etree.SubElement(el, akn("num")).text = node.display or node.num
        if node.heading:
            self.inline(etree.SubElement(el, akn("heading")), node.heading, eid)
        if node.children:
            if node.intro:
                self.blocks(etree.SubElement(el, akn("intro")), node.intro, f"{eid}__intro")
            for k, child in enumerate(node.children, 1):
                self.node(el, child, eid, k)
            if node.wrap:
                self.blocks(etree.SubElement(el, akn("wrapUp")), node.wrap, f"{eid}__wrapup")
        elif node.intro or node.wrap or node.tag in ("paragraph", "subparagraph"):
            content = etree.SubElement(el, akn("content"))
            self.blocks(content, node.intro + node.wrap, eid)
            if len(content) == 0:
                etree.SubElement(content, akn("p"), eId=f"{eid}__p_1")
        # A heading with nothing under it ("D. Methodological assessment ...",
        # taken up later in the report) is just its number and heading.

    def body(self, parent, root: Node, prefix: str):
        """mainBody of the report or of an attachment; returns the
        attachments found in it."""
        body = etree.SubElement(parent, akn("mainBody"))
        self.blocks(body, root.intro, prefix)
        attachments = []
        for k, child in enumerate((c for c in root.children if c.tag != "attachment"), 1):
            self.node(body, child, prefix, k)
        if root.wrap:
            self.blocks(body, root.wrap, f"{prefix}__wrapup" if prefix else "wrapup")
        attachments = [c for c in root.children if c.tag == "attachment"]
        if len(body) == 0:
            etree.SubElement(body, akn("p"), eId=f"{prefix}__p_1" if prefix else "p_1")
        return attachments

    def attachments(self, parent, atts: list[Node], prefix: str, component: str):
        if not atts:
            return
        container = etree.SubElement(parent, akn("attachments"))
        names = CONFIG["doc_names"]
        per_name: dict[str, int] = {}
        for k, att in enumerate(atts, 1):
            eid = f"{prefix}__att_{k}" if prefix else f"att_{k}"
            self.used_ids.add(eid)
            # <doc name> and component name: annexure_1, annexure_2, appendix_1
            name = names.get(att.kind, names["annex"])
            per_name[name] = per_name.get(name, 0) + 1
            part = f"{name}_{per_name[name]}"
            comp = f"{component}/{part}" if component else part
            el = etree.SubElement(container, akn("attachment"), eId=eid)
            heading = norm(plain(att.heading))
            self.inline(etree.SubElement(el, akn("heading")), att.heading, eid)
            if att.subheading:
                self.inline(etree.SubElement(el, akn("subheading")), att.subheading, eid)
            doc = etree.SubElement(el, akn("doc"), name=name)
            title = " ".join(filter(None, [heading, norm(plain(att.subheading))]))
            doc.append(self.meta(comp, title, top=False))
            inner = self.body(doc, att.body, eid)
            self.attachments(doc, inner, eid, comp)

    def preface(self, parent, bld: Builder):
        info = self.info
        preface = etree.SubElement(parent, akn("preface"))
        if info.symbol:
            p = etree.SubElement(preface, akn("p"), eId="preface__p_1")
            etree.SubElement(p, akn("docNumber")).text = info.symbol
        p = etree.SubElement(preface, akn("p"), eId="preface__p_2")
        etree.SubElement(p, akn("docTitle")).text = info.title
        if info.session:
            p = etree.SubElement(preface, akn("p"), eId="preface__p_3")
            etree.SubElement(p, akn("session"), value=info.session).text = info.session_label or \
                f"{_ORDINALS[int(info.session) - 1].capitalize()} session"
        if info.place:
            p = etree.SubElement(preface, akn("p"), eId="preface__p_4")
            etree.SubElement(p, akn("docDate"), date=info.date).text = info.place

    def document(self, bld: Builder) -> etree._Element:
        root = etree.Element(akn("akomaNtoso"), nsmap={None: AKN_NS})
        report = etree.SubElement(root, akn("debateReport"), name=CONFIG["document"]["name"])
        report.append(self.meta("main", self.info.title, top=True))
        self.preface(report, bld)
        atts = self.body(report, bld.root, "")
        self.attachments(report, atts, "", "")
        return root


def _merge(spans: list[Span]) -> list[Span]:
    """Join neighbouring spans with the same formatting."""
    out: list[Span] = []
    for s in spans:
        if s.note is None and not s.text:
            continue
        if out and s.note is None and out[-1].note is None and \
                (out[-1].bold, out[-1].italic, out[-1].underline, out[-1].vert) == \
                (s.bold, s.italic, s.underline, s.vert):
            out[-1] = Span(out[-1].text + s.text, s.bold, s.italic, s.underline, s.vert)
        else:
            out.append(s)
    # Formatting on white space only is dropped.
    return [Span(s.text, False, False, False, "") if s.note is None and not s.text.strip() else s
            for s in out]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def split_name(path: Path) -> tuple[str, str]:
    """(report, language): "2016-4.docx" -> ("2016-4", "en"),
    "2024-11_fr.docx" -> ("2024-11", "fr"); also for a file saved as
    "2016-4_fr.doc.docx". The languages are those in the settings."""
    stem = path.stem.removesuffix(".doc")
    report, _, lang = stem.rpartition("_")
    if report and lang in CONFIG.get("languages", {}) and lang != "en":
        return report, lang
    return stem, "en"


def report_name(path: Path) -> str:
    """"2016-4.docx" -> "2016-4"; "2024-11_fr.docx" -> "2024-11"."""
    return split_name(path)[0]


def _english_original(path: Path) -> Path | None:
    report = report_name(path)
    for name in (f"{report}.docx", f"{report}.doc.docx"):
        if (path.parent / name).exists():
            return path.parent / name
    return None


def convert(path: Path) -> tuple[etree._Element, Info]:
    if not CONFIG:
        load_config()
    lang = split_name(path)[1]
    english = None
    if lang != "en" and (original := _english_original(path)) is not None:
        set_language("en")
        en_blocks = read_blocks(original)
        english = document_info(original, build(en_blocks), en_blocks)
    set_language(lang)
    try:
        blocks = read_blocks(path)
        bld = build(blocks)
        if english is not None:
            info = translation_info(bld, english)
        else:
            info = document_info(path, bld, blocks)
            info.translation = lang != "en"
        return Writer(info).document(bld), info
    finally:
        set_language("en")


def output_path(out: Path, path: Path) -> Path:
    """<out>/2024-11.xml for an English report, <out>/fr/2024-11.xml for its
    French translation."""
    report, lang = split_name(path)
    return (out / lang if lang != "en" else out) / f"{report}.xml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="Word files (default: data/docx/*)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "xml")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="organization settings (default: config/akn4un_ipbes.toml)")
    args = ap.parse_args(argv)
    load_config(args.config)
    files = args.files or sorted((ROOT / "data" / "docx").glob("*.doc*"))
    args.out.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.suffix.lower() == ".doc":
            print(f"skip {path.name}: old Word format; save it as .docx in Word")
            continue
        root, info = convert(path)
        out = output_path(args.out, path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))
        counts = {t: sum(1 for _ in root.iter(akn(t))) for t in
                  ("division", "paragraph", "subparagraph", "attachment", "table", "authorialNote")}
        print(f"{path.name} -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  "
              f"session {info.session}, {info.date}, {info.symbol or 'no symbol'}  "
              + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    sys.exit(main())
