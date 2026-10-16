"""Check the quotations and DOIs in an answer against the text the model
was given.

Quotations are how an answer claims "the source says exactly this", so a
quotation that is not in the sources is a red flag. As in
ipbes_clean_v2.py's match(), text is compared on its letters and digits
only, so line breaks, punctuation, markdown and typographic quotes do not
matter, and a second pass ignores in-text citations ("(Díaz et al., 2019)",
"{2.1.3}"), which answers often leave out. A quotation with "..." or "…" is
checked part by part; "[...]" insertions are ignored.

DOIs are checked the same way: an invented DOI looks as authoritative as an
invented quotation.
"""
from __future__ import annotations

import re

# In-text citations: "(Author, 2019)" and traceable accounts "{2.1.3}".
CITATION = re.compile(
    r"\([^()]{0,400}?\b(?:19|20)\d{2}[a-z]?\b[^()]{0,400}?\)|\{[^{}]{1,200}\}"
)
# A quotation long enough to be a claim about the source's wording.
QUOTATION = re.compile(r"[“\"]([^”\"\n]{30,1500})[”\"]")
MIN_PART = 12   # letters and digits in a part worth checking
DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"“”<>|\]\)]+")


def _letters(text: str) -> str:
    return "".join(re.findall(r"[^\W_]", text)).lower()


def check_quotes(answer: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """(quotations found in the sources, quotations not found)."""
    text = " ".join(sources)
    haystack = _letters(text)
    without_citations = _letters(CITATION.sub(" ", text))
    found, missing = [], []
    for match in QUOTATION.finditer(answer):
        quotation = match.group(1).strip()
        parts = [p for p in re.split(r"\.\.\.|…|\[[^\]]*\]", quotation)
                 if len(_letters(p)) >= MIN_PART]
        if not parts:
            continue
        ok = all(_letters(p) in haystack
                 or _letters(CITATION.sub(" ", p)) in without_citations
                 for p in parts)
        (found if ok else missing).append(quotation)
    return found, missing


def check_dois(answer: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """(DOIs in the answer found in the sources, DOIs not found)."""
    known = " ".join(sources).lower()
    found, missing = [], []
    for doi in dict.fromkeys(m.group(0).rstrip(".,;:") for m in DOI.finditer(answer)):
        (found if doi.lower() in known else missing).append(doi)
    return found, missing
