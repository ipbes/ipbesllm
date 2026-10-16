"""Extract retrieval chunks from a PDF with PyMuPDF.

Each page's text is cleaned before chunking (the cleaning follows
the sentence index used for checking quotes):

  * running headers and footers are removed: lines that recur at the top or
    bottom of many pages of the document ("THE GLOBAL ASSESSMENT REPORT ON
    BIODIVERSITY ...", chapter titles), and bare page numbers;
  * words split at a line end are joined ("biodi-\\nversity"), unless the
    document uses the hyphenated form elsewhere or never the joined form
    ("decision-making" keeps its hyphen);
  * lines are joined into running text.

Pages that are reference lists get section "references", every other page
"body": at least 15 years and 3 DOIs or URLs (as in ipbes_clean_v2.py), or,
for older reports whose references have no DOIs (SCM, PPA), at least 8
lines starting with an author entry ("Tiebout, H. M.") and 8 years;
retrieval leaves the references out unless a question asks for them.

Chunks are made of whole sentences, up to `chunk_size` characters with about
`overlap` characters of overlap, and stay within one page, so a chunk's page
is exact. Each chunk also has the printed page number, from the offset
between printed and PDF page numbers (reports have front matter: GA1's
printed page = PDF page - 63), and the SHA-256 of the PDF, to show which
version of the file it came from.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pymupdf

# A sentence ends at . ! or ?, maybe followed by closing quotes or a
# bracket, then a space and a capital, opening quote or bracket.
SENTENCE_END = re.compile(r'(?<=[.!?])["”’)]*\s+(?=[A-Z“"(\[])')
PAGE_NUMBER = re.compile(r"\d{1,4}")
# Lines at the top or bottom of a page that may be running headers/footers.
EDGE_LINES = 3

# Printed page minus PDF page, for files where automatic detection fails or
# is wrong; e.g. {"nex_ch5.pdf": 358}. The indexer reports, for every file,
# the offset used and how it was found.
PAGE_OFFSETS: dict[str, int] = {}

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_LINK = re.compile(r"doi|https?://", re.IGNORECASE)
# A line starting a reference: "Surname, A. B." or "Two Names, A.".
_ENTRY = re.compile(r"^[A-Z][A-Za-z'’\-]+(?: [A-Z][A-Za-z'’\-]+)?,\s+(?:[A-Z]\.\s?-?){1,4}",
                    re.MULTILINE)


def is_reference_page(text: str) -> bool:
    """A page of a reference list (see the module docstring)."""
    years = len(_YEAR.findall(text))
    return ((years >= 15 and len(_LINK.findall(text)) >= 3)
            or (years >= 8 and len(_ENTRY.findall(text)) >= 8))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _edge_key(line: str) -> str:
    """A header/footer line with its numbers masked, so "CHAPTER 2 ... 45"
    and "CHAPTER 2 ... 46" count as the same running header."""
    return re.sub(r"\d+", "#", line.strip().casefold())


def _printed_page_offset(pages: list[list[str]]) -> int | None:
    """Printed page number minus PDF page number, voted on by the bare
    numbers in each page's first and last lines; None when unclear."""
    votes = Counter()
    for number, lines in enumerate(pages, 1):
        for line in lines[:2] + lines[-2:]:
            if PAGE_NUMBER.fullmatch(line.strip()):
                votes[int(line) - number] += 1
    if not votes:
        return None
    offset, count = votes.most_common(1)[0]
    return offset if count >= 5 and count / sum(votes.values()) >= 0.6 else None


def page_offset(pages: list[list[str]], file_name: str) -> tuple[int | None, str]:
    """(offset, how it was found): "auto", "given", "unknown", or "given, auto-detected N: CHECK"
    when PAGE_OFFSETS disagrees with the pages."""
    auto = _printed_page_offset(pages)
    given = PAGE_OFFSETS.get(file_name)
    if given is None:
        return auto, "auto" if auto is not None else "unknown"
    if auto is None or auto == given:
        return given, "given"
    return given, f"given, auto-detected {auto}: CHECK"


def _running_lines(pages: list[list[str]]) -> set[str]:
    """Keys of lines that recur at the top or bottom of many pages."""
    counts = Counter()
    for lines in pages:
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_edge_key(line) for line in edges if line.strip()})
    threshold = max(3, len(pages) // 50)
    return {key for key, n in counts.items() if n >= threshold}


def _hyphen_rule(pages: list[list[str]]):
    """A function joining words split at a line end: "biodi-" + "versity"
    becomes "biodiversity", but compounds keep their hyphen: the hyphenated
    form occurs elsewhere in the document, or the joined form never does."""
    text = "\n".join("\n".join(lines) for lines in pages)
    whole = text.replace("-\n", "\x00")     # line-end splits do not count
    counts = Counter(w.casefold() for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", whole))

    def join(match: re.Match) -> str:
        first, second = match.group(1), match.group(2)
        if second[0].isupper():             # "Science-\nPolicy", names
            return f"{first}-{second}"
        hyphenated = f"{first}-{second}".casefold()
        joined = f"{first}{second}".casefold()
        if counts[hyphenated] > 0 or counts[joined] == 0:
            return f"{first}-{second}"
        return first + second

    return lambda page_text: re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", join, page_text)


def _clean_pages(pages: list[list[str]]) -> list[str]:
    """Each page as running text, without headers, footers and line-end
    hyphenation."""
    running = _running_lines(pages)
    dehyphenate = _hyphen_rule(pages)
    cleaned = []
    for lines in pages:
        last = len(lines) - EDGE_LINES
        kept = [
            line for i, line in enumerate(lines)
            if not ((i < EDGE_LINES or i >= last)
                    and (_edge_key(line) in running
                         or PAGE_NUMBER.fullmatch(line.strip())))
        ]
        text = dehyphenate("\n".join(kept))
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned


def _split_long(sentence: str, size: int) -> list[str]:
    """A sentence longer than `size` (tables, lists), split at spaces."""
    parts, start = [], 0
    while start < len(sentence):
        end = min(start + size, len(sentence))
        if end < len(sentence):
            space = sentence.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        parts.append(sentence[start:end].strip())
        start = end
    return [p for p in parts if p]


def _chunk_page(text: str, size: int, overlap: int) -> list[str]:
    """Whole sentences packed into chunks of at most `size` characters; each
    chunk after the first repeats the last sentences of the previous one, up
    to `overlap` characters."""
    sentences = []
    for sentence in SENTENCE_END.split(text):
        sentences.extend(_split_long(sentence.strip(), size))

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > size:
            chunks.append(" ".join(current))
            carried = []
            for previous in reversed(current):
                if len(" ".join([previous] + carried + [sentence])) > overlap + len(sentence):
                    break
                carried.insert(0, previous)
            # No overlap when it would push this chunk over `size`.
            current = carried if len(" ".join(carried + [sentence])) <= size else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 2000,
    overlap: int = 300,
) -> list[dict]:
    """
    Extract a PDF's text and split it into chunks of whole sentences.

    Each chunk is a dictionary containing:
      - text
      - source_file
      - title         (PDF metadata title, or the file name without extension)
      - page          (PDF page, from 1)
      - printed_page  (page number printed on the page; None if unknown)
      - page_count
      - chunk_id
      - chunk_index   (index of this chunk within its page)
      - chunk_total   (number of chunks on this page)
      - total_chunks  (number of chunks across the whole document)
      - file_sha256   (first 16 hex digits of the PDF's SHA-256)
      - section       ("references" on reference-list pages, else "body")
      - page_offset, page_offset_note  (printed minus PDF page, and how it
        was found; not stored in Chroma, reported by the indexer)

    Raises:
        ValueError: if `overlap` is not strictly less than `chunk_size`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than "
            f"chunk_size ({chunk_size})."
        )

    path = Path(pdf_path)
    with pymupdf.open(path) as document:
        page_count = document.page_count
        title = ((document.metadata or {}).get("title") or "").strip() or path.stem
        pages = [
            [line for line in document.load_page(n).get_text("text").splitlines()
             if line.strip()]
            for n in range(page_count)
        ]

    offset, offset_note = page_offset(pages, path.name)
    references = [is_reference_page("\n".join(lines)) for lines in pages]
    sha = file_sha256(path)[:16]
    per_page = [_chunk_page(text, chunk_size, overlap) if text else []
                for text in _clean_pages(pages)]
    total_chunks = sum(len(c) for c in per_page)

    chunks: list[dict] = []
    for page_number, page_chunks in enumerate(per_page, start=1):
        printed = page_number + offset if offset is not None else None
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "source_file": path.name,
                "title": title,
                "page": page_number,
                "printed_page": printed if printed and printed > 0 else None,
                "page_count": page_count,
                "chunk_id": f"{path.stem}-page-{page_number}-chunk-{chunk_index}",
                "chunk_index": chunk_index,
                "chunk_total": len(page_chunks),
                "total_chunks": total_chunks,
                "file_sha256": sha,
                "section": "references" if references[page_number - 1] else "body",
                "page_offset": offset,
                "page_offset_note": offset_note,
            })
    return chunks
