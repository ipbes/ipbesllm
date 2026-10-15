"""A reference list of the IPBES assessments, for the XML and combined
prompts.

Questions about the assessments as a set ("Which assessments did the
Plenary approve?") are hard for retrieval: the decisions that approve an
assessment are long paragraphs about several things, so they match such a
question poorly. The list is built from the data instead:

  * every assessment in the TTL collection, with its title and year: all of
    them are approved IPBES products;
  * the Plenary's decisions on assessments in the Plenary reports (XML),
    by stage, so that starting an assessment is not mistaken for approving
    it: a scoping process, a scoping report, the undertaking of the
    assessment, and finally its summary for policymakers (SPM). SPM
    approvals are also quoted from the report itself ("The Plenary approved
    the summary for policymakers of ..."), with any conditions.

Both are read once per process; restart the app after re-indexing.
"""
from __future__ import annotations

import re
from functools import lru_cache

from loguru import logger

from rag_utils import chroma_client

# The stages of an assessment, in order, and how decisions word them. The
# first three use the present tense ("Approves"), which only decisions use,
# not the report's accounts of earlier sessions. The SPM approval is also
# taken from the report's own account ("The Plenary approved"), which adds
# any conditions.
STAGES = [
    ("Scoping started",
     r"approves (?:a scoping process|the initiation of scoping"
     r"|the following \w+ scoping processes)"),
    ("Scoping report approved", r"approves the (?:revised )?scoping report"),
    ("Assessment approved to start", r"approves the undertaking of"),
    ("Assessment completed (SPM approved, chapters accepted)",
     r"(?:The Plenary )?approve[sd] the summary for policymakers of"),
]
# A wording runs to the end of its clause (";") or sentence, or over a list
# introduced by ":". Document symbols such as "(IPBES/6/L.3)" have no space
# after their dots.
_TO_END = r"[^;:]*?(?::[^.]*(?:\.(?=\s|$)|$)|;|\.(?=\s|$)|$)"
_STAGE_PATTERNS = [(stage, re.compile(pattern + _TO_END, re.IGNORECASE))
                   for stage, pattern in STAGES]


def _xpath_key(xpath: str) -> list:
    """Sort key putting paragraph[2] before paragraph[10]."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", xpath)]


def _list_items(chunks: list[tuple[str, str]], xpath: str) -> str:
    """The items of a list introduced by the chunk at `xpath` (".../intro/p"
    ending in ":"): the other chunks under the same parent element, in
    document order, e.g. the assessments "the following fast-track
    assessments:" names."""
    parent = xpath.removesuffix("/intro/p")
    items = [text for item_xpath, text in sorted(chunks, key=lambda c: _xpath_key(c[0]))
             if item_xpath.startswith(parent + "/") and item_xpath != xpath]
    return "; ".join(f"({chr(97 + i)}) {text[:300].rstrip(';. ')}"
                     for i, text in enumerate(items))
_DECISION = re.compile(r"\bIPBES-\d+/\d+\b")
# The location lines xml_loader puts before a chunk's text.
_HEADER_LINE = re.compile(r"^(Document|Part of|Date|Section|Subsection|Paragraph): ")


def _body(document: str) -> str:
    """A chunk's text without its location lines, on one line."""
    lines = [line for line in document.splitlines()
             if not _HEADER_LINE.match(line)]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _assessments() -> list[tuple[str, str, str]]:
    """(code, title, year) of each assessment in the TTL collection."""
    metadatas = chroma_client().get_collection("ttl_documents").get(
        include=["metadatas"]
    )["metadatas"]
    found = {}
    for m in metadatas:
        if m.get("assessment") and m.get("title"):
            found.setdefault(m["assessment"], (m["title"], m.get("date", "")))
    return sorted(((code, title, year) for code, (title, year) in found.items()),
                  key=lambda item: (item[2], item[0]))


def _decisions() -> tuple[list[tuple[int, int, str, str, str]], set[int]]:
    """([(session, stage index, year, where, wording)], sessions uploaded);
    `where` is the decision, or "report", and the paragraph."""
    collection = chroma_client().get_collection("xml_documents")
    got = collection.get(include=["documents", "metadatas"])
    texts = [_body(d) for d in got["documents"]]
    by_file: dict[str, list[tuple[str, str]]] = {}
    for text, m in zip(texts, got["metadatas"]):
        by_file.setdefault(m["source_file"], []).append((m["xpath"], text))

    found, sessions = set(), set()
    for text, m in zip(texts, got["metadatas"]):
        session = int(m.get("number") or 0)
        sessions.add(session)
        part_of = m.get("part_of") or ""
        decision = _DECISION.search(part_of)
        where = (f"decision {decision.group(0)}" if decision
                 else part_of.split(" › ")[-1] if part_of else "report")
        if m.get("paragraph"):
            where += f", para. {m['paragraph'].rstrip('.')}"
        for index, (_, pattern) in enumerate(_STAGE_PATTERNS):
            for match in pattern.finditer(text):
                wording = match.group(0).rstrip(";").strip()
                if wording.endswith(":") and m["xpath"].endswith("/intro/p"):
                    wording += " " + _list_items(by_file[m["source_file"]],
                                                 m["xpath"])
                found.add((session, index, (m.get("date") or "")[:4], where,
                           wording[0].upper() + wording[1:]))
    return sorted(found), sessions - {0}


@lru_cache(maxsize=1)
def uploaded_session_numbers() -> frozenset[int]:
    """The Plenary sessions whose reports are in the XML collection."""
    metadatas = chroma_client().get_collection("xml_documents").get(
        include=["metadatas"]
    )["metadatas"]
    return frozenset(int(m["number"]) for m in metadatas if m.get("number"))


def uploaded_sessions(sessions: set[int]) -> str:
    """ "IPBES-1 to IPBES-8", or a list when some are missing."""
    ordered = sorted(sessions)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"IPBES-{ordered[0]} to IPBES-{ordered[-1]}"
    return ", ".join(f"IPBES-{n}" for n in ordered)


@lru_cache(maxsize=1)
def catalogue_block() -> str:
    """The reference list as a prompt section ("" if it cannot be built)."""
    try:
        assessments = _assessments()
        decisions, sessions = _decisions()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Assessment reference list unavailable: {exc}")
        return ""

    lines = [
        "Reference list: IPBES assessments",
        "",
        "These assessments are in the assessment collection; all of them are "
        "approved IPBES products. The year is the assessment's publication "
        "year, not the date of a Plenary decision:",
    ]
    lines += [f"- {code}: {title} ({year})" for code, title, year in assessments]
    if decisions:
        lines += [
            "",
            "An assessment passes through these Plenary decisions, in order: "
            + "; ".join(f"({n}) {stage}" for n, (stage, _) in enumerate(STAGES, 1))
            + ". Only the last means the assessment was approved as a "
            "finished product: the Plenary approves its summary for "
            "policymakers (SPM) and accepts its chapters. Approving a scoping "
            "report or the undertaking of an assessment only starts it. "
            "In IPBES procedure, a decision approving the undertaking of an "
            "assessment \"as outlined in\" its scoping report or scoping "
            "document also approves that scoping report, unless an earlier "
            "decision approved it separately.",
            "",
            f"Plenary reports uploaded so far: {uploaded_sessions(sessions)}. "
            f"Only these sessions' decisions are known here. For anything "
            f"decided at another session (e.g. the approval of an assessment "
            f"published later), do not state or guess the session, decision "
            f"or date: say that, based on the Plenary reports uploaded so far "
            f"({uploaded_sessions(sessions)}), it is not recorded.",
            "",
            "Plenary decisions on assessments in the uploaded reports:",
        ]
        for session in sorted({d[0] for d in decisions}):
            entries = [d for d in decisions if d[0] == session]
            lines.append(f"IPBES-{session} ({entries[0][2]}):")
            lines += [f"- [{STAGES[index][0]}] {where}: \"{wording}\""
                      for _, index, _, where, wording in entries]
    return "\n".join(lines)
