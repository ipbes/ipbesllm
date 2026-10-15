"""A reference list of the IPBES assessments, for the XML and combined
prompts.

Questions about the assessments as a set ("Which assessments did the
Plenary approve?") are hard for retrieval: the decisions that approve an
assessment are long paragraphs about several things, so they match such a
question poorly. The list is built from the data instead:

  * every assessment in the TTL collection, with its title and year: all of
    them are approved IPBES products;
  * every passage in the Plenary reports (XML) that approves the summary
    for policymakers of an assessment, in a decision ("Approves the summary
    for policymakers of ...") or in the report itself ("The Plenary approved
    the summary for policymakers of ..."), quoted with any conditions.

Both are read once per process; restart the app after re-indexing.
"""
from __future__ import annotations

import re
from functools import lru_cache

from loguru import logger

from rag_utils import chroma_client

# From the approval to the end of its clause (";") or sentence. Document
# symbols such as "(IPBES/6/L.3)" have no space after their dots.
_APPROVAL = re.compile(
    r"(?:The Plenary )?approve[sd] the summary for policymakers of "
    r"[^;]*?(?:;|\.(?=\s|$)|$)",
    re.IGNORECASE,
)
_DECISION = re.compile(r"\bIPBES-\d+/\d+\b")


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


def _approvals() -> tuple[list[tuple[int, str, str, str]], int]:
    """([(session, year, where, wording)], last session covered); `where`
    is the decision or "report" and the paragraph."""
    collection = chroma_client().get_collection("xml_documents")
    got = collection.get(include=["documents", "metadatas"])
    approvals, last_session = set(), 0
    for document, m in zip(got["documents"], got["metadatas"]):
        session = int(m.get("number") or 0)
        last_session = max(last_session, session)
        text = re.sub(r"\s+", " ", document.split("\n\n", 1)[-1])
        match = _APPROVAL.search(text)
        if not match:
            continue
        part_of = m.get("part_of") or ""
        decision = _DECISION.search(part_of)
        where = (f"decision {decision.group(0)}" if decision
                 else part_of.split(" › ")[-1] if part_of else "report")
        if m.get("paragraph"):
            where += f", para. {m['paragraph'].rstrip('.')}"
        wording = match.group(0).rstrip(";").strip()
        approvals.add((session, (m.get("date") or "")[:4], where,
                       wording[0].upper() + wording[1:]))
    return sorted(approvals), last_session


@lru_cache(maxsize=1)
def catalogue_block() -> str:
    """The reference list as a prompt section ("" if it cannot be built)."""
    try:
        assessments = _assessments()
        approvals, last_session = _approvals()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Assessment reference list unavailable: {exc}")
        return ""

    lines = [
        "Reference list: IPBES assessments",
        "",
        "These assessments are in the assessment collection; all of them are "
        "approved IPBES products. For each assessment the Plenary approves "
        "the summary for policymakers (SPM) and accepts the chapters; "
        "earlier, it approves the assessment's scoping report.",
    ]
    lines += [f"- {code}: {title} ({year})" for code, title, year in assessments]
    if approvals:
        lines += [
            "",
            f"SPM approvals recorded in the Plenary reports available here "
            f"(sessions 1 to {last_session}); assessments approved at later "
            f"sessions are not covered by these reports:",
        ]
        for session in sorted({a[0] for a in approvals}):
            entries = [a for a in approvals if a[0] == session]
            lines.append(f"IPBES-{session} ({entries[0][1]}):")
            lines += [f"- {where}: \"{wording}\"" for _, _, where, wording in entries]
    return "\n".join(lines)
