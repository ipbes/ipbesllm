import asyncio
import os
import re
from collections import defaultdict

from chromadb.errors import NotFoundError
from loguru import logger

from assessment_catalogue import catalogue_block, uploaded_session_numbers
from cache import CACHE_VERSION, cache, make_key
from rag_utils import (
    EMBED_MODEL, LLM_MODEL, MAX_CONTEXT_CHARS, NUM_CTX, USE_AZURE, chat,
    chroma_client, embed_query,
)
from settings import CHROMA_DIR




COLLECTION_NAME = "xml_documents"

# Part of the answer cache key: bump when the prompt changes.
PROMPT_VERSION = "4"
ANSWER_TTL = 60 * 60 * 6      # 6 hours

# Each matching paragraph also brings the NEIGHBOURS paragraphs before and
# after it in the same section: a paragraph that is mostly a list (e.g. the
# officers elected) matches questions poorly, but the one next to it often
# matches well.
NEIGHBOURS = 1

# The neighbours make the prompt 2-3 times longer. The Ollama limit
# (MAX_CONTEXT_CHARS, sized for an 8 GB machine) would cut lower-ranked
# matches; an Azure model takes far more.
XML_MAX_CONTEXT_CHARS = int(os.getenv(
    "XML_MAX_CONTEXT_CHARS", 60_000 if USE_AZURE else MAX_CONTEXT_CHARS
))

# Plenary sessions, however a question names them: "IPBES-1", "IPBES 1",
# "IPBES1", "#IPBES1", "first session", "1st session", "session 1". Each
# report's metadata has its session in `number`, so a named session also
# limits the search to its report.
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
            "thirteenth", "fourteenth", "fifteenth", "sixteenth",
            "seventeenth", "eighteenth", "nineteenth", "twentieth"]
_IPBES_SESSION = re.compile(r"#?\bIPBES[\s-]?(\d{1,2})\b", re.IGNORECASE)
_SESSION_PHRASES = [
    re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+(?:plenary\s+)?session\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s+(?:plenary\s+)?session\b", re.IGNORECASE),
    re.compile(r"\bsession\s+(\d{1,2})\b", re.IGNORECASE),
]


def session_numbers(question: str) -> set[str]:
    """The Plenary sessions a question names, as strings ("1", "2" ...)."""
    found = {int(n) for n in _IPBES_SESSION.findall(question)}
    for pattern in _SESSION_PHRASES:
        for value in pattern.findall(question):
            found.add(ORDINALS.index(value.lower()) + 1 if value.isalpha()
                      else int(value))
    return {str(n) for n in found if n >= 1}


def _spell_out_sessions(question: str) -> str:
    """ "IPBES-1" -> "the first session of the Plenary", as the reports say."""
    def spell(match):
        n = int(match.group(1))
        if 1 <= n <= len(ORDINALS):
            return f"the {ORDINALS[n - 1]} session of the Plenary"
        return match.group(0)
    return _IPBES_SESSION.sub(spell, question)


# "The next session", "the last session": no number, so they are taken to
# mean the latest session with a report. What the next session will be was
# decided there, in its decision (and report section) on future sessions.
_SESSION_WORDS = r"\s+(?:(?:IPBES|plenary)\s+)?(?:sessions?|meetings?|plenary)\b"
_NEXT_SESSION = re.compile(r"\b(?:next|upcoming|forthcoming|future)" + _SESSION_WORDS,
                           re.IGNORECASE)
_LATEST_SESSION = re.compile(r"\b(?:last|latest|most\s+recent|previous)" + _SESSION_WORDS,
                             re.IGNORECASE)
# The decision on future sessions ("Organization of the Plenary and dates
# and venues of future sessions", from IPBES-9) and the report section on
# them ("Provisional agenda, date and venue of future sessions" ...).
_FUTURE_SESSIONS = re.compile(r"future sessions|date and venue|dates and venues",
                              re.IGNORECASE)
_DECISION_SYMBOL = re.compile(r"^Decision (IPBES-\d+/\d+)")


def resolve_relative_sessions(question: str) -> str:
    """The question with the session that "the next session" or "the last
    session" means added, e.g. "... (The latest session with a report is
    IPBES-12; what it decided about future sessions.)". Unchanged when it
    names a session, or names none."""
    if session_numbers(question):
        return question
    asks_next = _NEXT_SESSION.search(question)
    if not (asks_next or _LATEST_SESSION.search(question)):
        return question
    sessions = uploaded_session_numbers()
    if not sessions:
        return question
    note = f"The latest session with a report is IPBES-{max(sessions)}"
    if asks_next:
        note += "; what it decided about future sessions"
    return f"{question} ({note}.)"


_PARAGRAPH_XPATH = re.compile(r"^(.*/paragraph)\[(\d+)\]$")


def retrieve(
    question: str,
    k: int = 10,
):

    client = chroma_client()

    # No embedding function: the question is embedded by embed_query
    # (rag_utils: OLLAMA_URL, OLLAMA_EMBED_MODEL, timeout and retries).
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
        )
    except (NotFoundError, ValueError) as exc:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found in '{CHROMA_DIR}'. "
            f"Run xml_index.py first. (Original error: {exc})"
        )

    sessions = session_numbers(question)
    results = collection.query(
        query_embeddings=[list(embed_query(_spell_out_sessions(question),
                                           EMBED_MODEL))],
        n_results=k,
        where={"number": {"$in": sorted(sessions)}} if sessions else None,
        include=["documents", "metadatas", "distances"],
    )
    results = _with_neighbours(collection, results)
    if sessions and (_NEXT_SESSION.search(question)
                     or _FUTURE_SESSIONS.search(question)):
        results = _with_future_sessions(collection, results, sessions)
    return results


def _with_future_sessions(collection, results, sessions: set[str]):
    """Put first, whole and in document order, what the sessions decided
    about future sessions: the decision (with its annexes, such as the
    next session's provisional agenda) and the report's section on them.
    Searched by similarity, these lose to agenda headings worded the same
    in every report, and the operative paragraphs ("Decides that the
    thirteenth session ...") are left out."""
    got = collection.get(where={"number": {"$in": sorted(sessions)}},
                         include=["documents", "metadatas"])
    decisions = {match.group(1)
                 for m in got["metadatas"]
                 if _FUTURE_SESSIONS.search(m.get("part_of") or "")
                 and (match := _DECISION_SYMBOL.match(m["part_of"]))}

    def wanted(m: dict) -> bool:
        part_of = m.get("part_of") or ""
        if part_of:
            return any(part_of.startswith(f"Decision {symbol}:")
                       or part_of.endswith(f"to decision {symbol}")
                       for symbol in decisions)
        return bool(_FUTURE_SESSIONS.search(m.get("division") or ""))

    rows = sorted(
        (row for row in zip(got["ids"], got["documents"], got["metadatas"])
         if wanted(row[2])),
        # The decision, then its annexes, then the report's section.
        key=lambda row: (row[2].get("source_file", ""),
                         _part_rank(row[2].get("part_of") or ""),
                         _xpath_key(row[2].get("xpath", ""))),
    )
    if not rows:
        return results
    first = {id_ for id_, _, _ in rows}
    out = {"ids": [], "documents": [], "metadatas": [], "distances": []}
    for id_, document, metadata in rows:
        out["ids"].append(id_)
        out["documents"].append(document)
        out["metadatas"].append(metadata)
        out["distances"].append(0.0)
    for i, id_ in enumerate(results["ids"][0]):
        if id_ not in first:
            for key in out:
                out[key].append(results[key][0][i])
    return {key: [value] for key, value in out.items()}


def _part_rank(part_of: str) -> tuple:
    """Decision first, then "Annex I to decision ...", "Annex II ...", then
    the main report (no part_of)."""
    if part_of.startswith("Decision "):
        return (0, "")
    return (1, part_of) if part_of else (2, "")


def _xpath_key(xpath: str) -> list:
    """Sort key putting paragraph[2] before paragraph[10]."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", xpath)]


def _with_neighbours(collection, results):
    """Add the paragraphs around each matching paragraph, in document order
    next to it. A neighbour gets its match's distance and a `neighbour_of`
    metadata key; one that is itself a match keeps its first position."""
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # (offset, (source_file, xpath)) of the neighbours wanted, by match.
    wanted: dict[int, list[tuple[int, tuple[str, str]]]] = defaultdict(list)
    for i, metadata in enumerate(metadatas):
        match = _PARAGRAPH_XPATH.match(metadata.get("xpath") or "")
        if not match:
            continue
        base, number = match.group(1), int(match.group(2))
        for offset in range(-NEIGHBOURS, NEIGHBOURS + 1):
            if offset and number + offset >= 1:
                wanted[i].append((offset, (metadata["source_file"],
                                           f"{base}[{number + offset}]")))

    xpaths_by_file: dict[str, set[str]] = defaultdict(set)
    for neighbours in wanted.values():
        for _, (source_file, xpath) in neighbours:
            xpaths_by_file[source_file].add(xpath)
    found = {}
    for source_file, xpaths in xpaths_by_file.items():
        got = collection.get(
            where={"$and": [{"source_file": source_file},
                            {"xpath": {"$in": sorted(xpaths)}}]},
            include=["documents", "metadatas"],
        )
        for id_, document, metadata in zip(got["ids"], got["documents"],
                                           got["metadatas"]):
            found[(source_file, metadata["xpath"])] = (id_, document, metadata)

    out = {"ids": [], "documents": [], "metadatas": [], "distances": []}
    seen: set[str] = set()

    def add(id_, document, metadata, distance):
        if id_ not in seen:
            seen.add(id_)
            out["ids"].append(id_)
            out["documents"].append(document)
            out["metadatas"].append(metadata)
            out["distances"].append(distance)

    for i, id_ in enumerate(ids):
        # Offsets run -NEIGHBOURS..NEIGHBOURS, so this is document order.
        for offset, key in wanted.get(i, []) or [(0, None)]:
            if offset > 0 and id_ not in seen:
                add(id_, documents[i], metadatas[i], distances[i])
            if key in found:
                n_id, n_doc, n_meta = found[key]
                add(n_id, n_doc,
                    {**n_meta, "neighbour_of": metadatas[i].get("eId")},
                    distances[i])
        add(id_, documents[i], metadatas[i], distances[i])

    return {key: [value] for key, value in out.items()}


def _build_context(results, max_chars: int = XML_MAX_CONTEXT_CHARS) -> tuple[str, int]:
    """Return (context text, number of chunks included).

    Chunks are added in order until `max_chars` would be exceeded; the first
    chunk is always included. The caller reports any truncation.
    """
    parts: list[str] = []
    used = 0

    for i, document in enumerate(results["documents"][0]):
        part = f"\nSOURCE {i + 1}\n\n" + format_chunk(
            document, results["metadatas"][0][i] or {}
        )
        if parts and used + len(part) + 1 > max_chars:
            break
        parts.append(part)
        used += len(part) + 1

    return "\n".join(parts), len(parts)


def format_chunk(document: str, m: dict) -> str:
    """One chunk as the prompt shows it: its location, then its text."""
    return f"""File: {m.get("source_file", "")}
Document type: {m.get("document_type", "")}
Title: {m.get("title", "")}
Date: {m.get("date", "")}

Section: {m.get("division", "")}
Subsection: {m.get("subdivision", "")}
Paragraph: {m.get("paragraph", "")}

Akoma Ntoso eId: {m.get("eId", "")}
Location: {m.get("xpath", "")}

Content:
{document}
"""


def generate_answer(
    question: str,
    results,
    answer_stream=None,
):

    context, used = _build_context(results)
    total = len(results["documents"][0])
    truncation_note = ""
    if used < total:
        logger.warning(f"Context capped at {used} of {total} retrieved chunks "
                       f"(XML_MAX_CONTEXT_CHARS={XML_MAX_CONTEXT_CHARS}).")
        truncation_note = (
            f"\nNOTE: the context limit was reached, so only the first {used} "
            f"of {total} retrieved chunks are shown above.\n"
        )

    prompt = f"""
You are answering questions about
Akoma Ntoso documents.

Use ONLY the supplied context.

The XML contains structural information such as:
- document
- the decision or annex a passage is part of ("Part of")
- section
- subsection
- paragraph
- table row
- Akoma Ntoso eId
- XML location

Use that information when identifying sources.

If the answer cannot be established from
the supplied context, say:

"I cannot determine that from the supplied documents."

Do not invent:
- facts
- dates
- names
- numbers
- decisions
- policies
- procedural rules

Question:
{question}

Context:
{context}
{truncation_note}
{catalogue_block()}

For which assessments exist or were approved, use the reference list above,
cited as [Reference list]. Present what it says about IPBES procedure as
IPBES procedure, not as a rule or instruction you were given.

Answer the question clearly.

At the end provide:

Sources:
- document
- decision or annex ("Part of"), if any
- section/subsection
- paragraph or table row
- Akoma Ntoso location
"""

    return chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied "
                    "Akoma Ntoso document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        answer_stream=answer_stream,
    )


async def generate_answer_cached(question: str, results, answer_stream=None) -> dict:
    """Cache the answer, keyed on the question and the retrieved context."""
    context, used = _build_context(results)
    total = len(results["documents"][0])
    key = make_key(
        "xml:answer",
        CACHE_VERSION,
        PROMPT_VERSION,
        LLM_MODEL,
        NUM_CTX,
        question.strip(),
        make_key("xml:ctx", context, catalogue_block()),
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"xml answer HIT  q={question[:50]!r}")
        return {**hit, "_cached": True}

    answer = await asyncio.to_thread(generate_answer, question, results, answer_stream)
    payload = {
        "answer": answer,
        "context_chunks": used,
        "context_truncated": used < total,
        "_cached": False,
    }
    await cache.set(key, payload, ttl=ANSWER_TTL)
    logger.debug(f"xml answer MISS q={question[:50]!r}")
    return payload


def main():

    print(
        f"LLM: {LLM_MODEL}"
    )

    print(
        f"Embedding: {EMBED_MODEL}"
    )

    print()

    question = input(
        "Question: "
    ).strip()

    if not question:
        return

    results = retrieve(
        question,
        k=5,
    )

    answer = generate_answer(
        question,
        results,
    )

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED XML SOURCES")
    print("=" * 80)

    for i, metadata in enumerate(
        results["metadatas"][0]
    ):

        distance = results["distances"][0][i]

        print(
            f"{i + 1}. "
            f"{metadata.get('source_file', '')} | "
            f"{metadata.get('division', '')} | "
            f"{metadata.get('subdivision', '')} | "
            f"paragraph {metadata.get('paragraph', '')} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()