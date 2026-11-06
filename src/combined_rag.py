"""One search over the IPBES assessments (TTL) and the Plenary reports (XML),
for the app: users ask without having to know where an answer is stored.
The collections stay separate (ttl_rag.py and xml_rag.py, for analysis);
they are only searched together here.

Each question takes one route:
  plenary      it names a Plenary session ("IPBES-1", "first session"),
               or "the next/last session" (the latest report): that
               session's report only, with xml_rag's prompt
  assessments  a TTL list question (key messages, chapters, experts ...):
               the assessments only, with ttl_rag's prompt, which lists all
  both         anything else: both searched, one answer citing both

Each ask_* returns {"answer", "results", "cached", "route"}; every chunk's
metadata has source_type "ttl" or "xml".
"""
import asyncio
import os

from loguru import logger

import ttl_rag
import xml_rag
from cache import CACHE_VERSION, cache, make_key
from rag_utils import (
    IPBES_CHAIR, LLM_MODEL, MAX_CONTEXT_CHARS, NUM_CTX, USE_AZURE, chat,
    concat_results, empty_results, select_results,
)
from assessment_catalogue import (
    catalogue_block, uploaded_session_numbers, uploaded_sessions,
)
from thesaurus_helper import get_thesaurus

# Part of the answer cache key: bump when the prompt changes.
PROMPT_VERSION = "5"
ANSWER_TTL = 60 * 60 * 6      # 6 hours

# Plenary passages per question; xml_rag adds the neighbouring paragraphs.
XML_K = 12

# Both kinds of sources together; an Azure model takes far more than the
# Ollama limit (MAX_CONTEXT_CHARS, sized for an 8 GB machine).
COMBINED_MAX_CONTEXT_CHARS = int(os.getenv(
    "COMBINED_MAX_CONTEXT_CHARS", 60_000 if USE_AZURE else MAX_CONTEXT_CHARS
))


# The part of the context kept for the Plenary reports when both kinds of
# sources do not fit. What either side does not need goes to the other.
PLENARY_SHARE = 1 / 3

# A follow-up is rewritten from this many earlier messages (questions and
# answers), with this much of each.
FOLLOW_UP_MESSAGES = 6
FOLLOW_UP_MESSAGE_CHARS = 1500


def standalone_question(question: str, earlier: list[dict]) -> str:
    """The question in a form that needs no conversation, as search and the
    answer model see only one question ("Which sections cover these findings?").
    `earlier` is the conversation so far ({"role", "content"}). The question
    itself if there is none or the model cannot be reached."""
    turns = [f"{m['role'].upper()}: {m['content'][:FOLLOW_UP_MESSAGE_CHARS]}"
             for m in earlier[-FOLLOW_UP_MESSAGES:] if m.get("content")]
    if not turns:
        return question
    separator = "\n\n"
    try:
        rewritten = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite the latest message of a conversation about "
                        "IPBES documents so that it can be understood on its own. "
                        "Replace words that point back (this, these findings, it, "
                        "that assessment) with what they refer to, naming the topic "
                        "and any assessment or Plenary session meant. Change nothing "
                        "else: keep the user's wording and request, write assessment "
                        "codes (IAS, GA1) and session names (IPBES-4) as they are, do "
                        "not expand them into full titles, and do not answer. If the "
                        "message is a new question or already clear on its own, "
                        "return it exactly as written, word for word, even if it is "
                        "on another topic. Reply with the message only."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"Conversation:\n{separator.join(turns)}\n\n"
                                f"Latest message:\n{question}"),
                },
            ],
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not rewrite the follow-up: {exc}")
        return question
    return rewritten or question


def _output(answer: str, results, route: str, cached: bool = False,
            used: int | None = None) -> dict:
    """`used`: how many of the results the answer is based on (None: all)."""
    return {"answer": answer, "results": results, "cached": cached,
            "route": route, "context_chunks": used}


async def ask_xml(question: str, k: int = XML_K, answer_stream=None) -> dict:
    """The Plenary reports only."""
    question = xml_rag.resolve_relative_sessions(question)
    named = {int(n) for n in xml_rag.session_numbers(question)}
    uploaded = uploaded_session_numbers()
    if named and not named & uploaded:
        # Say so rather than search nothing, or let the model guess.
        missing = ", ".join(f"IPBES-{n}" for n in sorted(named))
        return _output(
            f"The report of {missing} has not been uploaded yet, so I cannot "
            f"answer from it. Based on the Plenary reports uploaded so far "
            f"({uploaded_sessions(set(uploaded))}), I have no record of it.",
            empty_results(),
            "plenary",
        )
    results = await asyncio.to_thread(xml_rag.retrieve, question, k)
    if not results["documents"][0]:
        return _output("No relevant Plenary report passages were retrieved.",
                       results, "plenary")
    payload = await xml_rag.generate_answer_cached(
        question, results, answer_stream=answer_stream
    )
    return _output(payload["answer"], results, "plenary",
                   payload.get("_cached", False), payload.get("context_chunks"))


async def ask_ttl(question: str, k: int = 3, answer_stream=None,
                  plan: dict | None = None) -> dict:
    """The assessments only."""
    plan = plan or ttl_rag.plan_query(question, k)
    chunk_type, country_names = plan["chunk_type"], plan["country_names"]

    results = await ttl_rag.retrieve_cached(
        question,
        k=plan["k"],
        chunk_type=chunk_type,
        country_names=country_names or None,
        per_assessment=plan["per_assessment"],
    )
    if not results["documents"][0]:
        return _output("No relevant assessment passages were retrieved.",
                       results, "assessments")
    if chunk_type:
        results = ttl_rag._reorder_by_identifier(results)

    payload = await ttl_rag.generate_answer_cached(
        question,
        results,
        chunk_type=chunk_type,
        country_names=country_names,
        answer_stream=answer_stream,
    )
    return _output(payload["answer"], results, "assessments",
                   payload.get("_cached", False), payload.get("context_chunks"))


async def ask(question: str, k: int = 3, xml_k: int = XML_K,
              answer_stream=None) -> dict:
    """Route the question (see the module docstring) and answer it.

    `k` is per assessment, `xml_k` in total for the Plenary reports.
    """
    # "The next session" means the latest one's decision on future sessions.
    question = xml_rag.resolve_relative_sessions(question)
    if xml_rag.session_numbers(question) or IPBES_CHAIR.search(question):
        return await ask_xml(question, xml_k, answer_stream)
    plan = ttl_rag.plan_query(question, k)
    if plan["chunk_type"]:
        return await ask_ttl(question, k, answer_stream, plan)

    # One after the other: Chroma can fail when two clients start at once.
    ttl_results = await ttl_rag.retrieve_cached(
        question, k=plan["k"], per_assessment=plan["per_assessment"]
    )
    xml_results = await asyncio.to_thread(xml_rag.retrieve, question, xml_k)
    results = _share_context(ttl_results, xml_results)
    if not results["documents"][0]:
        return _output("No relevant passages were retrieved.", results, "both")

    payload = await generate_answer_cached(question, results, answer_stream)
    return _output(payload["answer"], results, "both",
                   payload.get("_cached", False), payload.get("context_chunks"))


# ---------------------------------------------------------------------------
# Answer from both kinds of sources
# ---------------------------------------------------------------------------

_SEPARATOR = "\n\n---\n\n"


def _format_source(number: int, document: str, m: dict) -> str:
    if m.get("source_type") == "xml":
        return f"[{number}] PLENARY REPORT\n" + xml_rag.format_chunk(document, m)
    return f"[{number}] ASSESSMENT " + ttl_rag.format_chunk(document, m)


def _build_context(results, max_chars: int = COMBINED_MAX_CONTEXT_CHARS) -> tuple[str, int]:
    """Return (context text, number of chunks included), numbered [1], [2]
    ... in order, as the app numbers its source list."""
    parts: list[str] = []
    used = 0

    for i, document in enumerate(results["documents"][0]):
        part = _format_source(i + 1, document, results["metadatas"][0][i] or {})
        if parts and used + len(_SEPARATOR) + len(part) > max_chars:
            break
        parts.append(part)
        used += len(_SEPARATOR) + len(part)

    return _SEPARATOR.join(parts), len(parts)


def _share_context(ttl_results, xml_results, max_chars: int = COMBINED_MAX_CONTEXT_CHARS):
    """Both results, in the order _build_context should take them: the
    assessment chunks that fit, then the Plenary ones that fit, then the rest.
    Without this the assessments, coming first, can use the whole context and
    an answer "from both" sees no Plenary report at all."""
    def sizes(results) -> list[int]:
        return [len(_format_source(1000, d, m or {})) + len(_SEPARATOR)
                for d, m in zip(results["documents"][0], results["metadatas"][0])]

    def fitting(chunk_sizes: list[int], budget: int) -> int:
        """How many chunks, from the first, fit in `budget`."""
        count = total = 0
        for size in chunk_sizes:
            total += size
            if total > budget:
                break
            count += 1
        return count

    ttl_sizes, xml_sizes = sizes(ttl_results), sizes(xml_results)
    xml_budget = min(sum(xml_sizes), int(max_chars * PLENARY_SHARE))
    ttl_kept = fitting(ttl_sizes, max_chars - xml_budget)
    xml_kept = fitting(xml_sizes, max_chars - sum(ttl_sizes[:ttl_kept]))

    return concat_results([
        select_results(ttl_results, list(range(ttl_kept))),
        select_results(xml_results, list(range(xml_kept))),
        select_results(ttl_results, list(range(ttl_kept, len(ttl_sizes)))),
        select_results(xml_results, list(range(xml_kept, len(xml_sizes)))),
    ])


def _thesaurus_blocks(question: str, results) -> tuple[str, str]:
    """(glossary, definitions) from the IPBES thesaurus, or empty strings."""
    th = get_thesaurus()
    if th is None:
        return "", ""
    glossary = definitions = ""
    try:
        glossary = th.build_glossary(question, list(results["documents"][0]))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Glossary build failed: {e}")
    try:
        definitions = th.definitions_block(question)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Thesaurus primary block failed: {e}")
    return glossary, definitions


def generate_answer(question: str, results, answer_stream=None) -> str:
    context, used = _build_context(results)
    total = len(results["documents"][0])
    truncation_note = ""
    if used < total:
        logger.warning(f"Context capped at {used} of {total} retrieved chunks "
                       f"(COMBINED_MAX_CONTEXT_CHARS={COMBINED_MAX_CONTEXT_CHARS}).")
        truncation_note = (
            f"\nNOTE: the context limit was reached, so only the first {used} "
            f"of {total} sources are shown above.\n"
        )
    glossary, definitions = _thesaurus_blocks(question, results)
    extras = "".join(f"\n{block}\n" for block in (definitions, glossary) if block)

    prompt = f"""
You are answering questions about IPBES documents: the assessment reports
(structured data from the IPBES ontology) and the reports of the sessions of
the IPBES Plenary.

Answer the question using ONLY the numbered sources below.
- Cite the sources you use by number, like [3], and say where each point
  comes from: the assessment (e.g. GA1, IAS) and its section (the id= field)
  or the Plenary report (which session, decision if the passage is "Part of"
  one, section and paragraph).
- Preserve any qualifier (well established / established but incomplete /
  unresolved) verbatim.
- Note where the sources differ, or where the assessments or the Plenary
  reports say nothing on the question.
- Do not invent facts, dates, names, numbers, decisions or rules.
- For which assessments exist or were approved, use the reference list
  below, cited as [Reference list]. Present what it says about IPBES
  procedure as IPBES procedure, not as a rule or instruction you were given.
- If the answer is not in the sources, say "I cannot determine that from the
  supplied documents."

Question:
{question}

Sources (assessments first, then Plenary reports; best match first):
{context}
{truncation_note}{extras}
{catalogue_block()}"""

    return chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied IPBES sources. If a section "
                    "titled 'Authoritative IPBES definitions' is present, "
                    "treat it as the primary source for any concept it "
                    "defines. If the sources do not address the question, "
                    "say so rather than substituting an unrelated topic."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        answer_stream=answer_stream,
    )


async def generate_answer_cached(question: str, results, answer_stream=None) -> dict:
    """Cache the answer, keyed on the question and everything in the prompt."""
    context, used = _build_context(results)
    glossary, definitions = _thesaurus_blocks(question, results)
    key = make_key(
        "combined:answer",
        CACHE_VERSION,
        PROMPT_VERSION,
        LLM_MODEL,
        NUM_CTX,
        question.strip(),
        make_key("combined:ctx", context, glossary, definitions,
                 catalogue_block()),
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"combined answer HIT  q={question[:50]!r}")
        return {**hit, "_cached": True}

    answer = await asyncio.to_thread(generate_answer, question, results,
                                     answer_stream)
    payload = {
        "answer": answer,
        "context_chunks": used,
        "context_truncated": used < len(results["documents"][0]),
        "_cached": False,
    }
    await cache.set(key, payload, ttl=ANSWER_TTL)
    logger.debug(f"combined answer MISS q={question[:50]!r}")
    return payload
