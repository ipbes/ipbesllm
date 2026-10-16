"""Compare PDF answers with the structured sources, like for like: one row
per question and scope (an assessment or a Plenary meeting).

    PYTHONPATH=src python src/pdf_ttl_compare.py                      # everything
    PYTHONPATH=src python src/pdf_ttl_compare.py --questions 1-5      # a part
    PYTHONPATH=src python src/pdf_ttl_compare.py --scopes GA1,IAS,IPBES-4

Scopes per question:
  * a question about the Plenary or a session (IPBES-4, "first session")
    gets a row per meeting (the sessions it names, else every uploaded one),
    comparing the PDF and XML versions of that meeting's report;
  * any other question gets a row per assessment (those it names, else all),
    comparing the assessment's PDFs with its TTL data.
Each scope is asked as the question plus "(Answer for the GA1 assessment
only.)" or "(Answer for IPBES-4 only.)": both pipelines limit their search to
what a question names, so each answer covers that scope alone. The `asked`
column records the question as sent.

Rows are written as each finishes, to tests/comparison_<run>[_<part>].csv;
answers are cached, so a rerun only spends time on what was not done.
"""
import argparse
import asyncio
import csv
import datetime
import re
from pathlib import Path

import combined_rag
import ttl_rag
import xml_rag
from assessment_catalogue import uploaded_session_numbers
from rag_utils import IPBES_CHAIR
from pdf_rag import (
    retrieve_cached as pdf_retrieve,
    generate_answer_cached as pdf_answer,
)


QUESTIONS = [
    "experts from China",
    "experts from Kenya",
    "experts from East Africa",
    "experts from Eastern Europe",
    "What are the key messages?",
    "What are the knowledge gaps?",
    "What are the key messages identified in the IPBES LDR assessment?",
    "What are the background messages identified in the IPBES LDR assessment?",
    "What are the chapters of the IPBES LDR assessment?",
    "Chapters of the assessments",
    "What findings across the different IPBES assessments refer to cities?",
    "What is NCP and how does it relate to nature?",
    "Provide a summary of the topics covered across these different assessments?",
    "Provide a summary of the topics covered across these different assessments and add the sections in the assessments where these topics are covered",
    "Who were the Review Editors of Ch. 4 of the values assessment",
    "How have previous IPBES assessments approached the IPBES Conceptual Framework? Do you have figures referring to it?",
    "Provide a summary of the key findings in IPBES assessments that directly refer to the KMGBF",
    # Priority topics: how each assessment responds. Each assessment is
    # searched separately, so answers should cover GA1, IAS and LDR in turn.
    # Pollution
    "What do IPBES assessments say about pollution?",
    "How do the assessments describe chemical pollution, plastics and nutrient runoff as drivers of nature loss?",
    # Cities
    "What do IPBES assessments say about cities and urbanization?",
    "What role do nature-based solutions and ecological connectivity play in cities according to the assessments?",
    # Climate change
    "What do IPBES assessments say about climate change?",
    "How do the assessments link climate change adaptation, mitigation and extreme events to biodiversity?",
    # Poverty
    "What do IPBES assessments say about poverty?",
    "How do the assessments link nature to poverty, inequality, vulnerability and food security?",
    # One assessment only (named in the question)
    "What does the IAS assessment say about poverty?",
    # Plenary sessions: compared PDF against XML, one row per meeting
    "What did the Plenary decide about the work programme?",
    "Who was elected as Chair of IPBES?",
    "What did the Plenary decide about the budget?",
    # add more here
]

FIELDNAMES = [
    "timestamp", "run_id", "question_no", "question", "scope_type", "scope",
    "asked", "pdf_response", "pdf_sources", "other_source", "other_response",
    "other_sources",
]
CONCURRENT_SCOPES = 4
# A question about the Plenary's meetings rather than the assessments.
MEETING_QUESTION = re.compile(r"\bplenary\b|\bmeetings?\b", re.IGNORECASE)


def scopes_for(question: str, assessments: list[str], sessions: list[str]) -> list[tuple[str, str]]:
    """[(scope_type, scope)] the question is compared on (module docstring)."""
    named_sessions = sorted(xml_rag.session_numbers(question), key=int)
    if named_sessions or MEETING_QUESTION.search(question) or IPBES_CHAIR.search(question):
        return [("meeting", f"IPBES-{n}") for n in named_sessions or sessions]
    named = ttl_rag.named_assessments(question, assessments)
    return [("assessment", a) for a in named or assessments]


def scoped(question: str, scope_type: str, scope: str) -> str:
    if scope_type == "meeting":
        return f"{question} (Answer for {scope} only.)"
    return f"{question} (Answer for the {scope} assessment only.)"


def _sources(results, describe) -> str:
    return "\n".join(
        f"{i}. {describe(m)} (d={d:.3f})"
        for i, (m, d) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1)
    )


# ---------------------------------------------------------------------------
# The pipelines
# ---------------------------------------------------------------------------

async def run_pdf(asked: str) -> tuple[str, str]:
    """(answer, sources) from the PDFs."""
    results = await pdf_retrieve(asked, k=5, country_names=None)
    if not results["documents"][0]:
        return "(no results)", ""
    payload = await pdf_answer(asked, results, country_names=None)

    def describe(m):
        printed = f" (printed {m['printed_page']})" if m.get("printed_page") else ""
        return f"{m.get('source_file', '?')} p{m.get('page', '?')}{printed}"
    return payload["answer"], _sources(results, describe)


async def run_ttl(asked: str) -> tuple[str, str]:
    """(answer, sources) from the TTL assessments."""
    out = await combined_rag.ask_ttl(asked)
    return out["answer"], _sources(out["results"], lambda m: (
        f"{m.get('assessment', '')} {m.get('chunk_type', '?')} "
        f"{m.get('identifier') or m.get('eId', '?')}"))


async def run_xml(asked: str) -> tuple[str, str]:
    """(answer, sources) from the XML Plenary reports."""
    out = await combined_rag.ask_xml(asked)
    return out["answer"], _sources(out["results"], lambda m: (
        f"{m.get('source_file', '?')} {m.get('part_of') or ''} "
        f"{m.get('division') or ''} para {m.get('paragraph') or '-'}".strip()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_numbers(spec: str | None, count: int) -> list[int]:
    """ "1-5,8" -> [1, 2, 3, 4, 5, 8] (1-based); None -> all."""
    if not spec:
        return list(range(1, count + 1))
    numbers = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        numbers += range(int(a), int(b or a) + 1)
    return [n for n in dict.fromkeys(numbers) if 1 <= n <= count]


async def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", help='question numbers, e.g. "1-5,8"')
    ap.add_argument("--scopes", help='e.g. "GA1,IAS,IPBES-4" (default: all)')
    args = ap.parse_args(argv)

    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d-%H%M%S")
    part = "_".join(filter(None, [
        f"q{args.questions}" if args.questions else "",
        args.scopes.replace(",", "-") if args.scopes else "",
    ]))
    out_path = (Path(__file__).resolve().parent.parent / "tests"
                / f"comparison_{run_id}{'_' + part if part else ''}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assessments = ttl_rag._get_assessments()
    sessions = [str(n) for n in sorted(uploaded_session_numbers())]
    wanted = {s.strip().upper() for s in args.scopes.split(",")} if args.scopes else None
    jobs = [
        (n, QUESTIONS[n - 1], scope_type, scope)
        for n in parse_numbers(args.questions, len(QUESTIONS))
        for scope_type, scope in scopes_for(QUESTIONS[n - 1], assessments, sessions)
        if not wanted or scope.upper() in wanted
    ]
    print(f"Run id: {run_id}")
    print(f"Output: {out_path}")
    print(f"Rows to produce: {len(jobs)} ({len({j[0] for j in jobs})} questions)\n")

    out_file = out_path.open("w", newline="", encoding="utf-8-sig")  # BOM: Excel
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    out_file.flush()
    limit = asyncio.Semaphore(CONCURRENT_SCOPES)
    done = 0

    async def run(n, question, scope_type, scope):
        nonlocal done
        asked = scoped(question, scope_type, scope)
        other = "XML" if scope_type == "meeting" else "TTL"
        async with limit:
            row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                   "run_id": run_id, "question_no": n, "question": question,
                   "scope_type": scope_type, "scope": scope, "asked": asked,
                   "other_source": other}
            for key, pipeline in (("pdf", run_pdf),
                                  ("other", run_xml if other == "XML" else run_ttl)):
                try:
                    row[f"{key}_response"], row[f"{key}_sources"] = await pipeline(asked)
                except Exception as e:  # noqa: BLE001
                    row[f"{key}_response"], row[f"{key}_sources"] = f"(error: {e})", ""
        writer.writerow(row)
        out_file.flush()
        done += 1
        print(f"[{done}/{len(jobs)}] Q{n} {scope}: {question[:60]}")

    await asyncio.gather(*(run(*job) for job in jobs))
    out_file.close()
    duration = (datetime.datetime.now() - started).total_seconds()
    print(f"\nWrote {done} rows to {out_path} ({duration:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
