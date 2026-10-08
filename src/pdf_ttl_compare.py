# src/pdf_ttl_compare.py
import asyncio
import csv
import datetime
from pathlib import Path

from pdf_rag import (
    retrieve_cached as pdf_retrieve,
    generate_answer_cached as pdf_answer,
)
from ttl_rag import (
    retrieve_cached as ttl_retrieve,
    generate_answer_cached as ttl_answer,
    _infer_chunk_type as ttl_infer_chunk_type,
)
from geo import infer_country_names


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
    "Provide a summary of the topics covered across these different assessments?",
    "Provide a summary of the topics covered across these different assessments and add the sections in the assessments where these topics are covered",
    "Who were the Review Editors of Ch. 4 of the values assessment",
    "How have previous IPBES assessments approached the IPBES Conceptual Framework? Do you have figures referring to it?",
    "Provide a summary of the key findings in IPBES assessments that directly refer to the KMGBF",
    # add more here
]

FIELDNAMES = [
    "timestamp",
    "run_id",
    "question",
    "pdf_response",
    "pdf_source",
    "ttl_response",
    "ttl_source",
]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

async def run_pdf(question: str) -> tuple[str, str]:
    """Return (answer, sources_as_string)."""
    results = await pdf_retrieve(question, k=5, country_names=None)
    if not results["documents"][0]:
        return "(no results)", ""

    payload = await pdf_answer(question, results, country_names=None)

    sources = []
    for i, m in enumerate(results["metadatas"][0], start=1):
        d = results["distances"][0][i - 1]
        sources.append(
            f"{i}. {m.get('source_file', '?')} "
            f"p{m.get('page', '?')} "
            f"(d={d:.3f})"
        )

    return payload["answer"], "\n".join(sources)


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

async def run_ttl(question: str) -> tuple[str, str]:
    """Return (answer, sources_as_string)."""
    chunk_type = ttl_infer_chunk_type(question)

    country_names: set[str] = set()
    if chunk_type == "person":
        country_names = infer_country_names(question)

    if chunk_type:
        k = 100
    else:
        k = 5
    if country_names:
        k = 200

    results = await ttl_retrieve(
        question,
        k=k,
        chunk_type=chunk_type,
        country_names=country_names or None,
    )
    if not results["documents"][0]:
        return "(no results)", ""

    payload = await ttl_answer(
        question,
        results,
        chunk_type=chunk_type,
        country_names=country_names or None,
    )

    sources = []
    for i, m in enumerate(results["metadatas"][0], start=1):
        d = results["distances"][0][i - 1]
        sources.append(
            f"{i}. "
            f"{m.get('identifier', '')} "
            f"{m.get('chunk_type', '?')} "
            f"country={m.get('country', '')} "
            f"eId={m.get('eId', '?')} "
            f"(d={d:.3f})"
        )

    return payload["answer"], "\n".join(sources)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    started = datetime.datetime.now()
    run_id = started.strftime("%Y%m%d-%H%M%S")

    out_path = Path(f"tests/comparison_{run_id}.csv")

    print(f"Run id: {run_id}")
    print(f"Output: {out_path.resolve()}")
    print(f"Questions: {len(QUESTIONS)}")
    print()

    rows = []

    for i, q in enumerate(QUESTIONS, start=1):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"[{i}/{len(QUESTIONS)}] {q}")

        try:
            pdf_ans, pdf_src = await run_pdf(q)
        except Exception as e:
            pdf_ans, pdf_src = f"(error: {e})", ""

        try:
            ttl_ans, ttl_src = await run_ttl(q)
        except Exception as e:
            ttl_ans, ttl_src = f"(error: {e})", ""

        rows.append(
            {
                "timestamp": ts,
                "run_id": run_id,
                "question": q,
                "pdf_response": pdf_ans,
                "pdf_source": pdf_src,
                "ttl_response": ttl_ans,
                "ttl_source": ttl_src,
            }
        )

        print(f"  pdf: {pdf_ans[:80]!r}")
        print(f"  ttl: {ttl_ans[:80]!r}")
        print()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    finished = datetime.datetime.now()
    duration = (finished - started).total_seconds()

    print(
        f"Wrote {len(rows)} rows to {out_path.resolve()} "
        f"({duration:.1f}s)"
    )


if __name__ == "__main__":
    asyncio.run(main())