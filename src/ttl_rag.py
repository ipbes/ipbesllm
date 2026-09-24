import os
import re

import chromadb
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except (NotFoundError, ValueError) as e:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found. "
            f"Run ttl_index.py first. (Original error: {e})"
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str, k: int = 5, chunk_type: str | None = None):
    collection = _get_collection()
    where = {"chunk_type": chunk_type} if chunk_type else None
    return collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
    )


def _infer_chunk_type(question: str) -> str | None:
    """
    Infer a chunk_type filter from the question text.

    Keep this conservative: only return a type when the question clearly
    refers to one kind of chunk. Otherwise return None and let vector
    search decide.
    """
    q = question.lower()

    # Order matters: check the more specific phrases first.
    if "key message" in q or "key messages" in q:
        return "key"
    if "knowledge gap" in q or "knowledge gaps" in q:
        return "kg"
    if "sub-message" in q or "submessage" in q or "sub message" in q:
        return "subm"
    if "background message" in q:
        return "bgm"
    if "illustration" in q or "figure" in q:
        return "il"
    if "reference" in q or "citation" in q:
        return "ref"
    if "author" in q or "person" in q or "expert" in q:
        return "person"
    if "subchapter" in q or "sub-chapter" in q:
        return "sch"

    return None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^([A-Za-z]+)\s*(\d+)")


def _identifier_sort_key(identifier: str):
    """
    'A1.' -> ('A', 1)
    'B12' -> ('B', 12)
    'LDR18-A1.' -> ('LDR', 18)   # fallback if no plain identifier
    Otherwise: (identifier, 0)
    """
    m = _IDENTIFIER_RE.match(identifier or "")
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    return (identifier or "", 0)


def _reorder_by_identifier(results):
    """Reorder query() results in place by metadata['identifier']."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    order = sorted(
        range(len(docs)),
        key=lambda i: _identifier_sort_key(metas[i].get("identifier", "")),
    )

    results["documents"][0] = [docs[i] for i in order]
    results["metadatas"][0] = [metas[i] for i in order]
    results["distances"][0] = [dists[i] for i in order]
    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_context(results) -> str:
    parts = []

    for i, document in enumerate(results["documents"][0]):
        m = results["metadatas"][0][i]

        # Strip the loader's header block ("Document: ... Date: ...")
        # so the model sees the message body, not a metadata preamble.
        body = document.split("\n\n", 1)[-1].strip()

        prefix_parts = []
        if m.get("identifier"):
            prefix_parts.append(f"id={m['identifier']}")
        if m.get("chunk_type"):
            prefix_parts.append(f"type={m['chunk_type']}")
        if m.get("qualifier"):
            prefix_parts.append(f"qualifier={m['qualifier']}")

        prefix = " ".join(prefix_parts)

        parts.append(f"[{i + 1}] {prefix}\n{body}")

    return "\n\n---\n\n".join(parts)


def _build_task(question: str, chunk_type: str | None) -> str:
    if chunk_type == "key":
        return (
            "The user is asking for the KEY MESSAGES of an IPBES assessment.\n"
            "Every retrieved chunk below is a KeyMessage.\n"
            "\n"
            "List ALL retrieved KeyMessages, in the order given, numbered.\n"
            "For each one output:\n"
            "  - identifier (e.g. A1, B3)\n"
            "  - qualifier (well established / established but incomplete / "
            "unresolved) if present\n"
            "  - the key message text\n"
            "\n"
            "CRITICAL:\n"
            "- Do NOT answer any question that appears inside a key message.\n"
            "- Do NOT invent a heading or rephrase into your own question.\n"
            "- Do NOT summarise; reproduce each key message faithfully.\n"
        )

    if chunk_type:
        return (
            f"Answer the user's question using ONLY the retrieved "
            f"{chunk_type} chunks below.\n"
            "If the answer is not present, say "
            "'I cannot determine that from the supplied documents.'"
        )

    return (
        "Answer the user's question using ONLY the retrieved chunks below.\n"
        "If the answer is not present, say "
        "'I cannot determine that from the supplied documents.'"
    )


def generate_answer(question: str, results, chunk_type: str | None = None) -> str:
    context = _build_context(results)
    task = _build_task(question, chunk_type)

    prompt = f"""
You are answering questions about an IPBES assessment report, represented
as RDF/Turtle using the IPBES ontology.

{task}

Preserve any qualifier (well established / established but incomplete /
unresolved) verbatim. Do not invent facts, dates, names, numbers, or
decisions.

User question:
{question}

Retrieved chunks (already sorted by identifier):
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied IPBES ontology context. "
                    "Preserve evidence qualifiers. Follow the task "
                    "instructions exactly, including for list questions."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print()

    question = input("Question: ").strip()
    if not question:
        return

    chunk_type = _infer_chunk_type(question)

    if chunk_type:
        print(f"(Filtering to chunk_type='{chunk_type}')")
        # Retrieve everything of this type, so we don't truncate.
        k = 100
    else:
        k = 5

    print()

    results = retrieve(question, k=k, chunk_type=chunk_type)

    if not results["documents"][0]:
        print("No results found.")
        return

    # Sort by identifier so lists appear in document order (A1, A2, ...).
    if chunk_type:
        results = _reorder_by_identifier(results)

    answer = generate_answer(question, results, chunk_type=chunk_type)

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED TTL SOURCES")
    print("=" * 80)

    for i, m in enumerate(results["metadatas"][0]):
        d = results["distances"][0][i]
        print(
            f"{i + 1:>2}. {m.get('identifier', ''):<5} "
            f"{m.get('chunk_type', '?'):<8} "
            f"eId={m.get('eId', '?'):<20} "
            f"distance={d:.4f}"
        )


if __name__ == "__main__":
    main()