import os
import re
from pathlib import Path

import chromadb
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SKOS
from cache import cache, make_key 
from loguru import logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"

# Path to the IPBES geography vocabulary.
GEO_PATH = Path("data/rdf/ipbes-geo.rdf")

ISO3166 = URIRef("http://purl.org/dc/terms/ISO3166")


# ---------------------------------------------------------------------------
# Geography helpers (load ipbes-geo.rdf)
# ---------------------------------------------------------------------------

_geo_graph: Graph | None = None


def _geo() -> Graph:
    global _geo_graph
    if _geo_graph is None:
        g = Graph()
        g.parse(GEO_PATH, format="xml")
        _geo_graph = g
    return _geo_graph


def _labels_of_type(subject, predicate) -> list[str]:
    g = _geo()
    out = []
    for o in g.objects(subject, predicate):
        lang = getattr(o, "language", None)
        if lang in (None, "en"):
            out.append(str(o).strip())
    return out


def _en_labels(subject) -> list[str]:
    labels = []
    for p in (SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel):
        labels.extend(_labels_of_type(subject, p))
    return labels


def _country_concepts():
    """Yield (concept, iso_code) for every country in the geo file."""
    g = _geo()
    for concept in g.subjects(
        RDF.type, URIRef("http://www.w3.org/2004/02/skos/core#Concept")
    ):
        for notation in g.objects(concept, SKOS.notation):
            if getattr(notation, "datatype", None) == ISO3166:
                yield concept, str(notation)
                break


def country_code_for_label(label: str) -> str | None:
    """Map a free-text country label to its ISO 3166 alpha-3 code."""
    if not label:
        return None
    needle = label.strip().lower()

    for concept, code in _country_concepts():
        for lab in _en_labels(concept):
            if lab.lower() == needle:
                return code
        if str(concept).rsplit("/", 1)[-1].lower() == needle:
            return code

    return None


def country_label_for_code(code: str) -> str | None:
    """
    Map an ISO 3166 alpha-3 code back to the plain English country name
    that the TTL files actually use for ipbes:country.

    Preference order:
      1. skos:hiddenLabel  (e.g. 'Tanzania', 'Netherlands', 'Gambia')
      2. skos:prefLabel with any trailing parenthetical stripped
    """
    code = (code or "").upper()
    for concept, c in _country_concepts():
        if c != code:
            continue
        hidden = _labels_of_type(concept, SKOS.hiddenLabel)
        if hidden:
            return hidden[0]
        pref = _labels_of_type(concept, SKOS.prefLabel)
        if pref:
            short = re.sub(r"\s*\(.*\)\s*$", "", pref[0]).strip()
            return short or pref[0]
        return None
    return None


def _expand_collection(uri) -> set[str]:
    """Recursively collect ISO codes under a skos:Collection."""
    g = _geo()
    codes: set[str] = set()
    stack = [uri]
    seen = set()

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)

        for member in g.objects(node, SKOS.member):
            for notation in g.objects(member, SKOS.notation):
                if getattr(notation, "datatype", None) == ISO3166:
                    codes.add(str(notation))
                    break
            else:
                stack.append(member)

    return codes


def _codes_for_region(label: str) -> set[str]:
    g = _geo()
    needle = label.strip().lower()

    # 1. Exact match.
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            if lab.lower() == needle:
                return _expand_collection(coll)

    # 2. Prefix match: "east africa" -> "East Africa and adjacent islands"
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            if lab.lower().startswith(needle):
                return _expand_collection(coll)

    # 3. Substring match, but skip the four continent-level collections
    #    unless the needle itself is that continent's name.
    broad = {"africa", "americas", "asia", "europe",
             "europe and central asia", "asia and the pacific"}
    for coll in g.subjects(RDF.type, SKOS.Collection):
        for lab in _en_labels(coll):
            lab_l = lab.lower()
            if needle in lab_l and lab_l not in broad:
                return _expand_collection(coll)

    return set()


# Region/sub-region names to look for, longest first so that
# "Central Asia" wins over "Asia".
_REGION_NAMES = [
    "east africa and adjacent islands",
    "east africa",
    "central and western europe",
    "europe and central asia",
    "asia and the pacific",
    "western europe",
    "eastern europe",
    "central asia",
    "north africa",
    "north america",
    "north-east asia",
    "south america",
    "south asia",
    "south-east asia",
    "southern africa",
    "west africa",
    "western asia",
    "central africa",
    "mesoamerica",
    "caribbean",
    "oceania",
    "africa",
    "americas",
    "asia",
    "europe",
]


def _infer_country_names(question: str) -> set[str]:
    """
    Return the country *names* implied by the question, matching what is
    stored in the current Chroma index (e.g. 'Kenya', 'Tanzania').

    Internally uses the geo vocabulary: region -> set of ISO codes,
    then ISO code -> plain name.
    """
    q = question.lower()

    codes: set[str] = set()

    # 1. Region / sub-region (Pick the longest region name that appears in the question)
    matches = [n for n in _REGION_NAMES if n in q]
    matches.sort(key=len, reverse=True)
    for name in matches:
        codes = _codes_for_region(name)
        if codes:
            break

    # 2. Individual country.
    if not codes:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", question)
        candidates = list(tokens) + [
            f"{tokens[i]} {tokens[i + 1]}"
            for i in range(len(tokens) - 1)
        ]
        for cand in candidates:
            code = country_code_for_label(cand)
            if code:
                codes = {code}
                break

    if not codes:
        return set()

    names = set()
    for c in codes:
        label = country_label_for_code(c)
        if label:
            names.add(label)
    return names


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

def retrieve(
    question: str,
    k: int = 5,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
):
    collection = _get_collection()

    where_clauses = []
    if chunk_type:
        where_clauses.append({"chunk_type": chunk_type})
    if country_names:
        names = sorted(country_names)
        if len(names) == 1:
            where_clauses.append({"country": names[0]})
        else:
            where_clauses.append({"country": {"$in": names}})

    if not where_clauses:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    return collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
    )


# ---------------------------------------------------------------------------
# Chunk-type inference
# ---------------------------------------------------------------------------

def _infer_chunk_type(question: str) -> str | None:
    q = question.lower()

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
    m = _IDENTIFIER_RE.match(identifier or "")
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    return (identifier or "", 0)


def _reorder_by_identifier(results):
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

        # Drop only the document-level preamble lines, keep the rest
        # (Heading / Chapter / Subchapter / Country / Roles ...).
        lines = document.splitlines()
        keep = [
            ln for ln in lines
            if not ln.startswith("Document:")
            and not ln.startswith("Date:")
        ]
        body = "\n".join(keep).strip()

        prefix_parts = []
        if m.get("identifier"):
            prefix_parts.append(f"id={m['identifier']}")
        if m.get("chunk_type"):
            prefix_parts.append(f"type={m['chunk_type']}")
        if m.get("qualifier"):
            prefix_parts.append(f"qualifier={m['qualifier']}")
        if m.get("country"):
            prefix_parts.append(f"country={m['country']}")

        prefix = " ".join(prefix_parts)

        parts.append(f"[{i + 1}] {prefix}\n{body}")

    return "\n\n---\n\n".join(parts)


def _build_task(
    question: str,
    chunk_type: str | None,
    country_names: set[str],
) -> str:
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

    if chunk_type == "person" and country_names:
        names = ", ".join(sorted(country_names))
        return (
            f"The user is asking about experts from: {names}.\n"
            f"Every retrieved chunk is a Person whose country field is one "
            f"of those names.\n"
            "\n"
            "List ALL retrieved persons by their full name (the heading).\n"
            "For each, also give the country and the role(s) mentioned in "
            "the body, if any.\n"
            "\n"
            "CRITICAL:\n"
            "- Only include persons whose country matches one of the names "
            "above.\n"
            "- If a retrieved chunk's body shows a different country, drop "
            "it.\n"
            "- Do not invent names, countries, or roles.\n"
        )

    if chunk_type == "person":
        return (
            "The user is asking about experts.\n"
            "Each retrieved chunk is a Person. List the persons and the "
            "country and roles mentioned in the body.\n"
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


def generate_answer(
    question: str,
    results,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
) -> str:
    context = _build_context(results)
    task = _build_task(question, chunk_type, country_names or set())

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
# Caching
# ---------------------------------------------------------------------------

# ---------- retrieval (async) ----------

async def retrieve_cached(
    question: str,
    k: int = 5,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
):
    """
    Async, cached version of retrieve(). Reuses your existing sync
    retrieve() by running it in a thread so the event loop isn't blocked
    (chromadb + Ollama embedding are sync).
    """
    import anyio

    norm_countries = sorted(country_names) if country_names else None
    key = make_key("ttl:retrieve", question, k, chunk_type, norm_countries)

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"ttl retrieve HIT  q={question[:50]!r}")
        return hit

    result = await anyio.to_thread.run_sync(
        lambda: retrieve(question, k=k, chunk_type=chunk_type,
                         country_names=country_names)
    )
    await cache.set(key, result, ttl=3600)
    return result


# ---------- generation (async) ----------

async def generate_answer_cached(
    question: str,
    results,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
) -> dict:
    """
    Cache the LLM call keyed on (question, chunk_type, country_names,
    context-hash). We hash the context because different retrievals for
    the same question (e.g. after re-indexing) must not collide.
    """
    import anyio

    context = _build_context(results)
    ctx_hash = make_key("ctx", context)[:16]

    key = make_key(
        "ttl:answer",
        question,
        chunk_type,
        sorted(country_names) if country_names else None,
        ctx_hash,
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"ttl answer HIT  q={question[:50]!r}")
        hit["_cached"] = True
        return hit

    answer = await anyio.to_thread.run_sync(
        lambda: generate_answer(
            question, results,
            chunk_type=chunk_type,
            country_names=country_names,
        )
    )

    payload = {
        "answer": answer,
        "retrieved": [
            {
                "identifier": m.get("identifier"),
                "chunk_type": m.get("chunk_type"),
                "country": m.get("country"),
                "eId": m.get("eId"),
                "distance": results["distances"][0][i],
            }
            for i, m in enumerate(results["metadatas"][0])
        ],
        "_cached": False,
    }
    await cache.set(key, payload, ttl=60 * 60 * 6)
    return payload

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print()

    question = input("Question: ").strip()
    if not question:
        return

    chunk_type = _infer_chunk_type(question)

    country_names: set[str] = set()
    if chunk_type == "person":
        country_names = _infer_country_names(question)

    if chunk_type:
        print(f"(Filtering to chunk_type='{chunk_type}')")
        k = 100
    else:
        k = 5

    if country_names:
        print(f"(Filtering to countries: {sorted(country_names)})")
        k = 200

    print()

    results = await retrieve_cached(
        question,
        k=k,
        chunk_type=chunk_type,
        country_names=country_names or None,
    )

    if not results["documents"][0]:
        if country_names:
            print(f"No results found for countries {sorted(country_names)}.")
        else:
            print("No results found.")
        return

    if chunk_type:
        results = _reorder_by_identifier(results)

    payload = await generate_answer_cached(
        question,
        results,
        chunk_type=chunk_type,
        country_names=country_names,
    )

    if payload.get("_cached"):
        print("(cache hit)")

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(payload["answer"])

    print()
    print("=" * 80)
    print("RETRIEVED TTL SOURCES")
    print("=" * 80)

    for i, m in enumerate(results["metadatas"][0]):
        d = results["distances"][0][i]
        print(
            f"{i + 1:>2}. "
            f"{m.get('identifier', ''):<5} "
            f"{m.get('chunk_type', '?'):<8} "
            f"country={m.get('country', ''):<25} "
            f"eId={m.get('eId', '?'):<25} "
            f"distance={d:.4f}"
        )


def main():
    import asyncio
    asyncio.run(async_main())


if __name__ == "__main__":
    main()