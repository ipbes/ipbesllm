
import asyncio
import sys
from pathlib import Path

import streamlit as st

# Ensure existing modules such as cache.py, geo.py and thesaurus_helper.py
# can be imported when Streamlit is launched from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pdf_rag
import ttl_rag


st.set_page_config(
    page_title="IPBES RAG Workbench",
    page_icon="🔎",
    layout="wide",
)

st.title("IPBES RAG Workbench")
st.caption(
    "Test PDF and TTL retrieval, inspect evidence, "
    "and evaluate grounded answers."
)


def run_pdf(question: str, k: int):
    """Run the existing PDF retrieval and generation pipeline."""
    country_names = pdf_rag.infer_country_names(question)

    # PDF country metadata is not currently indexed.
    # Preserve the existing pipeline's behavior and do not filter.
    country_names = set()

    results = asyncio.run(
        pdf_rag.retrieve_cached(
            question,
            k=k,
            country_names=None,
        )
    )

    if not results["documents"][0]:
        return {
            "answer": "No relevant PDF chunks were retrieved.",
            "results": results,
            "cached": False,
        }

    payload = asyncio.run(
        pdf_rag.generate_answer_cached(
            question,
            results,
            country_names=country_names,
        )
    )

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def run_ttl(question: str, k: int):
    """Run the existing TTL retrieval and generation pipeline."""
    chunk_type = ttl_rag._infer_chunk_type(question)

    country_names = set()
    if chunk_type == "person":
        country_names = {
            canonical
            for canonical in (
                ttl_rag.canonical_country(name)
                for name in ttl_rag.infer_country_names(question)
            )
            if canonical
        }

    per_assessment = bool(chunk_type or country_names)

    # Retain the specialized retrieval behavior used by ttl_rag.main().
    if country_names:
        retrieval_k = 200
    elif chunk_type:
        retrieval_k = 100
    else:
        retrieval_k = k

    results = asyncio.run(
        ttl_rag.retrieve_cached(
            question,
            k=retrieval_k,
            chunk_type=chunk_type,
            country_names=country_names or None,
            per_assessment=per_assessment,
        )
    )

    if not results["documents"][0]:
        return {
            "answer": "No relevant TTL chunks were retrieved.",
            "results": results,
            "cached": False,
        }

    if chunk_type:
        results = ttl_rag._reorder_by_identifier(results)

    payload = asyncio.run(
        ttl_rag.generate_answer_cached(
            question,
            results,
            chunk_type=chunk_type,
            country_names=country_names,
        )
    )

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def render_sources(source: str, results):
    """Display the actual chunks used by the selected RAG pipeline."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    st.subheader(f"Retrieved evidence ({len(documents)} chunks)")

    if not documents:
        st.info("No source chunks are available.")
        return

    for index, document in enumerate(documents):
        metadata = metadatas[index] or {}
        distance = distances[index] if index < len(distances) else None

        if source == "PDF":
            title = (
                f"{index + 1}. "
                f"{metadata.get('source_file', 'Unknown file')} — "
                f"page {metadata.get('page', '?')}"
            )
        else:
            title = (
                f"{index + 1}. "
                f"{metadata.get('identifier', 'No identifier')} — "
                f"{metadata.get('chunk_type', 'Unknown type')}"
            )

        with st.expander(title):
            if distance is not None:
                st.metric("Chroma distance", f"{distance:.4f}")

            if source == "PDF":
                keys = [
                    "source_file",
                    "title",
                    "page",
                    "page_count",
                    "chunk_index",
                    "chunk_total",
                    "country",
                ]
            else:
                keys = [
                    "identifier",
                    "chunk_type",
                    "qualifier",
                    "assessment",
                    "country",
                    "eId",
                    "division",
                    "subdivision",
                    "xpath",
                ]

            visible_metadata = {
                key: metadata[key]
                for key in keys
                if metadata.get(key) is not None
            }

            if visible_metadata:
                st.json(visible_metadata)

            st.markdown("**Retrieved text**")
            st.text(document)


with st.sidebar:
    st.header("Query settings")

    source = st.selectbox(
        "Knowledge source",
        ["PDF", "TTL"],
        help=(
            "TTL queries can use RDF thesaurus expansion and reranking "
            "through your existing pipeline."
        ),
    )

    k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        help=(
            "For TTL, specialized chunk-type and country queries "
            "use the pipeline's existing larger retrieval limits."
        ),
    )

    st.divider()
    st.caption("LLM model")
    st.code(
        pdf_rag.LLM_MODEL if source == "PDF" else ttl_rag.LLM_MODEL
    )
    st.caption("Embedding model")
    st.code(
        pdf_rag.EMBED_MODEL if source == "PDF" else ttl_rag.EMBED_MODEL
    )


with st.form("rag_query_form"):
    question = st.text_area(
        "Ask a question about IPBES assessments",
        placeholder=(
            "For example: What are the key messages "
            "about biodiversity and ecosystem services?"
        ),
        height=120,
    )

    submitted = st.form_submit_button(
        "Run RAG query",
        type="primary",
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner(f"Running {source} retrieval and generation..."):
                if source == "PDF":
                    output = run_pdf(question.strip(), k)
                else:
                    output = run_ttl(question.strip(), k)

            st.session_state["rag_output"] = output
            st.session_state["rag_source"] = source
            st.session_state["rag_question"] = question.strip()

        except Exception as exc:
            st.error(f"RAG query failed: {exc}")
            st.exception(exc)

        except SystemExit as exc:
            st.error(str(exc))


if "rag_output" in st.session_state:
    output = st.session_state["rag_output"]

    st.divider()
    st.subheader("Question")
    st.write(st.session_state["rag_question"])

    if output.get("cached"):
        st.caption("Answer served from cache.")

    st.subheader("Generated answer")
    st.markdown(output["answer"])

    render_sources(
        st.session_state["rag_source"],
        output["results"],
    )
