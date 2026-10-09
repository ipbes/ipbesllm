
import sys
from pathlib import Path

import streamlit as st

# This file lives in src/; make its sibling modules (cache.py, geo.py,
# thesaurus_helper.py ...) importable however Streamlit was launched.
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pdf_rag
import ttl_rag
import xml_rag
# Never asyncio.run() here: see cache.run_sync.
from cache import run_sync


st.set_page_config(
    page_title="IPBES RAG Workbench",
    page_icon="🔎",
    layout="wide",
)

st.title("IPBES RAG Workbench")
st.caption(
    "Test PDF, TTL and XML retrieval, inspect evidence, "
    "and evaluate grounded answers."
)


def run_pdf(question: str, k: int):
    """Run the existing PDF retrieval and generation pipeline."""
    country_names = pdf_rag.infer_country_names(question)

    # PDF country metadata is not currently indexed.
    # Preserve the existing pipeline's behavior and do not filter.
    country_names = set()

    results = run_sync(
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

    payload = run_sync(
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


def run_xml(question: str, k: int):
    """Run the XML (Akoma Ntoso) retrieval and generation pipeline."""
    results = xml_rag.retrieve(question, k=k)

    if not results["documents"][0]:
        return {
            "answer": "No relevant XML chunks were retrieved.",
            "results": results,
            "cached": False,
        }

    payload = run_sync(xml_rag.generate_answer_cached(question, results))

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def run_ttl(question: str, k: int):
    """Run the existing TTL retrieval and generation pipeline."""
    plan = ttl_rag.plan_query(question, k)
    chunk_type, country_names = plan["chunk_type"], plan["country_names"]
    per_assessment = plan["per_assessment"]

    results = run_sync(
        ttl_rag.retrieve_cached(
            question,
            k=plan["k"],
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

    payload = run_sync(
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
                f"{index + 1}. [{metadata.get('assessment', '?')}] "
                f"{metadata.get('source_file', 'Unknown file')} — "
                f"page {metadata.get('page', '?')}"
            )
        elif source == "XML":
            title = (
                f"{index + 1}. "
                f"{metadata.get('source_file', 'Unknown file')} — "
                f"{metadata.get('division') or metadata.get('eId', '?')}"
            )
        else:
            title = (
                f"{index + 1}. [{metadata.get('assessment', '?')}] "
                f"{metadata.get('identifier', 'No identifier')} — "
                f"{metadata.get('chunk_type', 'Unknown type')}"
            )

        with st.expander(title):
            if distance is not None:
                st.metric("Chroma distance", f"{distance:.4f}")

            if source == "PDF":
                keys = [
                    "assessment",
                    "source_file",
                    "title",
                    "page",
                    "page_count",
                    "chunk_index",
                    "chunk_total",
                    "country",
                ]
            elif source == "XML":
                keys = [
                    "source_file",
                    "document_type",
                    "title",
                    "date",
                    "division",
                    "subdivision",
                    "paragraph",
                    "table_title",
                    "table_row",
                    "eId",
                    "xpath",
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
        ["PDF", "TTL", "XML"],
        help=(
            "PDF and TTL search the assessments (GA1, IAS, LDR) with "
            "thesaurus expansion and re-ranking. XML searches the Akoma "
            "Ntoso plenary documents in data/xml."
        ),
    )

    k = st.slider(
        "Chunks per assessment (XML: chunks in total)",
        min_value=1,
        max_value=20,
        value=5,
        help=(
            "Each assessment (GA1, IAS, LDR) is searched separately, or only "
            "the ones the question names. For TTL, key-message, person and "
            "country questions use larger limits to list everything."
        ),
    )

    st.divider()
    st.caption("LLM model")
    pipeline = {"PDF": pdf_rag, "TTL": ttl_rag, "XML": xml_rag}[source]
    st.code(pipeline.LLM_MODEL)
    st.caption("Embedding model")
    st.code(pipeline.EMBED_MODEL)


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
                elif source == "XML":
                    output = run_xml(question.strip(), k)
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
