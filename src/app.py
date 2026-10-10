
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from loguru import logger

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
from rag_utils import (
    CHAT_KEEP_ALIVE, NUM_CTX, GenerationCancelled, chat_client, embed_query,
)
from thesaurus_helper import get_thesaurus


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


@st.cache_resource(show_spinner=False)
def start_warm_up():
    """Once per server process, load in the background what the first
    question would otherwise wait for: the thesaurus, the geography graph,
    the Chroma collections and both Ollama models."""
    def load_answer_models():
        for model in {pdf_rag.LLM_MODEL, ttl_rag.LLM_MODEL, xml_rag.LLM_MODEL}:
            # An empty prompt only loads the model. Same num_ctx as the
            # answers, or Ollama would reload it for the first question.
            chat_client().generate(model=model, prompt="",
                                   keep_alive=CHAT_KEEP_ALIVE,
                                   options={"num_ctx": NUM_CTX})

    steps = [
        ("thesaurus", get_thesaurus),
        ("geography", lambda: pdf_rag.infer_country_names("warm up")),
        ("PDF collection", pdf_rag._get_collection),
        ("TTL collection", ttl_rag._get_collection),
        ("embedding model", lambda: embed_query("warm up")),
        ("answer model", load_answer_models),
    ]

    def warm():
        for name, step in steps:
            try:
                step()
            except (Exception, SystemExit) as exc:  # noqa: BLE001
                logger.warning(f"Warm-up of {name} failed: {exc}")
        logger.info("Warm-up finished.")

    threading.Thread(target=warm, name="warm-up", daemon=True).start()


@st.cache_resource(show_spinner=False)
def pipeline_executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag")


start_warm_up()


def run_streamed(run, question: str, k: int):
    """Run a pipeline in a worker thread and show the answer as it streams.

    If the script stops first (a new query, a page reload), the generation is
    cancelled, so Ollama does not keep working on an answer nobody will see
    while the next question waits behind it.
    """
    pieces = queue.Queue()
    done = object()
    cancelled = threading.Event()

    def on_token(piece: str):
        if cancelled.is_set():
            raise GenerationCancelled()
        pieces.put(piece)

    future = pipeline_executor().submit(run, question, k, on_token)
    future.add_done_callback(lambda _: pieces.put(done))

    def stream():
        while (piece := pieces.get()) is not done:
            yield piece

    live = st.empty()
    try:
        with live.container():
            st.subheader("Generated answer")
            st.write_stream(stream())
    finally:
        cancelled.set()

    output = future.result()
    live.empty()
    return output


def run_pdf(question: str, k: int, on_token=None):
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
            on_token=on_token,
        )
    )

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def run_xml(question: str, k: int, on_token=None):
    """Run the XML (Akoma Ntoso) retrieval and generation pipeline."""
    results = xml_rag.retrieve(question, k=k)

    if not results["documents"][0]:
        return {
            "answer": "No relevant XML chunks were retrieved.",
            "results": results,
            "cached": False,
        }

    payload = run_sync(xml_rag.generate_answer_cached(question, results, on_token=on_token))

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def run_ttl(question: str, k: int, on_token=None):
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
            on_token=on_token,
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
        value=3,
        help=(
            "Each assessment (GA1, IAS, LDR) is searched separately, or only "
            "the ones the question names. For TTL, key-message, person and "
            "country questions use larger limits to list everything. "
            "More chunks make a longer prompt and a slower answer."
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
            run = {"PDF": run_pdf, "XML": run_xml, "TTL": run_ttl}[source]
            with st.spinner(f"Running {source} retrieval and generation..."):
                output = run_streamed(run, question.strip(), k)

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
