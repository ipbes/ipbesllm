import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from loguru import logger

# This file lives in src/; make its sibling modules (cache.py, geo.py,
# thesaurus_helper.py ...) importable however Streamlit was launched.
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import combined_rag
import pdf_rag
import ttl_rag
import xml_rag
# Never asyncio.run() here: see cache.run_sync.
from cache import run_sync
from rag_utils import (
    CHAT_KEEP_ALIVE, LLM_MODEL, NUM_CTX, USE_AZURE, AnswerStream, chat_client,
    embed_query,
)
from data_bundle import ensure_data
from feedback_store import save_rating
from thesaurus_helper import get_thesaurus


st.set_page_config(
    page_title="IBOK Chat",
    page_icon="🌿",
    layout="wide",
)

# Background photo (src/static/, served by enableStaticServing in
# .streamlit/config.toml), with the content on light panels so it stays
# readable.
st.markdown(
    """
    <style>
    [data-testid="stApp"] {
        background: url("./app/static/background.jpg") center / cover fixed;
    }
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1100px;
        margin: 4rem auto 1rem;
        padding: 2rem 2.5rem 1.5rem;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 1100px;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.90);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ALL is what users get; the single sources are for analysis (developer mode).
SOURCES = {
    "ALL": "IPBES Assessments and Plenary Reports",
    "PDF": "Assessments and Plenary Reports (PDF text)",
    "TTL": "Assessments (TTL/RDF)",
    "XML": "Plenary Reports (XML)",
}

SUGGESTED_QUESTIONS = {
    "assessments": [
        "What are the key messages of the IPBES LDR assessment?",
        "What do IPBES assessments say about pollution?",
        "What do IPBES assessments say about cities and urbanization?",
        "How do the assessments link climate change to biodiversity?",
        "How do the assessments link nature to poverty and food security?",
        "What does the IAS assessment say about the drivers of invasive alien species?",
        "What are the knowledge gaps identified in the assessments?",
        "What is NCP and how does it relate to nature?",
        "Which findings refer to the Kunming-Montreal Global Biodiversity Framework?",
        "What are the chapters of the IPBES LDR assessment?",
    ],
    "plenary": [
        "What did the Plenary decide about the work programme?",
        "What decisions were taken on the budget?",
        "Which assessments did the Plenary approve?",
        "What was decided about capacity-building?",
        "What did the Plenary decide about stakeholder engagement?",
        "What was decided about the next sessions of the Plenary?",
    ],
}


@st.cache_resource(show_spinner=False)
def start_warm_up():
    """Once per server process, load in the background what the first
    question would otherwise wait for: the thesaurus, the geography graph,
    the Chroma collections and both Ollama models."""
    def load_answer_models():
        if USE_AZURE:
            return          # nothing to load: the model runs in Azure
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


@st.cache_resource(show_spinner="Downloading the search data (first start only)…")
def prepare_data():
    """On a fresh deployment, fetch the Chroma database and RDF files."""
    ensure_data()


prepare_data()
start_warm_up()


def run_streamed(run, question: str, k: int, label: str):
    """Run a pipeline in a worker thread and show the answer as it streams.

    Streamlit only acts on Stop (or a new query, or a reload) when the script
    sends something to the page, so while waiting the status line is updated
    every second. When the script is stopped, the request to Ollama is cut,
    so it does not keep working on an answer nobody will see while the next
    question waits behind it.
    """
    pieces = queue.Queue()
    done = object()
    answer_stream = AnswerStream(pieces.put)

    future = pipeline_executor().submit(run, question, k, answer_stream)
    future.add_done_callback(lambda _: pieces.put(done))

    status = st.status(f"Searching {label}…")
    answer = st.empty()
    started = time.monotonic()
    text = ""
    try:
        while True:
            try:
                piece = pieces.get(timeout=1)
            except queue.Empty:
                piece = None
            if piece is done:
                break
            if piece:
                text += piece
                answer.markdown(text)
            elapsed = time.monotonic() - started
            status.update(
                label=f"Writing the answer… {elapsed:.0f} s" if text else
                f"Searching {label} and reading the sources… {elapsed:.0f} s"
            )
    finally:
        answer_stream.cancelled.set()

    output = future.result()
    output["seconds"] = time.monotonic() - started
    status.update(label=f"Done in {output['seconds']:.0f} s", state="complete")
    return output


def run_pdf(question: str, k: int, answer_stream=None):
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
            answer_stream=answer_stream,
        )
    )

    return {
        "answer": payload["answer"],
        "results": results,
        "cached": payload.get("_cached", False),
    }


def run_all(question: str, k: int, answer_stream=None):
    """Assessments and Plenary reports together (combined_rag.ask)."""
    return run_sync(combined_rag.ask(question, k, answer_stream=answer_stream))


def run_xml(question: str, k: int, answer_stream=None):
    """The XML (Akoma Ntoso) Plenary reports only."""
    return run_sync(combined_rag.ask_xml(question, k, answer_stream))


def run_ttl(question: str, k: int, answer_stream=None):
    """The TTL assessments only."""
    return run_sync(combined_rag.ask_ttl(question, k, answer_stream))


def source_title(source: str, number: int, metadata: dict) -> str:
    """Card title; `number` matches "SOURCE n" in the prompt."""
    if source == "PDF":
        name = metadata.get("title") or metadata.get("source_file", "Unknown file")
        return (f"{number}. [{metadata.get('assessment', '?')}] {name} — "
                f"page {metadata.get('page', '?')}")
    if source == "XML":
        name = (metadata.get("part_of") or metadata.get("title")
                or metadata.get("source_file", "Unknown file"))
        return (f"{number}. {name}"
                f" — {metadata.get('division') or metadata.get('eId', '?')}")
    return (f"{number}. [{metadata.get('assessment', '?')}] "
            f"{metadata.get('identifier', 'No identifier')} — "
            f"{metadata.get('chunk_type', 'Unknown type')}")


METADATA_KEYS = {
    "PDF": ["assessment", "source_file", "title", "page", "page_count",
            "chunk_index", "chunk_total", "country"],
    "XML": ["source_file", "document_type", "title", "date", "division",
            "subdivision", "paragraph", "table_title", "table_row", "eId", "xpath",
            "neighbour_of", "part_of"],
    "TTL": ["identifier", "chunk_type", "qualifier", "assessment", "country",
            "eId", "division", "subdivision", "xpath"],
}


# What each chunk is, from its source_type metadata (combined answers mix
# TTL and XML chunks); PDF chunks have none.
CHUNK_KINDS = {"ttl": "TTL", "xml": "XML"}


def render_sources(source: str, results, developer: bool):
    """The chunks the answer was based on, numbered as in the prompt."""
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return

    with st.expander(f"Sources ({len(documents)})"):
        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            kind = CHUNK_KINDS.get(metadata.get("source_type"), source)
            st.markdown(f"**{source_title(kind, index + 1, metadata)}**")
            if developer:
                visible = {key: metadata[key] for key in METADATA_KEYS[kind]
                           if metadata.get(key) is not None}
                if index < len(distances):
                    visible["distance"] = round(distances[index], 4)
                st.json(visible, expanded=False)
            st.text(document)
            st.divider()


def save_feedback(index: int):
    """Store the rating of answer `index` (feedback_store: Azure Blob when
    FEEDBACK_CONTAINER_URL is set, otherwise feedback/ratings.csv)."""
    message = st.session_state.messages[index]
    rating = st.session_state.get(f"feedback_{index}")
    if rating is None:
        return
    results = message.get("results") or {"documents": [[]]}
    try:
        save_rating(
            rating="up" if rating == 1 else "down",
            question=message["question"],
            answer=message["content"],
            source=message["source"],
            route=message.get("route"),
            model=message["model"],
            sources=len(results["documents"][0]),
        )
        st.toast("Thank you for your feedback.")
    except Exception:  # noqa: BLE001
        logger.exception("Could not store the rating")
        st.toast("Sorry, your rating could not be saved.")


def render_answer(index: int, message: dict, developer: bool):
    st.markdown(message["content"])
    if message.get("results") is not None:
        render_sources(message["source"], message["results"], developer)
    details = [SOURCES[message["source"]]]
    if developer:
        if message.get("route"):
            details.append(f"route: {message['route']}")
        details.append(message["model"])
        if message.get("seconds") is not None:
            details.append(f"{message['seconds']:.0f} s")
        if message.get("cached"):
            details.append("from cache")
    st.caption(" · ".join(details))
    st.feedback("thumbs", key=f"feedback_{index}",
                on_change=save_feedback, args=(index,))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestion_seed" not in st.session_state:
    st.session_state.suggestion_seed = random.random()

with st.sidebar:
    st.title("🌿 IBOK Chat")

    if st.button("New conversation", icon=":material/add:",
                 use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("About"):
        st.markdown(
            "Answers questions from machine readable versions of IPBES assessments (GA1, IAS, LDR, SUA, TCA, VA) and the reports of "
            "the IPBES Plenary sessions (IPBES 1 to IPBES 8), using only the retrieved text, and "
            "lists the sources it used. "
            "Each question is answered on its own: earlier questions are not "
            "taken into account.\n\n"
            "This is a pilot reseach aid. Please refer to original documents from www.ipbes.net for authoriative documents and citations."
        )

    developer = st.toggle("Developer mode",
                          help="Show retrieval settings, models and metadata.")
    source = "ALL"
    if developer:
        source = st.selectbox(
            "Search in",
            list(SOURCES),
            format_func=SOURCES.get,
            help=(
                "All: the assessments (TTL) and the Plenary reports (XML), "
                "routed per question. The single sources are for analysis."
            ),
        )
    # PDF, TTL and ALL search each assessment for k chunks; XML takes k in
    # total, and its paragraphs are short, so it needs a larger k (ALL uses
    # combined_rag.XML_K for the Plenary reports).
    k = 12 if source == "XML" else 3
    if developer:
        k = st.slider(
            "Chunks per assessment (XML: chunks in total)",
            min_value=1,
            max_value=20,
            value=k,
            help=(
                "Each assessment is searched separately, or "
                "only the ones the question names. For TTL, key-message, "
                "person and country questions use larger limits to list "
                "everything. More chunks make a longer prompt and a slower "
                "answer."
            ),
        )
        pipeline = {"PDF": pdf_rag, "TTL": ttl_rag, "XML": xml_rag}.get(source, ttl_rag)
        st.caption("LLM model" + (" (Azure OpenAI)" if USE_AZURE else " (Ollama)"))
        st.code(pipeline.LLM_MODEL)
        st.caption("Embedding model")
        st.code(pipeline.EMBED_MODEL)


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(index, message, developer)


question = st.chat_input("Ask a question about IPBES assessments (GA1, IAS, LDR, SUA, TCA, VA) and Plenary sessions (IPBES 1 to IPBES 8)")

# A new "Search in" choice applies to the next question; offer to ask the
# last one again there, rather than leaving its old answer looking current.
last = st.session_state.messages[-1] if st.session_state.messages else None
if not question and last and last["role"] == "assistant" and last["source"] != source:
    st.info(f"Answers above came from {SOURCES[last['source']]}. "
            f"New questions will search {SOURCES[source]}.")
    if st.button(f"Ask “{last['question']}” again in {SOURCES[source]}",
                 icon=":material/refresh:"):
        question = last["question"]

if not st.session_state.messages and not question:
    with st.chat_message("assistant"):
        st.markdown(
            "**Hi, I'm IBOK Chat.** Ask me about the IPBES "
            "assessments or plenary documents, or start with one of these:"
        )
    pool = {"XML": SUGGESTED_QUESTIONS["plenary"],
            "ALL": SUGGESTED_QUESTIONS["assessments"] + SUGGESTED_QUESTIONS["plenary"],
            }.get(source, SUGGESTED_QUESTIONS["assessments"])
    suggestions = random.Random(st.session_state.suggestion_seed).sample(pool, 4)
    columns = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if columns[i % 2].button(suggestion, key=f"suggestion_{i}",
                                 use_container_width=True):
            question = suggestion
    if not question and st.button("More suggestions", icon=":material/shuffle:"):
        st.session_state.suggestion_seed = random.random()
        st.rerun()

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        run = {"ALL": run_all, "PDF": run_pdf, "XML": run_xml,
               "TTL": run_ttl}[source]
        message = {"role": "assistant", "source": source, "question": question,
                   "model": LLM_MODEL}
        try:
            output = run_streamed(run, question, k, SOURCES[source])
            message.update(content=output["answer"], results=output["results"],
                           cached=output["cached"], seconds=output["seconds"],
                           route=output.get("route"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Query failed")
            message.update(content=f"Sorry, the query failed: {exc}", results=None)
        except SystemExit as exc:
            message.update(content=str(exc), results=None)

    st.session_state.messages.append(message)
    st.rerun()

st.caption(
    "The response is generated by AI and may contain inaccuracies. "
    "Please verify all information with reference documents."
)
