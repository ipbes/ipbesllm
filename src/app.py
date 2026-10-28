import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

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
from assessment_catalogue import catalogue_block
from feedback_store import FEEDBACK_CONTAINER_URL, save_rating
from history_store import (
    delete_conversation, list_conversations, load_conversation, new_id,
    save_conversation, title_for, user_folder,
)
from quote_check import check_dois, check_quotes
from thesaurus_helper import get_thesaurus


st.set_page_config(
    page_title="IBOK Chat",
    page_icon="🌿",
    layout="wide",
)

# Page background: the IPBES teal by default, or APP_BACKGROUND=photo for
# the photo (files in src/static/ are served by enableStaticServing in
# .streamlit/config.toml). The content sits on light panels either way.
PAGE_BACKGROUND = (
    'url("./app/static/background.jpg") center / cover fixed'
    if os.getenv("APP_BACKGROUND") == "photo" else "#037473"
)

st.markdown(
    """
    <style>
    [data-testid="stApp"] {
        background: PAGE_BACKGROUND;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent;
    }
    /* Streamlit's top bar lies over the page. Give it the page colour so
       content visibly slides under it, and let clicks through everywhere
       except its own buttons: otherwise it silently blocks whatever is
       scrolled beneath it, such as the arrow of an open "Sources". */
    [data-testid="stHeader"] {
        background: HEADER_BACKGROUND;
    }
    [data-testid="stHeader"],
    [data-testid="stHeader"] * {
        pointer-events: none;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] a,
    [data-testid="stHeader"] [role="button"] {
        pointer-events: auto;
    }
    /* The question box: a solid black outline so it stands out. */
    [data-testid="stChatInput"] > div {
        border: 2px solid #000000 !important;
        background: #ffffff;
    }
    /* At least 2rem of background on each side, so the panel never runs
       into the (also light) sidebar. */
    [data-testid="stMainBlockContainer"] {
        max-width: min(1100px, calc(100% - 4rem));
        margin: 4rem auto 1rem;
        padding: 2rem 2.5rem 1.5rem;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: min(1100px, calc(100% - 4rem));
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.90);
    }
    /* Sidebar rows read as a menu, as in ChatGPT: labels on the left,
       one line, long conversation titles cut with an ellipsis. */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button > div {
        justify-content: flex-start;
    }
    [data-testid="stSidebar"] button p {
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """.replace("PAGE_BACKGROUND", PAGE_BACKGROUND)
       .replace("HEADER_BACKGROUND",
                "transparent" if PAGE_BACKGROUND.startswith("url") else PAGE_BACKGROUND),
    unsafe_allow_html=True,
)

# ALL is the default; the single sources are for analysis.
SOURCES = {
    "ALL": "IPBES Assessments and Plenary Reports",
    "PDF": "Assessments and Plenary Reports (PDF text)",
    "TTL": "Assessments (TTL/RDF)",
    "XML": "Plenary Reports (XML)",
}
# For the "Search scope" row in the sidebar.
SCOPE_LABELS = {"ALL": "All", "PDF": "PDF", "TTL": "TTL", "XML": "XML"}

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
        page = f"PDF page {metadata.get('page', '?')}"
        if metadata.get("printed_page"):
            page += f" (printed p. {metadata['printed_page']})"
        return f"{number}. [{metadata.get('assessment', '?')}] {name} — {page}"
    if source == "XML":
        name = (metadata.get("part_of") or metadata.get("title")
                or metadata.get("source_file", "Unknown file"))
        return (f"{number}. {name}"
                f" — {metadata.get('division') or metadata.get('eId', '?')}")
    return (f"{number}. [{metadata.get('assessment', '?')}] "
            f"{metadata.get('identifier', 'No identifier')} — "
            f"{metadata.get('chunk_type', 'Unknown type')}")


METADATA_KEYS = {
    "PDF": ["assessment", "source_file", "title", "page", "printed_page",
            "page_count", "chunk_index", "chunk_total", "file_sha256"],
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


def feedback_key(index: int) -> str:
    return f"feedback_{st.session_state.conversation_id}_{index}"


def save_feedback(index: int):
    """Store the rating of answer `index` (feedback_store: Azure Blob when
    FEEDBACK_CONTAINER_URL is set, otherwise feedback/ratings.csv)."""
    message = st.session_state.messages[index]
    rating = st.session_state.get(feedback_key(index))
    if rating is None:
        return
    message["rating"] = rating
    remember_conversation()
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


def quote_report(question: str, output: dict) -> dict:
    """The answer's quotations checked against all the text the model was
    given: the retrieved passages, the reference list and the thesaurus."""
    documents = output["results"]["documents"][0]
    glossary, definitions = combined_rag._thesaurus_blocks(question, output["results"])
    sources = documents + [catalogue_block(), glossary, definitions]
    found, missing = check_quotes(output["answer"], sources)
    dois_found, dois_missing = check_dois(output["answer"], sources)
    return {"found": len(found), "missing": missing,
            "dois_found": len(dois_found), "dois_missing": dois_missing}


def render_answer(index: int, message: dict, developer: bool):
    st.markdown(message["content"])
    quotes = message.get("quotes") or {}
    unverified = ([f"- “{q}”" for q in quotes.get("missing", [])]
                  + [f"- DOI {d}" for d in quotes.get("dois_missing", [])])
    if unverified:
        st.warning("Not found word for word in the sources; check before "
                   "using:\n" + "\n".join(unverified),
                   icon=":material/format_quote:")
    if message.get("results") is not None:
        render_sources(message["source"], message["results"], developer)
    details = [SOURCES[message["source"]]]
    checked = [f"{n} {label}{'s' if n != 1 else ''}"
               for n, label in ((quotes.get("found", 0), "quote"),
                                (quotes.get("dois_found", 0), "DOI")) if n]
    if checked:
        details.append(" and ".join(checked) + " checked against the sources")
    if developer:
        if message.get("route"):
            details.append(f"route: {message['route']}")
        details.append(message["model"])
        if message.get("seconds") is not None:
            details.append(f"{message['seconds']:.0f} s")
        if message.get("cached"):
            details.append("from cache")
    st.caption(" · ".join(details))
    st.feedback("thumbs", key=feedback_key(index), default=message.get("rating"),
                on_change=save_feedback, args=(index,))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def history_folder() -> str | None:
    """Where this user's conversations are kept, or None to keep them for
    this session only: a hosted app that does not say who the viewer is
    must not show everyone the same history. Run locally (no storage
    container, or opened on localhost), there is one user."""
    try:
        email = st.user.get("email")
    except Exception:  # noqa: BLE001
        email = None
    if email:
        return user_folder(email)
    try:
        host = urlsplit(st.context.url).hostname
    except Exception:  # noqa: BLE001
        host = None
    if not FEEDBACK_CONTAINER_URL or host in LOCAL_HOSTS:
        return user_folder("local")
    return None


def remember_conversation():
    """Save the current conversation and put it at the top of the list."""
    state = st.session_state
    messages = state.messages
    if not any(m["role"] == "assistant" for m in messages):
        return
    if state.conversation_id is None:
        state.conversation_id = new_id()
    if state.history_folder is None:
        state.session_conversations[state.conversation_id] = list(messages)
        state.conversations = [
            {"id": state.conversation_id, "title": title_for(messages[0]["content"])}
        ] + [c for c in state.conversations if c["id"] != state.conversation_id]
        return
    try:
        state.conversations = save_conversation(
            state.history_folder, state.conversation_id, messages)
    except Exception:  # noqa: BLE001
        logger.exception("Could not save the conversation")
        st.toast("Sorry, this conversation could not be saved.")


def start_conversation():
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.suggestion_seed = random.random()


def open_conversation(conversation_id: str):
    state = st.session_state
    try:
        if state.history_folder is None:
            messages = state.session_conversations[conversation_id]
        else:
            messages = load_conversation(state.history_folder, conversation_id)
    except Exception:  # noqa: BLE001
        logger.exception("Could not load the conversation")
        st.toast("Sorry, this conversation could not be opened.")
        return
    state.messages = list(messages)
    state.conversation_id = conversation_id


def forget_conversation(conversation_id: str):
    state = st.session_state
    if state.history_folder is None:
        state.session_conversations.pop(conversation_id, None)
        state.conversations = [c for c in state.conversations
                               if c["id"] != conversation_id]
    else:
        try:
            state.conversations = delete_conversation(
                state.history_folder, conversation_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not delete the conversation")
            st.toast("Sorry, this conversation could not be deleted.")
            return
    if state.conversation_id == conversation_id:
        start_conversation()


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.suggestion_seed = random.random()
    st.session_state.history_folder = history_folder()
    st.session_state.session_conversations = {}
    st.session_state.conversations = []
    if st.session_state.history_folder:
        try:
            st.session_state.conversations = list_conversations(
                st.session_state.history_folder)
        except Exception:  # noqa: BLE001
            logger.exception("Could not read the conversation list")

# The name as a wordmark, top left: dark in the open sidebar, white on the
# page colour when the sidebar is closed.
st.logo(str(SRC_DIR / "static" / "ibok-chat-dark.svg"), size="large",
        icon_image=str(SRC_DIR / "static" / "ibok-chat-light.svg"))

with st.sidebar:
    st.button("New conversation", icon=":material/edit_square:", type="tertiary",
              use_container_width=True, on_click=start_conversation)

    source = st.session_state.get("source", "ALL")
    with st.popover(f"Search scope · {SCOPE_LABELS[source]}",
                    icon=":material/travel_explore:", type="tertiary",
                    use_container_width=True):
        source = st.selectbox(
            "Search in",
            list(SOURCES),
            format_func=SOURCES.get,
            key="source",
            help=(
                "All: the assessments (TTL) and the Plenary reports (XML), "
                "routed per question. The single sources are for analysis."
            ),
        )

    # PDF, TTL and ALL search each assessment for k chunks; XML takes k in
    # total, and its paragraphs are short, so it needs a larger k (ALL uses
    # combined_rag.XML_K for the Plenary reports). One setting per source.
    k_key = f"k_{source}"
    k = st.session_state.get(k_key, 12 if source == "XML" else 3)
    with st.popover(f"Chunks · {k}", icon=":material/segment:", type="tertiary",
                    use_container_width=True):
        k = st.slider(
            "Chunks per assessment (XML: chunks in total)",
            min_value=1,
            max_value=20,
            value=k,
            key=k_key,
            help=(
                "Each assessment is searched separately, or "
                "only the ones the question names. For TTL, key-message, "
                "person and country questions use larger limits to list "
                "everything. More chunks make a longer prompt and a slower "
                "answer."
            ),
        )

    # For information only: the models are set by the deployment.
    pipeline = {"PDF": pdf_rag, "TTL": ttl_rag, "XML": xml_rag}.get(source, ttl_rag)
    with st.popover("Model", icon=":material/memory:", type="tertiary",
                    use_container_width=True):
        st.caption("Answers" + (" (Azure OpenAI)" if USE_AZURE else " (Ollama)"))
        st.code(pipeline.LLM_MODEL)
        st.caption("Embeddings")
        st.code(pipeline.EMBED_MODEL)
    developer = st.toggle("Developer mode",
                          help="Show routes, timings and chunk metadata.")

    st.caption(":material/history: Recent conversations")
    if not st.session_state.conversations:
        st.caption("Your conversations will appear here.")
    for conversation in st.session_state.conversations:
        current = conversation["id"] == st.session_state.conversation_id
        title_column, delete_column = st.columns(
            [0.85, 0.15], gap=None, vertical_alignment="center")
        title_column.button(
            conversation["title"], key=f"open_{conversation['id']}",
            type="secondary" if current else "tertiary",
            help=conversation["title"], use_container_width=True,
            on_click=open_conversation, args=(conversation["id"],))
        delete_column.button(
            "", icon=":material/delete:", key=f"delete_{conversation['id']}",
            type="tertiary", help="Delete this conversation",
            on_click=forget_conversation, args=(conversation["id"],))
    if st.session_state.history_folder is None:
        st.caption("Kept until you close or reload the page.")

    st.divider()
    with st.expander("About"):
        st.markdown(
            "Answers questions from machine readable versions of IPBES assessments (BBA, GA1, IAS, LDR, NXS, PPA, SCM, SUA, TCA, VA) and the reports of "
            "the IPBES Plenary meeting reports and decisions (IPBES 1 to IPBES 12), using only the retrieved text, and "
            "lists the sources it used. "
            "Each question is answered on its own: earlier questions are not "
            "taken into account.\n\n"
            "This is a pilot reseach aid. Please refer to original documents from www.ipbes.net for authoriative documents and citations."
        )


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(index, message, developer)


question = st.chat_input("Ask a question about IPBES assessments (BBA, GA1, IAS, LDR, NXS, PPA, SCM, SUA, TCA, VA) and Plenary sessions (IPBES 1 to IPBES 12)")

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
            # Only what render_sources shows, so saved conversations stay small.
            results = {key: output["results"].get(key)
                       for key in ("documents", "metadatas", "distances")
                       if key in output["results"]}
            message.update(content=output["answer"], results=results,
                           cached=output["cached"], seconds=output["seconds"],
                           route=output.get("route"),
                           quotes=quote_report(question, output))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Query failed")
            message.update(content=f"Sorry, the query failed: {exc}", results=None)
        except SystemExit as exc:
            message.update(content=str(exc), results=None)

    st.session_state.messages.append(message)
    remember_conversation()
    st.rerun()

st.caption(
    "The response is generated by AI and may contain inaccuracies. "
    "Please verify all information with reference documents."
)
