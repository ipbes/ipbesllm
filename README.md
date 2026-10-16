# IBOK Chat: IPBES document RAG

A Retrieval-Augmented Generation (RAG) application for exploring and comparing IPBES documents, and the code behind **IBOK Chat**, its Streamlit chat app.

It is a tool and is not meant to replace reading the PDF files, which remain the source of truth for quotes, page numbers and provenance. It builds on the work of converting the assessments and Plenary reports into machine-readable formats with mapped semantic relations.

The project is an evaluation platform for comparing and improving retrieval and answer generation across PDF, XML and RDF/TTL representations of IPBES knowledge. 

## Contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Running the app](#running-the-app)
- [How questions are answered](#how-questions-are-answered)
- [Pipelines](#pipelines)
- [Rebuilding the collections](#rebuilding-the-collections)
- [Answer caching](#answer-caching)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Evaluation](#evaluation)
- [Development workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Contributing and colleague testing](#contributing-and-colleague-testing)
- [Current status](#current-status)
- [License and data](#license-and-data)

---

## Overview

### Source formats

| Format | Content | Processing | Chroma collection |
|---|---|---|---|
| **TTL** (RDF/Turtle) | Assessments: GA1, IAS, LDR, SUA, TCA, VA, BBA, PPA, SCM | Parsed with RDFLib into typed chunks (key messages, chapters, sub-chapters, experts, figures, knowledge gaps …), then embedded | `ttl_documents` |
| **XML** (Akoma Ntoso / LegalDocML) | Plenary session reports IPBES-1 to IPBES-8, with their decisions | Parsed into structured paragraphs and table rows, then embedded | `xml_documents` |
| **PDF** | Assessment reports and SPMs of the same assessments in TTL folder | Text extracted with PyMuPDF, chunked, then embedded | `pdf_documents` |
| **RDF** | IPBES thesaurus and geography | Used for query expansion, definitions and country matching; the thesaurus is also embedded | `ipbes_thesaurus` |

The collections are kept separate, which keeps them comparable for analysis. The app searches the TTL and XML collections together by default, so users do not have to know where an answer is stored (see [How questions are answered](#how-questions-are-answered)).

### Models

The application uses **Azure OpenAI** models, deployed in Microsoft Foundry:

| Role | Model (Azure deployment) | Setting |
|---|---|---|
| Answers | `gpt-5.4` | `AZURE_OPENAI_DEPLOYMENT` |
| Embeddings (documents and questions) | `text-embedding-3-small` | `AZURE_OPENAI_EMBED_DEPLOYMENT` |

Without the Azure settings, the code falls back to local **Ollama** models (`llama3.1:latest` and `nomic-embed-text`) for offline development; see [Optional: Ollama instead of Azure](#optional-ollama-instead-of-azure). The collections must be indexed with the same embedding model that questions are embedded with; see [Switching the embedding model](#switching-the-embedding-model).

### Pipeline

```text
Documents
   │
   ├── TTL ──► RDFLib parser [key messages, chapters, experts …] ──► Chunks ──► Embeddings ──► Chroma ─┐
   │                                                                                                  │
   ├── XML ──► Akoma Ntoso parser [paragraphs, tables, decisions] ──► Chunks ──► Embeddings ──► Chroma ─┼──► Retrieval ──► LLM (Azure OpenAI) ──► Answer + sources
   │                                                                                                  │
   └── PDF ──► Text extraction ──► Chunks ──► Embeddings ──► Chroma ─────────────────────────────────┘
```

The system preserves source provenance. Answers and the app's source list can include:

- source document and, for Plenary reports, the decision a passage is part of
- PDF page
- XML section, subsection, paragraph, eId and XPath
- table title, header and row
- assessment, chapter and identifier (TTL)
- retrieval distance, response time and cache status (developer mode)

The application answers only from retrieved source material. If the context does not establish an answer, the prompts instruct the model to say:

> I cannot determine that from the supplied documents.

---

## Project structure

```text
ipbesllm/
├── .streamlit/config.toml       # Static file serving, light theme
├── .env                         # Local settings and secrets (never committed)
├── chroma/                      # Chroma vector database and index manifests (not in git)
├── data/                        # Source documents (not in git; add locally)
│   ├── pdf/*.pdf                # <assessment>_report.pdf, <assessment>_spm.pdf
│   ├── rdf/*.rdf                # ipbes-thesaurus.rdf, ipbes-geo.rdf
│   ├── ttl/*.ttl                # e.g. GA1_v11.ttl: the assessment is the name before "_v"
│   └── xml/*.xml                # Plenary reports, e.g. 2013-1.xml
├── dist/ipbes-data.zip          # Data bundle for the deployed app (built by data_bundle.py)
├── docs/IPBES_Ontology_v06.ttl  # IPBES report ontology: the structure of the TTL files
├── feedback/ratings.csv         # Answer ratings, when not stored in Azure
├── tests/comparison_<run>.csv   # Written by pdf_ttl_compare.py (not in git)
├── src/
│   ├── app.py                   # IBOK Chat (Streamlit)
│   ├── static/background.jpg    # App background
│   ├── combined_rag.py          # One search over TTL and XML, for the app
│   ├── pdf_rag.py               # PDF retrieval and answers
│   ├── ttl_rag.py               # TTL retrieval and answers
│   ├── xml_rag.py               # XML retrieval and answers
│   ├── pdf_index.py             # Index PDFs into Chroma
│   ├── ttl_index.py             # Index TTL files into Chroma
│   ├── xml_index.py             # Index XML files into Chroma
│   ├── rdf_index.py             # Index the thesaurus concept profiles into Chroma
│   ├── pdf_loader.py            # PDF text extraction and chunking
│   ├── ttl_loader.py            # TTL parsing into chunks
│   ├── ttl_audit.py             # Check TTL files against the ontology
│   ├── xml_loader.py            # Akoma Ntoso parsing into chunks
│   ├── chunking.py              # Splits oversized TTL chunks
│   ├── rag_utils.py             # Models, embeddings, Chroma client, batching helpers
│   ├── thesaurus_helper.py      # Query expansion, re-ranking and definitions from the thesaurus
│   ├── enrichment.ttl           # Local SKOS additions to the IPBES thesaurus
│   ├── geo.py                   # Country and region lookup (ipbes-geo.rdf)
│   ├── cache.py                 # Answer and retrieval cache (Redis or in-memory)
│   ├── feedback_store.py        # Answer ratings (Azure Blob or a local CSV)
│   ├── data_bundle.py           # Packs chroma/ and the RDF for the deployed app
│   ├── settings.py              # Project paths; loads .env
│   ├── pdf_ttl_compare.py       # Runs a question set through PDF and TTL
│   ├── *_query.py               # Retrieval only, without an answer (pdf, ttl, xml, rdf)
│   ├── rdf_rag.py, rdf_graph.py, rdf_loader.py   # Thesaurus RAG (hybrid vector + graph)
│   ├── test_xml_loader.py       # Regression tests for the XML parser (pytest)
│   └── *_test.py, xml_test_chunks.py, diagnose_chunks.py, ...   # Sanity checks and diagnostics
├── README.md
└── requirements.txt
```

Some files may be added or renamed as the project develops. The important separation is:

| Location | Holds |
|---|---|
| `data/` | Source documents |
| `docs/` | Guidance documents e.g. ontology |
| `src/` | Application and RAG code |
| `chroma/` | Vector database, plus the index manifests the RAG code reads |
| Redis | Cached retrievals and answers (in-memory when `REDIS_URL` is not set) |

---

## Quick start

### 1. Requirements

- macOS, Linux or Windows
- Python 3.11
- Git
- An Azure OpenAI (Microsoft Foundry) resource with deployments of `gpt-5.4` and `text-embedding-3-small`
- Optionally Ollama, to work offline

### 2. Clone the repository

```bash
git clone git@github.com:ipbes/ipbesllm.git
cd ipbesllm
```

### 3. Create a virtual environment

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists only the packages the code imports, pinned; pip installs their dependencies. When the code starts using a new package, add it there by hand. Do not overwrite the file with `pip freeze`, which pins every sub-dependency and pulls in whatever else happens to be installed.

### 5. Configure Azure OpenAI

Create a `.env` file in the project root. It is listed in `.gitignore`: never commit it, because it holds the key.

```text
AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
```

The endpoint, deployment names and key are on the deployment's page in the Foundry portal. The endpoint also works as copied, with `/responses` on the end. Environment variables of the same name override `.env`.

### 6. Add the data and build the indexes

Put the source files in `data/` (see [Project structure](#project-structure)), then build the collections; see [Pipelines](#pipelines). With Azure, indexing everything takes about 30 minutes.

### Settings reference

| Setting | Default | Purpose |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | — | Azure OpenAI resource; without them, answers come from Ollama |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.4` | Answer model |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | — | Embedding model; without it, embeddings come from Ollama |
| `AZURE_OPENAI_REASONING_EFFORT` | `low` | How long GPT-5 models think before answering (`low`, `medium`, `high`) |
| `AZURE_OPENAI_EMBED_DIMENSIONS` | `768` | Embedding size; changing it means re-indexing everything |
| `XML_MAX_CONTEXT_CHARS`, `COMBINED_MAX_CONTEXT_CHARS` | `60000` with Azure | Prompt size limit for XML and combined answers |
| `RAG_MAX_CONTEXT_CHARS` | `20000` | Prompt size limit for PDF and TTL answers |
| `REDIS_URL` | — | Redis for the cache; in-memory without it |
| `RAG_CACHE_VERSION` | `v1` | Change to ignore all cached entries |
| `DATA_BUNDLE_URL` | — | Deployed app: where to download the data bundle |
| `FEEDBACK_CONTAINER_URL` | — | Where to store answer ratings (Azure Blob container) |
| `LOCAL_EMBEDDINGS` | — | `1`: embed questions in the app with nomic-embed-text, for data indexed with Ollama |
| `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_URL` | `llama3.1:latest`, `nomic-embed-text`, `http://localhost:11434` | Ollama models and server |

### Optional: Ollama instead of Azure

For offline development, the code uses local Ollama models when the Azure settings are absent: for answers without `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`, and for embeddings without `AZURE_OPENAI_EMBED_DEPLOYMENT`. Install Ollama from <https://ollama.com/> and pull the models:

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```

On a CPU-only laptop, answers take minutes instead of seconds. The collections must then be indexed with `nomic-embed-text` too: vectors from different embedding models do not mix (see [Switching the embedding model](#switching-the-embedding-model)).

---

## Running the app

```bash
streamlit run src/app.py
```

Paths to `chroma/` and `data/` are resolved from the project root (`src/settings.py`), so the app can be started from any directory. On start-up it loads the thesaurus, the geography data and the collections in the background, so the first question does not wait for them.

- **Chat.** Users ask in a chat box or pick a suggested question. Answers stream in as they are written and cite their sources by number, e.g. [3], matching the numbered list under "Sources". Each question is answered on its own; earlier questions are not taken into account.
- **Ratings.** 👍/👎 under each answer adds a row to `ratings.csv` (see [Answer ratings](#answer-ratings)).
- **Developer mode** (sidebar toggle). Choose a single source (PDF, TTL or XML) instead of the combined search, set the number of chunks, and see the models, the route taken, distances, metadata, timings and cache status.
- **Stop.** Stopping a question, asking a new one or reloading the page cancels the request to the model.

---

## How questions are answered

### Routing (`combined_rag.py`)

The app sends each question one of three ways:

| Question | Searched | Example |
|---|---|---|
| Names a Plenary session | That session's report only, with the XML prompt | "Who was elected chair at IPBES-1?" |
| A list question the TTL data answers completely: key messages, background messages, sub-messages, chapters, sub-chapters, knowledge gaps, figures, references, experts | The assessments only, with the TTL list prompts | "What are the chapters of the BBA assessment?" |
| Anything else | Assessments and Plenary reports together; one answer that says where each point comes from | "What do IPBES documents say about invasive alien species?" |

### Assessments (TTL and PDF)

- **One search per assessment.** Each assessment is searched separately (3 chunks each by default) and the results are interleaved by rank, so the largest assessment cannot crowd out the others. If the question names assessments, by code or by name ("the LDR assessment", "the global and IAS assessments"), only those are searched; a topic alone ("invasive alien species") still searches them all. For PDFs, the assessment comes from the file name (`ga1_report.pdf` is GA1).
- **List questions** fetch every chunk of that type (up to 100 per assessment), sorted by identifier, and the prompt asks for all of them.
- **Knowledge gaps** are listed as on [IBOK](https://ibok.ipbes.net/en/kg): a heading per assessment (year, title, number of gaps), newest first, then a table of #, identifier, category and description. The tables are built from the data, so counts and descriptions are exact; the model only chooses which gaps answer the question (all, those of named assessments, or those on a topic). Categories come from `ipbes:hasCategory` in the TTL, which only some files have.
- **Experts by country.** "Experts from Kenya" or "from East Africa" filters experts by country code. Regions come from the geography RDF.
- **Search text.** Words that describe the whole corpus ("IPBES", "assessments", "what do … say about") are removed before searching, because they match every chunk and drown out the topic. The model still gets the question as asked.
- **Thesaurus.** Concepts named in the question add synonyms and neighbouring terms to the search, re-rank the results, and put their definitions into the prompt ("Authoritative IPBES definitions").
- **Inferred links.** Where the thesaurus has no broader, narrower or related links (Pollution and Poverty have none), `thesaurus_helper.py` infers them from words: concepts sharing the name (*Soil pollution*), concepts named in the definition, and concepts whose definitions mention it (*Direct drivers*). They are labelled "not curated" in the prompt.
- **Enrichment.** `src/enrichment.ttl` holds local SKOS additions loaded on top of the official thesaurus, e.g. `skos:altLabel "cities"` on *Urban*, which has no "City" concept. Add curated `skos:altLabel` or `skos:related` triples there; they take effect at the next start.

### Plenary reports (XML)

- **Sessions.** "IPBES-1", "IPBES 1", "IPBES1", "#IPBES1", "first session", "1st session" and "session 1" all name the first session. A named session limits the search to its report, and the wording is rewritten as the reports phrase it ("the first session of the Plenary").
- **Neighbouring paragraphs.** 12 passages are retrieved, and each brings the paragraph before and after it in the same section. A paragraph that is mostly a list (e.g. the officers elected) matches questions poorly, but the one next to it often matches well.
- **Decisions.** Passages inside a decision or annex carry its name ("Part of: Decision IPBES-3/4: Communications, stakeholder engagement and strategic partnership"), so the search, the model and the source list can identify the decision.

---

## Pipelines

Each pipeline can also be used from the command line: index, inspect the retrieval, or ask a question.

### TTL / RDF (assessments)

Place Turtle files in `data/ttl/`. The assessment is the file name before `_v` (`GA1_v11.ttl` is GA1); keep one version of each assessment in the folder.

The files follow the IPBES report ontology in [`docs/IPBES_Ontology_v06.ttl`](docs/IPBES_Ontology_v06.ttl), which lists, for each class, the properties an item should have:

| Class | Chunk type | Main properties read |
|---|---|---|
| `ipbes:KeyMessage` | `key` | `skos:prefLabel`, `hasDescription`, `dcterms:identifier`, confidence level |
| `ipbes:BackgroundMessage` | `bgm` | its description plus the text of its sub-messages, confidence level |
| `ipbes:SubMessage` | `subm` | `hasDescription`, `dcterms:identifier`, confidence level |
| `ipbes:KnowledgeGap` | `kg` | `hasDescription`, `hasCategory` (not in the ontology; only GA1 and BBA have it); the gap's own ID (`GA119-1`) is its identifier, and `dcterms:identifier` holds the related sections |
| `ipbes:Chapter` | `chapter` | `skos:prefLabel`, `dcterms:identifier`; its authors (`foaf:Person`) as text |
| `ipbes:SubChapter` | `sch` | `skos:prefLabel`, `dcterms:identifier`, `hasDescription` |
| `ipbes:Illustration` | `il` | `dcterms:identifier`, `skos:prefLabel` (caption) and `hasDescription` |
| `foaf:Person` | `person` | `skos:prefLabel` (or `foaf:firstName` + `foaf:lastName`), `ipbes:country`, roles (`ca`, `cl`, `cs`, `fl`, `la`, `re`) |
| `ipbes:Reference` | `ref` | usually skipped (`--skip-types ref`) |

The confidence level comes from `hasWellestablished`, `hasEstablishedIncomplete`, `hasUnresolved` or `hasInconclusive`. Items with no text are skipped, except experts, who are kept with "not recorded" when they have no country or role.

| Task | Command |
|---|---|
| Index (incremental: only new or changed chunks are embedded) | `PYTHONPATH=src python src/ttl_index.py --skip-types ref` |
| Check only: parse and validate, nothing written | `PYTHONPATH=src python src/ttl_index.py --dry-run --skip-types ref` |
| Wipe and start over | `PYTHONPATH=src python src/ttl_index.py --rebuild --skip-types ref` |
| Inspect retrieval | `PYTHONPATH=src python src/ttl_query.py` |
| Ask a question | `PYTHONPATH=src python src/ttl_rag.py` |
| Sanity checks | `PYTHONPATH=src python src/ttl_test.py` |

`--skip-types ref` leaves out the bibliographic references (about 33,000 chunks), which are rarely useful for questions. The current index is built without them.

#### Checking TTL files

Run the audit on new or updated TTL files before indexing them. It compares each file with the ontology in `docs/IPBES_Ontology_v06.ttl`, and runs the loader to find what the indexer would leave out. It changes nothing.

```bash
PYTHONPATH=src python src/ttl_audit.py                        # every file in data/ttl/
PYTHONPATH=src python src/ttl_audit.py data/ttl/SUA_v02.ttl   # some files
PYTHONPATH=src python src/ttl_audit.py --all                  # also properties partial by nature
```

For each class and file it reports:

| Line | Meaning | Example |
|---|---|---|
| `below 100%` | Share of the items that have a property the ontology lists. Confidence levels, expert roles, DOIs and optional links are partial by nature and only shown with `--all`. | `foaf:Person: 90` / `skos:prefLabel 0%`: SUA's experts have no name label |
| `not in ontology` | Properties the file uses that the ontology does not list | `ipbes:hasCategory` on GA1 and BBA knowledge gaps |
| `not indexed` | Items the loader leaves out, usually because they have no text or name | `LDR18-boxSPM1`: an illustration with neither caption nor description |
| `garbled text` | Values whose encoding was garbled before the TTL was written (`Earthâ€™s`, `2Â°C`). The loader repairs them (`repair_text`, using ftfy), but the conversion should be fixed. | IAS, LDR and VA |

Fix what matters in the source TTL (or its conversion), or in `ttl_loader.py` when the data is valid but read the wrong way. Then check the countries with `ttl_index.py --dry-run` and index.

#### Country values

Expert countries are matched to ISO codes with the geography RDF. Several countries in one value are split on `;`, `/` or `,` ("Argentina, Spain"); a person can also have several country values (nationality and residence), and all are kept. Colloquial names, nations of the United Kingdom and known misspellings in the source files ("UK", "Scotland", "Italia", "Australian", "Tunesia" …) are listed in `_COUNTRY_ALIASES` in `src/geo.py`.

Indexing stops when a country value cannot be resolved (`STRICT_COUNTRY_RESOLUTION` in `ttl_index.py`) and lists the values. Fix them in the source TTL, or add an alias to `geo.py`, and run the check again (`--dry-run`) until it passes.

#### Thesaurus

Re-index the thesaurus whenever the IPBES thesaurus RDF changes:

```bash
PYTHONPATH=src python src/rdf_index.py
```

It reads `data/rdf/ipbes-thesaurus.rdf`; another file can be given as an argument. To try the thesaurus RAG on its own:

```bash
PYTHONPATH=src python -c "
from rdf_rag import ThesaurusRAG
rag = ThesaurusRAG()
print(rag.answer('How does ecological integrity relate to ecosystem health?'))
"
```

### XML / Akoma Ntoso (Plenary reports)

Place XML files in `data/xml/`. The session number is read from each file's metadata (`FRBRnumber`).

The parser preserves structural information: document title and date, the decision or annex a passage is part of, division, subdivision, paragraph number, eId, XPath, table title and table rows. This matters because XML tables often carry meaning in the relationship between a table title, headers, year, country and row value.

It produces three chunk types:

- **`paragraph`**: the text of a `<paragraph>`, including its `<intro>` and `<subparagraph>` text. Footnotes (`<authorialNote>`) are left out, but the text that follows them is kept.
- **`standalone_p`**: a `<p>` that is not part of a paragraph or a table, e.g. directly under `<mainBody>`.
- **`table_row`**: one chunk per table row, repeating the table title, description and header in every row. Leading rows of `<th>` cells are the header, and each value is labelled with its column, e.g. `Country: Germany | 2012: 400,000`. Merged cells (`colspan`/`rowspan`) are expanded so values line up with their columns. If a table has no `<th>` cells, the first row is shown as a probable header, but values are not labelled.

| Task | Command |
|---|---|
| Index (incremental) | `PYTHONPATH=src python src/xml_index.py` |
| Wipe and start over (after changing `xml_loader.py`) | `PYTHONPATH=src python src/xml_index.py --rebuild` |
| Parser regression tests (no files in `data/` needed) | `PYTHONPATH=src python -m pytest src/test_xml_loader.py` |
| Inspect the chunks of your files | `PYTHONPATH=src python src/xml_test.py` |
| Inspect retrieval | `PYTHONPATH=src python src/xml_query.py` |
| Ask a question | `PYTHONPATH=src python src/xml_rag.py` |

### PDF

Place PDF files in `data/pdf/`, named `<assessment>_report.pdf` and `<assessment>_spm.pdf` (e.g. `bba_report.pdf`): the assessment is the part of the name before the first `_`, and must match the TTL code (BBA, GA1, IAS …) so that questions naming an assessment find its PDFs. The indexer extracts the text with PyMuPDF, cleans it, creates chunks, embeds them with the configured embedding model and stores them in Chroma (`pdf_loader.py`):

- **Running headers and footers** are removed: lines that recur at the top or bottom of many pages ("THE GLOBAL ASSESSMENT REPORT ON BIODIVERSITY …", chapter titles), and bare page numbers.
- **Words split at a line end** are joined ("biodi-versity"), unless the document uses the hyphenated form elsewhere or never the joined one, so compounds such as "decision-making" keep their hyphen.
- **Chunks are whole sentences**, up to 2,000 characters with about 300 characters of overlap, and stay within one page, so a chunk's page is exact.
- **Printed page numbers** are stored next to the PDF page: reports have front matter, so the printed page differs (GA1 report: printed page = PDF page − 63). Answers and the source list cite both, e.g. "PDF page 1089 (printed p. 1026)": the PDF page opens the file at the right place, the printed page is the one to quote.
- **File fingerprint:** each chunk and the manifest (`chroma/pdf_index.json`) record the PDF's SHA-256, showing which version of a file an answer came from.

The cleaning follows `pdf_index_diana.py`, a sentence index of a PDF used for checking quotes.

| Task | Command |
|---|---|
| Index (incremental) | `PYTHONPATH=src python src/pdf_index.py` |
| Wipe and start over | `PYTHONPATH=src python src/pdf_index.py --rebuild` |
| Inspect retrieval | `PYTHONPATH=src python src/pdf_query.py` |
| Ask a question | `PYTHONPATH=src python src/pdf_rag.py` |

---

## Rebuilding the collections

The PDF, TTL and XML indexers are incremental: they embed only chunks whose text changed, remove chunks that no longer exist, and rewrite the metadata of chunks whose metadata alone changed (e.g. after updating `ipbes-geo.rdf` or the country handling) without re-embedding them. After changing chunking or parsing logic, a normal run is usually enough. Use `--rebuild` to wipe a collection and start over. The thesaurus indexer (`rdf_index.py`) always rebuilds its collection.

Do not confuse the cache with `chroma/`: the cache holds retrieval results and generated answers; `chroma/` holds the vectorised document data.

After re-indexing, rebuild and re-upload the data bundle for the deployed app (see [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)).

### Switching the embedding model

Questions are embedded with the current embedding model, so every collection must be indexed with that same model. After changing `AZURE_OPENAI_EMBED_DEPLOYMENT` or `AZURE_OPENAI_EMBED_DIMENSIONS`, or switching between Azure and Ollama, rebuild all four collections (back up `chroma/` first):

```bash
PYTHONPATH=src python src/xml_index.py --rebuild
PYTHONPATH=src python src/ttl_index.py --rebuild --skip-types ref
PYTHONPATH=src python src/rdf_index.py
PYTHONPATH=src python src/pdf_index.py --rebuild
```

With Azure this takes about 30 minutes.

---

## Answer caching

The PDF and TTL pipelines cache both retrieval results and generated answers (`src/cache.py`, built on `aiocache`); the XML and combined pipelines cache answers only.

Where the cache lives depends on `REDIS_URL`:

- **Not set (default):** an in-memory cache. It lasts only as long as the Python process, so for the command-line scripts it is empty on every run; it is mainly useful in the long-running app. You will see the warning `REDIS_URL not set; using in-memory cache (dev only)`.
- **Set:** Redis, shared between runs and processes (see [Using Redis](#using-redis)).

Cache keys are hashes of:

| Cached item | Key includes | Expires after |
|---|---|---|
| Retrieval results | cache version, pipeline/collection, `RETRIEVAL_VERSION`, embedding model, time the index was last written, question, `k`, TTL filters (chunk type, countries, per-assessment), PDF re-ranking settings | 1 hour |
| Answers | cache version, `PROMPT_VERSION`, LLM model, `num_ctx`, question, TTL filters, a hash of the retrieved context (and thesaurus glossary) | 6 hours |

Re-indexing therefore invalidates cached retrievals automatically, and an answer is only reused when the same question was answered from exactly the same context. Empty retrieval results are never cached. Asking the exact same question again avoids another LLM request:

```text
First request:   CACHE MISS   retrieval 0.2 s, generation 8.0 s
Second request:  CACHE HIT    0.003 s
```

If a RAG prompt changes, increment `PROMPT_VERSION` in the corresponding `*_rag.py`; if retrieval logic changes, increment `RETRIEVAL_VERSION`. Old entries are then no longer used.

### Clearing the cache

- **In-memory cache:** restart the process (e.g. stop and restart Streamlit).
- **Everything, without touching Redis:** set a new cache version, e.g. `export RAG_CACHE_VERSION=v2`.
- **Redis:** delete the project's keys (all prefixed `ipbes`):

  ```bash
  redis-cli --scan --pattern 'ipbes*' | xargs -r redis-cli del
  ```

  Add `-n <db>` to both `redis-cli` calls if `REDIS_URL` uses a database other than 0. Or, without `redis-cli`, using the same `REDIS_URL` as the app:

  ```bash
  PYTHONPATH=src python -c "import asyncio, cache; asyncio.run(cache.invalidate_all())"
  ```

None of these touch the Chroma vector database in `chroma/`.

### Using Redis

macOS:

```bash
brew install redis
brew services start redis
export REDIS_URL=redis://localhost:6379/0
```

Linux:

```bash
sudo apt install redis-server
sudo systemctl start redis-server
export REDIS_URL=redis://127.0.0.1:6379/0
```

On WSL without systemd, use `sudo service redis-server start` instead of `systemctl`.

`export` only lasts for the current terminal. To set `REDIS_URL` in every new terminal, add it to `~/.bashrc`:

```bash
echo 'export REDIS_URL=redis://localhost:6379/0' >> ~/.bashrc
```

---

## Deploying to Streamlit Community Cloud

The deployed app uses the same Azure OpenAI models as local development, so it needs no Ollama. The search data is not in git (`chroma.sqlite3` is over GitHub's 100 MB file limit); the app downloads it once at start-up from `DATA_BUNDLE_URL`.

1. **Build the data bundle** (Chroma database plus the thesaurus and geography RDF) after indexing:

   ```bash
   python src/data_bundle.py
   ```

2. **Upload** `dist/ipbes-data.zip` to a private Azure Blob container and create a read-only SAS URL for it.
3. **Create the app** on share.streamlit.io from this repository: branch `main`, main file `src/app.py`, Python 3.11.
4. **Set the Secrets:**

   ```toml
   AZURE_OPENAI_ENDPOINT = "https://<resource>.services.ai.azure.com/openai/v1/"
   AZURE_OPENAI_API_KEY = "<key>"
   AZURE_OPENAI_DEPLOYMENT = "gpt-5.4"
   AZURE_OPENAI_EMBED_DEPLOYMENT = "text-embedding-3-small"
   DATA_BUNDLE_URL = "<SAS URL of ipbes-data.zip>"
   FEEDBACK_CONTAINER_URL = "<SAS URL of the feedback container>"
   ```

   The data bundle must have been indexed with the same embedding model. (`LOCAL_EMBEDDINGS = "1"` instead of `AZURE_OPENAI_EMBED_DEPLOYMENT` embeds questions in the app with nomic-embed-text, for a bundle indexed with Ollama.)

5. **Limit sharing** to invited viewers: the app spends the Azure key's budget.

Pushing to `main` redeploys the code automatically. After re-indexing, rebuild and re-upload the bundle under the same name, then **reboot** the app: the download only happens when `chroma/` is missing.

### Answer ratings

Community Cloud keeps no files across restarts, so ratings are stored in Azure Blob Storage. Create a private container (e.g. `feedback`), generate a SAS URL for the container with **Read, Add, Create and Write** permissions, and set it as `FEEDBACK_CONTAINER_URL` (in the Secrets, and in `.env` for local runs). Every rating is appended to `ratings.csv` in that container; download it from the portal. Without the setting, ratings go to `feedback/ratings.csv`.

Columns: timestamp (UTC), rating, question, answer, source, route, model and number of sources. The file starts with a UTF-8 byte order mark so that Excel shows accented characters and quotes correctly.

---

## Evaluation

### Principle

PDF, XML and TTL are not necessarily identical representations of the same documents. Comparisons between them should be treated as **separate corpus/retrieval experiments**, unless the same underlying information is available in equivalent form.

When evaluating, record at least the question, corpus/format, retrieved sources, retrieval time, generation time, cache hit or miss, and the answer. For more rigorous evaluation, also record whether the answer is correct, partially correct, unsupported, or not answerable from the supplied context.

The goal is not only to produce an answer, but to understand how representation and retrieval strategy affect the answer and its provenance.

### Comparing PDF and TTL

`src/pdf_ttl_compare.py` runs its question list through both pipelines and writes one row per question to `tests/comparison_<run_id>.csv`, as each question finishes:

```bash
PYTHONPATH=src python src/pdf_ttl_compare.py
```

With Redis (`REDIS_URL`), answers are cached across runs, so a rerun skips questions already answered with the same model and context.

### Sample questions

These are useful for testing retrieval and source attribution.

**Plenary reports (XML)**

- What are the main topics discussed in the report?
- What decisions were taken during the session?
- What was discussed regarding the work programme?
- What financial arrangements are described in the report?
- How many member states did the platform have as of 21 January 2013?
- What was the in-kind contribution for Germany in 2012?
- What is the status of in-kind contributions to IPBES?
- Who was elected chair at IPBES-1?
- What cash contribution did Germany make in 2012?
- What sections are included in the report?
- What was discussed under the institutional arrangements section?
- What are some of the main ways that stakeholders can engage with the work of IPBES? (Decision IPBES-3/4)

For table questions, check that the retrieved context includes the table title, table description, column headers, year, country/entity and value: a correct answer depends on preserving these relationships.

**Assessments (TTL and PDF)**

- What are the key messages identified in the IPBES LDR assessment?
- What are the background messages identified in the IPBES LDR assessment?
- What are the chapters of the IPBES LDR assessment?
- Which experts come from Kenya?
- Which experts come from East Africa?
- What findings across the different IPBES assessments refer to cities?
- Provide a summary of the topics covered across these different assessments.
- Provide a summary of the topics covered across these different assessments, and add the sections where these topics are covered.
- I am preparing a report on the major findings IPBES has had that are relevant to marine systems. Prepare a summary of 10 key points traceable across IPBES assessments that are relevant to marine systems.
- Who were the Review Editors of Ch. 4 of the values assessment?
- How have previous IPBES assessments approached the IPBES Conceptual Framework? Do you have figures referring to it?
- Provide a summary of the key findings in IPBES assessments that directly refer to the KMGBF.
- Provide the same output in a table by KMGBF goal and target vs. assessment, including the sections the information is taken from.

**Ambiguous questions**

- How much has Germany contributed to IPBES?
- What are the main findings and contributions of IPBES assessments?

The source material can contain more than one type of contribution. A useful response should explain the ambiguity and give the values or categories supported by the retrieved sources, rather than silently selecting one.

---

## Development workflow

```text
1. Make a change locally
2. Test with Python
3. Test sample questions
4. Check retrieved sources
5. Commit and push to GitHub (main redeploys the Streamlit app)
6. Re-index if needed; rebuild and upload the data bundle; reboot the app
7. Ask colleagues to test
8. Collect feedback (ratings.csv)
9. Iterate
```

The application is being prepared for a Docker-based deployment, so that colleagues can test it without installing Python, Chroma or the document-processing dependencies, and so that the environment can be reproduced consistently. The longer-term deployment target is Azure.

```text
                 Browser
                    │
                    ▼
              ┌───────────┐
              │ Streamlit │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │    RAG    │
              │  modules  │
              └─────┬─────┘
                    │
                    ▼
              ┌──────────────┐
              │ Azure OpenAI │
              └──────┬───────┘
             ┌───────┴─────────────────┐
             ▼                         ▼
          gpt-5.4          text-embedding-3-small
```

Deployment path: local development → Docker → GitHub → Streamlit Community Cloud → colleague testing → feedback and iteration → Azure deployment.

---

## Troubleshooting

### Azure OpenAI errors

- **401 / authentication:** check `AZURE_OPENAI_API_KEY` in `.env` (or the app's Secrets).
- **404 / deployment not found:** `AZURE_OPENAI_DEPLOYMENT` or `AZURE_OPENAI_EMBED_DEPLOYMENT` must match the deployment names in the Foundry portal exactly.
- **429 / rate limit:** the indexers retry automatically; if it persists, raise the deployment's tokens-per-minute quota in Foundry.
- **Answers cite unrelated passages after a model change:** the collections were indexed with a different embedding model (see [Switching the embedding model](#switching-the-embedding-model)).

### Chroma collection not found

The index has not been built yet. Run the indexer for that pipeline (see [Pipelines](#pipelines)).

### The deployed app gives old answers

The app still has the previous data bundle. Check that the new `ipbes-data.zip` was uploaded under the same name, then reboot the app.

### Answers say the information cannot be determined

First inspect the retrieved sources: turn on developer mode in the app, or run `xml_query.py`, `ttl_query.py` or `pdf_query.py`. A good answer needs all three:

```text
Correct document retrieved
        +
Correct passage or table retrieved
        +
Sufficient surrounding context
        =
Good answer
```

If the correct table is retrieved but the answer is still wrong, check whether the table title, headers, year and row value were retrieved together.

### Ollama (only when using Ollama)

- **Not responding:** check that it runs with `ollama list`, then try `ollama run llama3.1:latest`. If that works, exit the model and retry.
- **Model not found:** `ollama pull llama3.1:latest` and `ollama pull nomic-embed-text`, then check with `ollama list`.

---

## Contributing and colleague testing

The best way to report issues is by using 👍/👎 in the app. You could also report issues by sending specific feedback. When doing this, please send a screenshot or report the following:

1. The exact question.
3. Which corpus was searched: the app's default (assessments and Plenary reports together), or PDF, XML or TTL in developer mode.
4. The answer returned.
5. Whether the answer appears correct.
6. Whether the cited source supports the answer.
7. Any unexpected retrieval results.
8. Approximate response time.
9. Any error message available

Example:

```text
Corpus:    XML
Question:  What was the in-kind contribution for Germany in 2012?
Answer:    [copy answer]
Expected:  USD 400,000
Source:    Table 2
Issue:     The system retrieved the correct table but did not identify the Germany row/value.
```

This information is particularly useful for improving chunking, retrieval, prompts and provenance.

---

## Current status

| Component | Status |
|---|---|
| Azure OpenAI answers (gpt-5.4) | Working |
| Azure OpenAI embeddings (text-embedding-3-small) | Working |
| Ollama (offline fallback) | Working |
| PDF extraction, embeddings and RAG | Working |
| XML parsing, embeddings and RAG | Working |
| XML table-aware retrieval | Working |
| XML sessions, neighbouring paragraphs and decision names | Working |
| TTL parsing, embeddings and RAG (9 assessments) | Working |
| Combined TTL + XML search for the app | Working |
| Answer cache | Working (in-memory, or Redis via `REDIS_URL`) |
| Streamlit app (IBOK Chat) - Chat over TTL + XML; single sources in developer mode | Working |
| Store ratings in Azure Blob Storage)  | Working |
| Test/compare results | In progress |
| Helper for countries (`ipbes-geo.rdf`, `geo.py`) | In progress |
| Helper for the thesaurus (`ipbes-thesaurus.rdf`, `thesaurus_helper.py`) | In progress |
| Colleague testing | In progress |
| Helper for deliverables | Next |
| Helper for meetings | Next |
| Helper for experts | Next |
| Docker deployment | Planned |
| Architecture diagrams | Later stage |
| Security testing | Later stage |
| OICT approval | Later stage |
| Azure deployment | Later stage |
| Entra ID| Later stage |
| CloudFlare | Later stage |

---

## License and data

The application is intended to work with publicly available IPBES material. The documents used by a particular deployment should be checked against their applicable source and licensing terms.

No confidential, personal or draft documents should be added to this project.

## Questions and feedback

For bugs, retrieval problems, incorrect answers or suggestions, provide the exact question and the corpus used, so that the issue can be reproduced.
