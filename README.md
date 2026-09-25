# IPBES Local Document RAG

A local Retrieval-Augmented Generation (RAG) application for exploring and comparing IPBES documents across three source formats:

- **PDF** — document text extracted with PyMuPDF, chunked, embedded, and stored in Chroma.
- **XML** — Akoma Ntoso / LegalDocML documents parsed into structured paragraphs and table rows, then embedded in Chroma.
- **TTL / RDF ** — RDF/Turtle knowledge represented as a graph and queried with RDFLib/SPARQL.

The application uses **Ollama** for local language-model generation and embeddings.

The current generation model is:

```text
llama3.1:latest
```

The current embedding model is:

```text
nomic-embed-text
```

The project is being developed as an evaluation platform for comparing retrieval and answer generation across PDF, XML, and RDF/TTL representations of IPBES knowledge. This will help build a chat function for IPBES data and guide allocation of resources.

The project currently has 3 pipelines that stores data separately in pdf_documents, ttl_documents and xml_documents. Once testing is finalized the pipelines will be merged.

---

## What the application does

The RAG pipeline follows this general pattern:

```text
Documents
   │
   ├── PDF ──► Text extraction ──► Chunks ──► Embeddings ──► Chroma
   │
   ├── XML ──► XML parser[Metadata, Paragraphs, Tables] ──► structured chunks ──► Embeddings ──► Chroma ──► Answer ──► Structured XML provenance
   │
   └── TTL ──► RDFLib / SPARQL [Subjects, Predicates Objects] ──► Graph retrieval  ──► LLM (Ollama)──► Answer ──► Sources

```

The system is designed to preserve source provenance. Answers can include information such as:

- source document
- PDF page
- XML section/subsection
- XML paragraph/eId
- XML XPath
- table information
- retrieval distance
- retrieval time
- generation time
- cache status

The application should answer only from retrieved source material. If the available context does not establish an answer, the RAG prompts instruct the model to say:

> I cannot determine that from the supplied documents.

---

## Current project structure

The project is organized approximately as follows:

```text
ipbesllm/
│
├── .venv/
│   └── local Python virtual environment
│
├── chroma/
│   └── Chroma persistent vector database
│
├── data/
│   ├── pdf/
│   │   └── *.pdf
│   ├── rdf/
│   │   └── *.rdf
│   ├── ttl/
│   │   └── *.ttl
│   └── xml/
│       └── *.xml
│
├── src/
│   ├── __pycache__/
│   │   └── *.pyc
│   ├── cache.py
│   ├── geo.py
│   ├── models.py
│   ├── pdf_index.py
│   ├── pdf_loader.py
│   ├── pdf_query.py
│   ├── pdf_rag.py
│   ├── pdf_test_embeddings.py
│   ├── pdf_test.py
│   ├── stage1_ollama.py
│   ├── ttl_index.py
│   ├── ttl_loader.py
│   ├── ttl_query.py
│   ├── ttl_rag.py
│   ├── ttl_test.py
│   ├── xml_index.py
│   ├── xml_loader.py
│   ├── xml_query.py
│   ├── xml_rag.py
│   ├── xml_test_chunks.py
│   └── xml_test.py
├── .webui_secret_key
├── app.py
├── README.md
└── requirements.txt
```

Some files may be added or renamed as the project develops. The important separation is:

```text
data/      source documents
src/       application and RAG code
chroma/    vector database
cache      cached answers
```

---

# Quick start

## 1. Requirements

You need:

- macOS, Linux, or Windows
- Python 3.11
- Git
- Ollama
- the required Python packages from `requirements.txt`

Docker support is planned for deployment and colleague testing. The local Python/Ollama setup described below is useful for development and troubleshooting.

---

## 2. Install Ollama

Install Ollama from:

https://ollama.com/

Verify that it is available:

```bash
ollama --version
```

Pull the required models llama3.1:latest(generation/chat model) and nomic-embed-text (embedding model):

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```

Check the available models:

```bash
ollama list
```

You should see the generation and embedding models.

---

## 3. Clone the repository

```bash
git clone git@github.com:ipbes/ipbesllm.git
cd ipbesllm
```

---

## 4. Create a Python virtual environment

Python 3.11 is recommended.

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 5. Install Python dependencies

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```
To create or take a snapshot of project dependencies 
```bash
python -m pip freeze > requirements.txt
```

Additional packages may be required as features, the Streamlit interface and Docker deployment are added.

---

## 5. Open WebUI
Open WebUI already exposes Ollama through its API, so we can integrate our finished pipeline with it later rather than allowing the UI's retrieval implementation to obscure the experiment.
```bash
python3.11 -m venv .venv
source .venv/bin/activate
open-webui serve
```


---

# Configure Ollama

The application uses environment variables so that the model can be changed without modifying the Python source code.

macOS/Linux:

```bash
export OLLAMA_MODEL="llama3.1:latest"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
```

Windows PowerShell:

```powershell
$env:OLLAMA_MODEL="llama3.1:latest"
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
```

The defaults in the code are already configured for these models, so this step is mainly useful when testing a different model.

---

# PDF RAG

## Index the PDFs

Place PDF files in:

```text
data/pdf/
```

Then build/rebuild the PDF Chroma collection:

```bash
PYTHONPATH=src python src/index_pdf.py
```

The indexer extracts text from the PDFs, creates chunks, generates embeddings using `nomic-embed-text`, and stores them in Chroma.

## Test PDF retrieval

```bash
PYTHONPATH=src python src/query_pdf.py
```

## Ask a PDF RAG question

```bash
PYTHONPATH=src python src/pdf_rag.py
```

---

# XML / Akoma Ntoso RAG

The XML pipeline is designed specifically for Akoma Ntoso / LegalDocML documents.

The parser preserves structural information such as:

- document title
- document date
- division
- subdivision
- paragraph number
- eId
- XPath
- table title
- table rows

This is important because XML tables often contain meaning in the relationship between a table title, headers, year, country, and row value.

## Test the XML parser

```bash
PYTHONPATH=src python src/xml_test.py
```

## Index XML documents

Place XML files in:

```text
data/xml/
```

Then:

```bash
PYTHONPATH=src python src/xml_index.py
```

The XML collection is:

```text
xml_documents
```

## Query XML retrieval

```bash
PYTHONPATH=src python src/xml_query.py
```

## Ask an XML RAG question

```bash
PYTHONPATH=src python src/xml_rag.py
```

---

# TTL / RDF

The TTL/RDF pipeline uses RDFLib and SPARQL-style graph retrieval rather than treating the data as ordinary document text.

Place Turtle files in:

```text
data/ttl/
```

## Test the TTL parser

```bash
PYTHONPATH=src python src/ttll_test.py
```

## Index TTL documents

Place TTL files in:

```text
data/ttl/
```

Then:

```bash
PYTHONPATH=src python src/ttl_index.py
```

The TTL collection is:

```text
ttl_documents
```

## Query TTL retrieval

```bash
PYTHONPATH=src python src/ttl_query.py
```

## Ask a TTL question

```bash
PYTHONPATH=src python src/ttl_rag.py
```


---

# Sample questions

The following questions are useful for testing retrieval and source attribution.

## PDF vs XML questions

### Specific questions

```text
What are the main topics discussed in the report?
```

```text
What decisions were taken during the session?
```

```text
What was discussed regarding the work programme?
```

```text
What financial arrangements are described in the report?
```

```text
How many member states did the platform have as of 21 January 2013?
```

```text
What was the in-kind contribution for Germany in 2012?
```

```text
What is the status of in-kind contributions to IPBES?
```

```text
Who was selected as chair by the Plenary?
```

```text
What cash contribution did Germany make in 2012?
```

```text
What sections are included in the report?
```

```textWhat was discussed under the institutional arrangements section?

```

For table questions, pay particular attention to whether the retrieved context includes:

```text
table title
table description
column headers
year
country/entity
value
```

A correct answer depends on preserving these relationships.
### Ambiguous questions

```text
How much has Germany contributed to IPBES?
```


## PDF vs TTL Questions
### Specific questions
```text
What are the key messages identified in the IPBES LDR assessment?
```

```text
What are the background messages identified in the IPBES LDR assessment?
```

```text
What are the chapter of the IPBES LDR assessment?
```

```text
Which experts come from Kenya?
```

```text
Which experts come from East Africa?
```

```text
What findings across the different IPBES assessments refer to cities?
```

```text
Provide a summary of the topics covered across these different assessments?
```
```text
Provide a summary of the topics covered across these different assessments and add the sections in the assessments where these topics are covered
```

```text
I am preparing a report on the major findings IPBES has had that are relevant to marine systems. Prepare a summary of 10 key points traceable across IPBES assessments that are relevant to marine systems
```

```text
Who were the Review Editors of Ch. 4 of the values assessment
```

```text
How have previous IPBES assessments approached the IPBES Conceptual Framework? Do you have figures referring to it?
```

```text
Provide a summary of the key findings in IPBES assessments that directly refer to the KMGBF
```

```text
Provide the same output in a table by KMGBF goal and target vs. assessment, and including also the reference to the sections where the information is being obtained from
```

## Ambiguous questions

The source material can contain more than one type of contribution. For example:

```text
What are the main findings and contributions of IPBES assessments?
```

This may require the application to distinguish between different contribution categories rather than silently selecting one.

A useful RAG response should explain the ambiguity and provide the relevant values/categories supported by the retrieved source.

---

# Answer caching

The application includes a lightweight SQLite answer cache:

```text
rag_cache.sqlite3
```

The cache is keyed using:

- corpus
- question
- model
- prompt version

This means that asking the exact same question again can avoid another LLM generation request.

For example:

```text
First request:
CACHE MISS
Retrieval time:  0.2s
Generation time: 8.0s

Second request:
CACHE HIT
Time: 0.003s
```

If the RAG prompt changes, increment the corresponding `PROMPT_VERSION` so that old answers are not reused.

To clear the answer cache:

```bash
rm rag_cache.sqlite3
```

This does **not** delete the Chroma vector database.

---

# Rebuilding a vector collection

When changing the chunking or parsing logic, rebuild the corresponding Chroma collection.

For example, after changing XML table parsing:

```bash
PYTHONPATH=src python src/index_xml.py
```

The indexer should recreate the relevant collection when required.

For PDF changes:

```bash
PYTHONPATH=src python src/index_pdf.py
```

Do not confuse:

```text
rag_cache.sqlite3
```

with:

```text
chroma/
```

The first contains cached answers. The second contains vectorized document data.

---

# Development workflow

The intended development workflow is:

```text
1. Make a change locally
       ↓
2. Test with Python
       ↓
3. Test sample questions
       ↓
4. Check retrieved sources
       ↓
5. Commit to Git
       ↓
6. Push to GitHub
       ↓
7. Test Streamlit deployment
       ↓
8. Ask colleagues to test
       ↓
9. Collect feedback
       ↓
10. Iterate
```

The application is being prepared for a Docker + Streamlit deployment so that colleagues can test the application without installing Python, Ollama, Chroma, or the document-processing dependencies locally.

The longer-term deployment target is Azure.

---

# Important evaluation principle

PDF, XML, and TTL are not necessarily identical representations of the same documents.

Therefore, comparisons between them should be treated as **separate corpus/retrieval experiments** unless the same underlying information is available in equivalent form.

When evaluating the systems, record at least:

```text
Question
Corpus / format
Retrieved sources
Retrieval time
Generation time
Cache hit/miss
Answer
```

For more rigorous evaluation, also record whether the answer is:

```text
Correct
Partially correct
Unsupported
Not answerable from supplied context
```

The goal is not only to produce an answer, but to understand how representation and retrieval strategy affect the answer and its provenance.

---

# Troubleshooting

## Ollama is not responding

Check:

```bash
ollama list
```

Then verify that Ollama is running.

Try:

```bash
ollama run llama3.1:latest
```

If that works, exit the model and retry the application.

---

## Model not found

Pull the models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Then check:

```bash
ollama list
```

---

## Chroma collection not found

You probably need to build the index first.

For PDF:

```bash
PYTHONPATH=src python src/pdf_index.py
```

For XML:

```bash
PYTHONPATH=src python src/xml_index.py
```

For TTL:

```bash
PYTHONPATH=src python src/ttl_index.py
```
---

## Answers say that information cannot be determined

First inspect the retrieved sources:

```bash
PYTHONPATH=src python src/xml_query.py
```

or:

```bash
PYTHONPATH=src python src/pdf_query.py
```

The important distinction is:

```text
Correct document retrieved
        +
Correct passage/table retrieved
        +
Sufficient surrounding context
        =
Good answer
```

If the correct table is retrieved but the answer is still incorrect, inspect whether the table title, headers, year, and row value were retrieved together.

---

# Docker and Streamlit

The project is being prepared for a Docker-based deployment with Streamlit.

The intended architecture is:

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
              ┌───────────┐
              │  Ollama   │
              └───────────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
          Llama 3.1    Embeddings
```

During development, Docker allows the local environment to be reproduced consistently.

The planned agile deployment path is:

```text
Local development
       ↓
Docker
       ↓
GitHub
       ↓
Streamlit Community Cloud
       ↓
Colleague testing
       ↓
Feedback / iteration
       ↓
Azure deployment
```

---

# Contributing / colleague testing

When testing a new version, please report:

1. The exact question.
2. Which corpus was selected: PDF, XML, or TTL.
3. The answer returned.
4. Whether the answer appears correct.
5. Whether the cited source supports the answer.
6. Any unexpected retrieval results.
7. Approximate response time.

Example:

```text
Corpus: XML

Question:
What was the in-kind contribution for Germany in 2012?

Answer:
[copy answer]

Expected:
USD 400,000

Source:
Table 2

Issue:
The system retrieved the correct table but did not identify the Germany row/value.
```

This information is particularly useful for improving chunking, retrieval, prompts, and provenance.

---

# Current status

| Component | Status |
|---|---|
| Ollama local LLM | Working |
| PDF extraction | Working |
| PDF embeddings / Chroma | Working |
| PDF RAG | Working |
| XML parsing | Working |
| XML embeddings / Chroma | Working |
| XML RAG | Working |
| XML table-aware retrieval | Being improved |
| TTL parsing | Working |
| TTL embeddings | Working |
| TTL RAG | Working |
| Answer cache | Planned/ in development |
| Streamlit UI | Planned / in development |
| Docker deployment | Planned / in development |
| Colleague testing | Next deployment stage |
| Architecture diagrams | Later stage |
| OICT approval | Later stage |
| Azure deployment | Later stage |

---

# License and data

The application is intended to work with publicly available IPBES material.

The documents used by a particular deployment should be checked against their applicable source and licensing terms.

No confidential, personal or draft documents should be added to this project.

---

## Questions and feedback

For bugs, retrieval problems, incorrect answers, or suggestions, provide the exact question and corpus used so that the issue can be reproduced.
