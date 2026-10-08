# src/rdf_index.py
"""
Index the thesaurus concept profiles into Chroma.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.rdf_graph import ThesaurusGraph
from src.rdf_loader import build_all_documents, ConceptDocument

logger = logging.getLogger(__name__)

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ipbes_thesaurus"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_embedding_function():
    """
    Reuse whatever embedding function your other indexes use.
    If your other modules use a local sentence-transformer or Ollama,
    swap this to match.
    """
    from chromadb.utils import embedding_functions
    # Example: local sentence-transformer — adjust to taste
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_index(rdf_path: str, reset: bool = True) -> chromadb.Collection:
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection %s", COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    graph = ThesaurusGraph(rdf_path)
    docs: list[ConceptDocument] = build_all_documents(graph)

    # Batch insert
    ids = [d.concept_id for d in docs]
    texts = [d.text for d in docs]
    metadatas = [d.metadata for d in docs]

    BATCH = 100
    for i in range(0, len(docs), BATCH):
        collection.add(
            ids=ids[i : i + BATCH],
            documents=texts[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )

    logger.info("Indexed %d concepts into %s", len(docs), COLLECTION_NAME)
    return collection


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    rdf_file = sys.argv[1] if len(sys.argv) > 1 else "data/rdf/ipbes-thesaurus.rdf"
    build_index(rdf_file)