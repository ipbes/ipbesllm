# src/rdf_index.py
"""
Index the thesaurus concept profiles into Chroma.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.rag_utils import (
    EMBED_MODEL, embed_client, embed_texts, make_batches, make_embedding_function,
)
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
        embedding_function=make_embedding_function(EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )

    graph = ThesaurusGraph(rdf_path)
    docs: list[ConceptDocument] = build_all_documents(graph)

    # Embed with the same Ollama model as the other indexes (rag_utils) and
    # hand the vectors to Chroma.
    chunks = [{"chunk_id": d.concept_id, "text": d.text, "metadata": d.metadata}
              for d in docs]
    client = embed_client()
    for batch in make_batches(chunks):
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embed_texts(client, [c["text"] for c in batch], EMBED_MODEL),
        )

    logger.info("Indexed %d concepts into %s (%s)", len(docs), COLLECTION_NAME,
                EMBED_MODEL)
    return collection


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    rdf_file = sys.argv[1] if len(sys.argv) > 1 else "data/rdf/ipbes-thesaurus.rdf"
    build_index(rdf_file)