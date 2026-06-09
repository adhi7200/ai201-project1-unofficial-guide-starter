"""Milestone 4 — Embedding + vector store.

Embeds chunks with multi-qa-mpnet-base-dot-v1 (per planning.md) and stores them
in a local, persistent ChromaDB collection with source + chunk-index metadata.

Why dot-product: multi-qa-mpnet-base-dot-v1 was trained with a dot-product
objective and produces UN-normalized embeddings (vector magnitude carries
signal). So the Chroma collection uses the inner-product space ("ip"). For "ip",
Chroma reports distance = 1 - dot(query, chunk), so we expose
    score = 1 - distance = dot(query, chunk)   (higher = more relevant).
"""

from functools import lru_cache
from pathlib import Path

import chromadb

from ingest import EMBEDDING_MODEL  # sentence-transformers/multi-qa-mpnet-base-dot-v1

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "unofficial_guide"
# Inner-product space — matches the model's dot-product training objective.
COLLECTION_METADATA = {"hnsw:space": "ip"}


@lru_cache(maxsize=1)
def get_embedder():
    """Load the SentenceTransformer once and reuse it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Return the existing collection (create empty if missing)."""
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )


def reset_collection():
    """Drop and recreate the collection so re-indexing never duplicates."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — fine if it doesn't exist yet
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )


def embed_texts(texts, batch_size=64, show_progress_bar=True):
    """Encode a list of strings into un-normalized dot-product embeddings."""
    embedder = get_embedder()
    return embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep magnitude — model is dot-product
    )


def index_chunks(chunks, add_batch=512):
    """Embed and store chunk records into a fresh collection.

    chunks: list of {"text", "source", "chunk_index"} (from ingest.chunk_records()).
    Returns the populated collection.
    """
    collection = reset_collection()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"Embedding {len(docs)} chunks with {EMBEDDING_MODEL} ...")
    embeddings = embed_texts(docs)

    print("Writing to ChromaDB ...")
    for start in range(0, len(ids), add_batch):
        end = start + add_batch
        collection.add(
            ids=ids[start:end],
            embeddings=[e.tolist() for e in embeddings[start:end]],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )
    return collection
