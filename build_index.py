"""Milestone 4 — Build the vector index.

Runs the full ingestion pipeline and loads every chunk into ChromaDB:
    documents/  ->  load_documents()  ->  chunk_text()  ->  embed  ->  ChromaDB

Run after adding/changing documents:
    python build_index.py
"""

from collections import Counter

from embed_store import COLLECTION_NAME, index_chunks
from ingest import chunk_records


def main():
    print("Loading + chunking documents...")
    chunks = chunk_records()
    print(f"  {len(chunks)} chunks ready.")

    by_source_type = Counter(
        next((e for e in (".pdf", ".csv", ".md", ".txt") if e in c["source"].lower()), "?")
        for c in chunks
    )
    print(f"  by type: {dict(by_source_type)}")

    collection = index_chunks(chunks)
    count = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' now holds {count} chunks.")
    if count != len(chunks):
        print(f"  [warn] expected {len(chunks)} but collection has {count}.")


if __name__ == "__main__":
    main()
