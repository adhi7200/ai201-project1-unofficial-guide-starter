"""Milestone 4 — Retrieval.

Embeds a query with the same model used for indexing and returns the top-k most
similar chunks from ChromaDB, with source attribution and dot-product scores.

    python retrieve.py            # test the 5 eval queries from planning.md
    python retrieve.py "your question here"   # ad-hoc query
"""

import sys

from embed_store import embed_texts, get_collection

# The 5 evaluation questions from planning.md (questions only — we judge
# retrieval relevance by inspection, not against the expected answers).
EVAL_QUERIES = [
    "What do students say about wait times at Dining Hall West during lunch?",
    "Which professor should I take for BIO 2311 at UTD?",
    "How do UTD pre-med students typically get clinical hours in Dallas?",
    "What are the unwritten rules for finding parking on campus without a permit?",
    "What GPA do students say you need to be taken seriously by UTD pre-health advising?",
]


def retrieve(query, k=5):
    """Return the top-k chunks for a query as a list of dicts:
    {"text", "source", "chunk_index", "score", "distance"}.
    score = dot-product similarity (higher = more relevant)."""
    collection = get_collection()
    query_emb = embed_texts([query], show_progress_bar=False)[0].tolist()
    res = collection.query(query_embeddings=[query_emb], n_results=k)

    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "score": 1.0 - dist,  # ip: distance = 1 - dot  ->  score = dot
        })
    return results


def _safe(s):
    return s.encode("ascii", "replace").decode()


def _print_results(query, results, preview=320):
    print("\n" + "=" * 74)
    print(f"QUERY: {query}")
    print("=" * 74)
    for rank, r in enumerate(results, 1):
        print(f"\n  #{rank}  score={r['score']:.2f}  "
              f"[{r['source']} | chunk {r['chunk_index']}]")
        print("      " + _safe(r["text"][:preview]).replace("\n", " "))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        _print_results(q, retrieve(q))
    else:
        if get_collection().count() == 0:
            print("Collection is empty — run `python build_index.py` first.")
            sys.exit(1)
        for q in EVAL_QUERIES:
            _print_results(q, retrieve(q))
