"""Milestone 5 — End-to-end query function.

ask(question) ties the whole pipeline together:
    retrieve top-k chunks  ->  generate a grounded answer  ->  attach sources.

Source attribution is computed PROGRAMMATICALLY from the retrieved chunks (the
unique source documents that were fed to the model), so attribution is guaranteed
by the pipeline rather than left to the LLM. On a refusal, no sources are shown
(nothing in the corpus supported an answer).

    python query.py                 # interactive REPL
    python query.py "your question" # single question
"""

import sys

from generate import REFUSAL, generate_answer
from retrieve import retrieve


def ask(question, k=5):
    """Run the full RAG pipeline.

    Returns {"answer": str, "sources": list[str], "chunks": list[dict]}.
    """
    chunks = retrieve(question, k=k)
    answer = generate_answer(question, chunks)

    if answer.strip() == REFUSAL:
        sources = []  # nothing supported an answer
    else:
        # Unique retrieved source documents, in retrieval order.
        seen, sources = set(), []
        for c in chunks:
            src = c["source"]
            if src not in seen:
                seen.add(src)
                sources.append(src)

    return {"answer": answer, "sources": sources, "chunks": chunks}


def _print(result):
    print("\nAnswer:\n" + result["answer"])
    if result["sources"]:
        print("\nRetrieved from:")
        for s in result["sources"]:
            print(f"  • {s}")
    else:
        print("\n(no sources — the system declined to answer)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print(ask(" ".join(sys.argv[1:])))
    else:
        print("Ask The Unofficial Guide (blank line or Ctrl-C to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            _print(ask(q))
