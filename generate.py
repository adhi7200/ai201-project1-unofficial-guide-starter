"""Milestone 5 — Grounded answer generation with Groq.

Takes a question and the retrieved chunks, and asks llama-3.3-70b-versatile to
answer using ONLY those chunks. Grounding is enforced in the system prompt (not
merely suggested): the model is told to use only the provided context and to
return a fixed refusal string when the context is insufficient.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # read GROQ_API_KEY from .env

GROQ_MODEL = "llama-3.3-70b-versatile"

# The exact string the model must return when the context cannot answer.
# query.py keys off this to suppress source attribution on refusals.
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are The Unofficial Guide, a question-answering assistant for \
UT Dallas students. You answer ONLY from the numbered context documents provided \
in each request. These documents are real student posts, reviews, and advising \
materials.

Strict rules:
1. Use ONLY information stated in the provided context documents. Never use your \
own general or prior knowledge, even if you are confident.
2. If the context does not contain enough information to answer the question, \
reply with EXACTLY this sentence and nothing else: "{REFUSAL}"
3. When you do answer, cite the document(s) you used inline, e.g. \
"(source: RMP Reviews combined.txt)". Only cite documents that actually support \
your answer.
4. Reflect what students/sources actually said — be concrete and specific. Do not \
invent names, numbers, professors, or policies that are not in the context.
5. Be concise: a short, direct answer grounded in the sources."""


@lru_cache(maxsize=1)
def _get_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — add it to .env")
    return Groq(api_key=api_key)


def format_context(chunks):
    """Render retrieved chunks as a numbered, source-labeled context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']}):\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(query, chunks, model=GROQ_MODEL):
    """Generate a grounded answer from the retrieved chunks.

    Returns the model's text. If there are no chunks, returns the refusal
    without calling the API.
    """
    if not chunks:
        return REFUSAL

    user_prompt = (
        f"Context documents:\n\n{format_context(chunks)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, following the rules."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic, reduces drift from the context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
