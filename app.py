"""Milestone 5 — Gradio query interface for The Unofficial Guide.

A viewer enters a question; the app retrieves grounded context, generates an
answer, and shows which source documents the answer drew from.

    python app.py      # opens http://localhost:7860
"""

import gradio as gr

from query import ask


def handle_query(question):
    question = (question or "").strip()
    if not question:
        return "Type a question above and press Ask.", ""
    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(none — the system declined to answer from the corpus)"
    return result["answer"], sources


# The four eval questions that return grounded, cited answers (Q1–Q4).
# Q5 (GPA) is intentionally omitted — it triggers a refusal and is documented as a
# failure case; type it live in the demo to show the refusal behavior.
EXAMPLES = [
    "What majors are recommended for premed?",
    "Which professor should I take for BIO 2312 at UTD?",
    "How do UTD pre-med students typically get clinical hours in Dallas?",
    "What are the unwritten rules for finding parking on campus without a permit?",
]

with gr.Blocks(title="The Unofficial Guide — UT Dallas") as demo:
    gr.Markdown(
        "# The Unofficial Guide — UT Dallas\n"
        "Ask about UTD professors, pre-health, transfers, parking, housing, and "
        "campus life. Answers are grounded in real student posts, Rate My "
        "Professors reviews, and HPAC advising documents — with sources shown."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. Which professor should I take for BIO 2311?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
