# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
Campus Survival Guide: This will be a compilation of all the ins and niche secrets of UT Dallas that only current and former students can know. This knowledge is important to answering the ultra specific and sometimes embarrassing questions that incoming freshmen or transfer students might be left wondering about. Particularly, this will focus on general questions applying to all students and pre-health questions applying to students related to in pre-health tracks or in majors related to those tracks in terms of academics.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

**Official UTD documentation** — `documents/Official website documentation/` (3 PDFs)

| # | Source | Type | Location |
|---|--------|------|----------|
| 1 | Pre-Health Starter Kit 2024 | PDF | `UTD-Pre-Health-Starter-Kit-2024-DB.pdf` |
| 2 | HPAC pages (merged): starter kit, application support, contact, research, student orgs, summer programs, success rates, JAMP, other opportunities | PDF | `Pre-health_merged.pdf` |
| 3 | Natural Sciences & Mathematics (NSM) pages | PDF | `NSM_pages.pdf` |

**Reddit r/utdallas + r/TransferToTop25** — `documents/reddit/` (mixed)

| # | Source | Type | Location |
|---|--------|------|----------|
| 4 | Housing Megathread (Fall Semester) | PDF | `Housing Megathread - Fall Semester _ r_utdallas.pdf` |
| 5 | Academic tips for UTD freshmen (curated) | TXT | `Academic Tips for UTD Freshmen.txt` |
| 6 | r/utdallas comment threads (16 files: base + (1)–(15)) | Markdown | `post_comments-utdallas-2026-06-09*.md` |
| 7 | Transfer-to-Top-25 discussion | Markdown | `post_comments-TransferToTop25-2026-06-09.md` |
| 8 | Subreddit posts metadata (id, title, author, score, num_comments, created_utc, selftext, …) | CSV | `subreddit_posts-utdallas-2026-06-09.csv` |
| 9 | r/utdallas comments, structured (body, author, score, parent_id, depth, …) — 9 files | CSV | `post_comments-utdallas-2026-06-09*.csv` |

**Rate My Professors** — `documents/RMP/` (1 TXT)

| # | Source | Type | Location |
|---|--------|------|----------|
| 10 | Rate My Professors reviews, combined | TXT | `RMP Reviews combined.txt` |

> Reddit was collected via `documents/reddit/redditjsonretrieval.py` (public `.json` endpoint, rate-limited at 100). PRAW is listed in `requirements.txt` to move to the authenticated Reddit API if higher volume is needed (see Milestone 4 retrieval step).

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 400–500 tokens

**Overlap:** 50–75 tokens

**Reasoning:** While most of the Reddit posts/comments are short and self-contained (ideally 100–300 tokens), the UTD pages are longer and more structured, so 400–500 tokens captures context more appropriately while keeping Reddit posts mostly intact. Overlap handles the case where a key fact, like a professor's name and their rating, sits at a chunk boundary.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** multi-qa-mpnet-base-dot-v1 via sentence-transformers (768-dim, 512-token window, dot-product similarity; tuned on 215M question→answer pairs for asymmetric semantic search). Chosen over all-MiniLM-L6-v2 because MiniLM's 256-token window would silently truncate our 400–500 token chunks, and this model is purpose-built for the question→passage retrieval this guide does.

**Top-k:** 5

**Production tradeoff reflection:** multi-qa-mpnet is a strong, portable local QA-retrieval model, but it still won't understand UTD-specific jargon (e.g. ECS, JSOM, Comet Card) as well as a domain-finetuned model. So in production I'd weigh something like OpenAI's `text-embedding-3-large` for accuracy (at a cost tradeoff), or `instructor-xl` for instruction-tuned retrieval quality. If this were a live chatbot, latency is another factor. But for a student guide with <1000 chunks, this model works for now.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about wait times at Dining Hall West during lunch? | Long waits at rush hour (e.g. 12–1pm, 6–7pm); students recommend going before noon or after 1:30. |
| 2 | Which professor should I take for BIO 2311 at UTD? | Amy Jo (MacBey) Gomez — cited for positive reviews, clear lectures, and/or fair exams/grading. |
| 3 | How do UTD pre-med students typically get clinical hours in Dallas? | UT Southwestern volunteering, Parkland Hospital volunteering, UTD Health Center, and other entry-level work in a local clinical environment. |
| 4 | What are the unwritten rules for finding parking on campus without a permit? | Lot-specific tips (free after 7pm, visitor lots, etc.) from student posts. |
| 5 | What GPA do students say you need to be taken seriously by UTD pre-health advising? | ~3.5+ mentioned in Reddit threads; HPAC office expectations. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Reddit is heavily noisy and notoriously subjective. I'm afraid the manual extraction of data makes the bias towards most discussed topics rather than strictly academic insights. Chunks from these sources may be outdated, irrelevant or just simply inaccurate due to inconsistent context from the chunks retrieved. Without timestamp metadata, the pipeline can't flag staleness of data.

2. Chunk-boundary splits on structured UTD pages, which often pack bulkier text into compressed forms larger than the designated chunk size. If a chunk cuts mid-table or mid-form, we'd lose row context e.g. "CHEM 1311" separated from its "prereq: none" cell. Consider splitting on document structure rather than raw token count for those docs.

3. PDF parsing might struggle on content cut in half between pages on official documentation. Attempt to read it altogether and then chunk based on designated sizing from the total text parsed together.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
![Pipeline: Document Ingestion (pathlib + pdfplumber/pandas/plaintext) → Chunking (custom chunk_text, 500 tok / 75 overlap) → Embedding (multi-qa-mpnet-base-dot-v1, sentence-transformers) → Vector Store (ChromaDB local persistent, dot-product) → Retrieval (ChromaDB similarity search, top-k=5) → Generation (Groq llama-3.3-70b-versatile)](architecture.png)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:** Claude Cowork + Code to research appropriate parameters before implementing them through the Code agent. Input is the Chunking Strategy and Documents sections. Expected output is `ingest.py` with `load_documents()` and `chunk_text(text, chunk_size=500, overlap=75)`. Verify: print chunk count and the first 3 chunks for each document type, confirming no chunk is empty or >600 tokens.

**Milestone 4 — Embedding and retrieval:** Claude Code reads the Retrieval Approach section and the output schema from `ingest.py`. Expected output is `embed.py` (loads chunks, embeds with MiniLM, stores in ChromaDB) and `retrieve.py` with `get_relevant_chunks(query, k=5)`. Verify: run the 5 evaluation questions and manually check that returned chunks are topically relevant (not just keyword matches).

**Milestone 5 — Generation and interface:** Claude Code reads the full `planning.md`, the working `retrieve.py`, and the 5 Q&A pairs. Expected output is `generate.py` that takes a query → retrieves chunks → builds a prompt → calls the LLM → returns a cited answer, plus a minimal Gradio interface. Verify: run all 5 eval questions end-to-end and check answers against the expected-results table, flagging any hallucinations or off-topic responses.