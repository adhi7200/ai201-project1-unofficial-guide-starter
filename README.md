# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language
questions about UT Dallas using real student-generated knowledge — Reddit
threads, Rate My Professors reviews, and official advising documents — and
returns grounded, source-cited answers.

**Pipeline:** `documents/` → ingest + clean ([ingest.py](ingest.py)) → token-based
chunking → embed with multi-qa-mpnet-base-dot-v1 ([embed_store.py](embed_store.py))
→ ChromaDB vector store ([build_index.py](build_index.py)) → top-k retrieval
([retrieve.py](retrieve.py)) → grounded generation with Groq (Milestone 5).

**Run:**
```bash
pip install -r requirements.txt
python build_index.py     # ingest, chunk, embed, store (≈90s on CPU)
python retrieve.py        # test retrieval on the 5 eval queries
python app.py             # Gradio query interface (Milestone 5)
```

---

## Domain

**Campus survival guide for UT Dallas — general student life + pre-health academics.**

The system makes searchable the niche, hard-won knowledge that current and former
UTD students share with each other but that official channels never publish:
which professor to take for a specific course, how pre-med students actually find
clinical hours in Dallas, the unwritten rules of campus parking, what GPA the
pre-health advising office really expects, and whether transferring in is worth it.

This knowledge is valuable precisely *because* it is hard to find through official
channels. The university's course catalog lists who teaches BIOL 2311 but says
nothing about their lecturing style or exam difficulty; the housing handbook
describes the lottery but not which off-campus complexes students warn each other
about. That experiential, opinion-based knowledge lives in Reddit threads, Rate My
Professors reviews, and word-of-mouth — scattered, unsearchable, and quick to age
out. This guide consolidates it into one queryable, cited corpus.

---

## Document Sources

30 source files across four formats. Together they cover transfers, professors,
pre-health/pre-med, research, clubs, housing, parking, and general first-year life.

**Official UTD documentation** — `documents/Official website documentation/` (3 PDFs)

| # | Source | Type | Location |
|---|--------|------|----------|
| 1 | Pre-Health Starter Kit 2024 | PDF | `UTD-Pre-Health-Starter-Kit-2024-DB.pdf` |
| 2 | HPAC pages (merged): starter kit, application support, contact, research, student orgs, summer programs, success rates, JAMP, other opportunities | PDF | `Pre-health_merged.pdf` |
| 3 | Natural Sciences & Mathematics (NSM) pages | PDF | `NSM_pages.pdf` |

**Reddit — r/utdallas + r/TransferToTop25** — `documents/reddit/` (mixed)

| # | Source | Type | Location |
|---|--------|------|----------|
| 4 | Housing Megathread (Fall Semester) | PDF | `Housing Megathread - Fall Semester _ r_utdallas.pdf` |
| 5 | Academic tips for UTD freshmen (curated) | TXT | `Academic Tips for UTD Freshmen.txt` |
| 6 | r/utdallas comment threads (16 files: base + (1)–(15)) | Markdown | `post_comments-utdallas-2026-06-09*.md` |
| 7 | Transfer-to-Top-25 discussion | Markdown | `post_comments-TransferToTop25-2026-06-09.md` |
| 8 | Subreddit posts metadata (998 posts: id, title, author, score, num_comments, created_utc, selftext) | CSV | `subreddit_posts-utdallas-2026-06-09.csv` |
| 9 | r/utdallas comment threads, structured (body, author, score, parent_id, depth) — 9 files | CSV | `post_comments-utdallas-2026-06-09*.csv` |

**Rate My Professors** — `documents/RMP/` (1 TXT)

| # | Source | Type | Location |
|---|--------|------|----------|
| 10 | Rate My Professors reviews for UTD professors, combined | TXT | `RMP Reviews combined.txt` |

Reddit was collected via `documents/reddit/redditjsonretrieval.py` (public `.json`
endpoint, rate-limited at 100 items).

---

## Chunking Strategy

**Chunk size:** 500 tokens (target range 400–500), measured with the embedding
model's own tokenizer (`multi-qa-mpnet-base-dot-v1`).

**Overlap:** 75 tokens, carried sentence-by-sentence between consecutive chunks.

**Why these choices fit the documents.** The corpus mixes two shapes. Most Reddit
posts and comments are short and self-contained (≈100–300 tokens), while the
official UTD PDFs are long-form and densely structured. A 500-token chunk keeps an
entire short post or review intact in one chunk, yet is large enough to capture a
coherent section of a long PDF. Critically, 500 tokens stays inside the embedding
model's **512-token window**, so no chunk is silently truncated at embedding time.
The 75-token overlap exists so a fact that straddles a boundary — e.g. a
professor's name in one sentence and their rating in the next — remains
retrievable from at least one chunk. Splitting is **sentence-aware** (paragraph →
line → sentence), not a blind fixed-width cut, so chunks end on natural
boundaries. Chunks under 12 tokens (bare titles, one-word replies) are dropped as
fragments that carry no standalone meaning.

**Preprocessing before chunking.** Each file type gets type-specific cleaning in
[ingest.py](ingest.py):
- **All types:** decode HTML entities, strip HTML tags, remove markdown
  images/links and bare URLs, un-escape Reddit markdown (`\-`, `\^`), normalize
  whitespace.
- **PDF:** drop web-print chrome — date-time running headers, URL/page-number
  footers, nav (`Skip to main content`, `Log In`), ads (Washington Post,
  Promoted, `Learn More`), and Reddit sidebar bleed.
- **CSV:** `subreddit_posts` → one record per post (title + selftext); comment
  CSVs → one threaded record per file (post + comments, with author/score);
  `[removed]`/`[deleted]` comments dropped.
- **RMP TXT:** strip the rating-histogram and UI scaffolding, keep professor name,
  course, and review text.

**Final chunk count:** **1196 chunks** (.txt 50 · .md 57 · .pdf 85 · .csv 1004).
0 empty chunks, 0 chunks over 600 tokens; token sizes min/avg/max = 12 / 163 / 500.

---

## Sample Chunks

Five representative chunks, each labeled with its source document. (Reproduce with
`python ingest.py chunks`.)

**1. `Academic Tips for UTD Freshmen.txt` (chunk 0, 492 tokens)**
> Academic Tips for UTD Freshmen. Start Studying Early: Begin preparing for finals
> from the beginning of the semester by creating cheat sheets and memorization
> aids. "Start studying for finals from the beginning of the semester." Attend
> Classes: Even if attendance is optional, prioritize going to class… Utilize
> Resources: Make use of the library, study spaces, and office hours to enhance
> your learning. "Go to office hours at least once…"

**2. `RMP Reviews combined.txt` (chunk 11, 484 tokens)**
> Very easy to get an A; I didn't show up to class, but I'm sure the extra credit
> would've made it even easier. His slides are very good and detailed… The least
> caring teacher I've ever seen, also quite rude. Bad lecturer as well, speaks at
> an extreme pace. … BIOL3455 Dec 17th, 2025 For Credit: Yes Attendance: Mandatory
> Grade: B Textbook: Yes…

**3. `Pre-health_merged.pdf` (chunk 28, 493 tokens)**
> Tutor and teach others if you can. Apply science through research or independent
> study. FOCUSED PREP TIME! Study the test content and format. Plan your studies.
> – usually 1 week. Review basic material – usually 4 to 6 weeks. Work LOTS of
> practice passages and read the answer explanations – usually 8 to 10 weeks. Take
> the test. In early summer, apply to professional schools…

**4. `post_comments-utdallas-2026-06-09 (2).md` (chunk 17, 486 tokens)**
> Otherwise, go to UTD for the business school given that's what you want to study.
> UTD will get you a solid Dallas job. > > **BeastMasterAlphaCo** · 1 points · …
> Ok I disagree, better off doing Econ at UT than transferring to UTD. I've never
> met one UTD grad in investment banking and I ran recruiting for my team for 3
> years… UTD has zero credibility in finance and zero network from what I have seen.

**5. `subreddit_posts-utdallas-2026-06-09.csv#1tuq1am` (chunk 0, 53 tokens)**
> JSOM Service Hours Class or Volunteer Hours. Hey y'all, is it better to do the
> 100 service hours for JSOM or just take the class? I just completed my freshman
> year, so I still have time and was wondering which option is more worth it.

---

## Embedding Model

**Model used:** `sentence-transformers/multi-qa-mpnet-base-dot-v1` — 768-dimensional
embeddings, a 512-token context window, and a **dot-product** similarity objective.
It runs locally (no API key, no rate limits) and was trained on 215M
question→answer pairs specifically for **asymmetric semantic search** (short query
→ longer passage), which is exactly the shape of this system. Chunks are stored in
**ChromaDB** using the inner-product space (`hnsw:space: "ip"`) to match the
model's dot-product objective. We chose it over the assignment's default
`all-MiniLM-L6-v2` because MiniLM's 256-token window would silently truncate our
400–500-token chunks, discarding the back half of every long chunk at embedding
time.

**Production tradeoff reflection.** multi-qa-mpnet is a strong, portable local
model, but for a real deployment I'd weigh several factors before committing:

- **Domain accuracy.** Off-the-shelf models don't deeply "understand" UTD-specific
  jargon (ECS, JSOM, Comet Card, HPAC, JAMP). For production I'd evaluate a
  domain-fine-tuned model or an instruction-tuned one (e.g. `instructor-xl`) and
  measure retrieval recall on a labeled query set before switching.
- **Cost vs. latency vs. control.** A hosted API like OpenAI's
  `text-embedding-3-large` would likely improve accuracy and remove the local
  compute burden, but adds per-call cost, network latency, and a dependency on an
  external service. A local model keeps data private and costs fixed — worth it for
  a student tool with <2k chunks.
- **Context length.** Longer-context embedders (1k–8k tokens) would let me embed
  whole long PDF sections without splitting, reducing boundary loss — at the cost
  of coarser retrieval granularity.
- **Multilingual support.** UTD has a large international population; if queries
  arrived in other languages, a multilingual embedder (e.g. `multilingual-e5`)
  would matter. For this English corpus it isn't a priority.

---

## Retrieval Test Results

Retrieval embeds the query with the same model and returns the top-k (k=5) chunks
by similarity. Because the model uses **dot-product** similarity, scores are
reported as **dot-product magnitude where higher = more relevant** (the
assignment's "distance < 0.5" heuristic assumes cosine/normalized embeddings and
does not apply here). Relevance is judged by score *separation* plus inspection:
strong matches score ≈27–29, weak/tangential matches ≈18–22. Reproduce with
`python retrieve.py`.

### Query A — "Which professor should I take for BIO 2311 at UTD?"
| Rank | Score | Source | Chunk |
|---|---|---|---|
| 1 | 29.5 | `subreddit_posts…#1siuxmm` | "Biology lab at UTD Help!! Which one should I pick BIOL 2281 1106 1107?" |
| 2 | 28.4 | `RMP Reviews combined.txt` | "She gives so many chances for extra credit… Take this class!!!! … BIOL2312" |
| 3 | 27.8 | `subreddit_posts…#1t5j1g0` | "Biology 2311 Help … (eberhard voit). If I wanted to switch out … which prof would be better for my GPA…" |

**Why these are relevant:** all three center on choosing a *biology* professor at
UTD. Result #3 is almost the exact question — a student asking which BIOL 2311
professor to take to protect their GPA — and #2 is a concrete Rate My Professors
review of a bio professor with grading detail. The system correctly pulled from
*both* the Reddit (peer opinion) and RMP (review) sources that hold this answer.

### Query B — "How do UTD pre-med students typically get clinical hours in Dallas?"
| Rank | Score | Source | Chunk |
|---|---|---|---|
| 1 | 27.4 | `subreddit_posts…#1sio3uz` | "Any suggestions for getting clinical hours over the summer in Dallas? … Paid or volunteer opportunities are both fine." |
| 2 | 25.2 | `Pre-health_merged.pdf` | "research training … 10-week program, $4750 stipend … Green Fellowships … UT Southwestern…" |
| 3 | 24.4 | `Pre-health_merged.pdf` | "Pre-health Living and Learning Communities … Applying to Professional School … Success Rates…" |

**Why these are relevant:** #1 is the same question phrased almost identically
(getting clinical/volunteer hours in Dallas as a pre-med), and #2/#3 come from the
official HPAC pre-health document describing research programs and UT Southwestern
opportunities — the formal counterpart to the Reddit question. Retrieval bridged
the colloquial query to both peer and official sources.

### Query C — "What are the unwritten rules for finding parking on campus without a permit?"
| Rank | Score | Source | Chunk |
|---|---|---|---|
| 1 | 23.4 | `subreddit_posts…#1s85mxq` | "Get your act together, UTD Parking & Transportation … tightened its grip on parking…" |
| 2 | 22.3 | `subreddit_posts…#1s8d7me` | "Neighborhood Parking … I don't want to buy a parking permit cuz it's a waste of money so is it okay to park in the neighborhood streets…" |
| 3 | 21.3 | `subreddit_posts…#1sumjnd` | "utd parking help … where did your photographer park? … pay by space meter parking…" |

**Why these are relevant:** result #2 is precisely the question — avoiding a permit
by parking on neighborhood streets and whether you'll be ticketed — and the others
are on-topic parking complaints/logistics. All come from the Reddit post corpus
where this kind of unofficial advice lives.

> **Note (a retrieval limitation to revisit):** the 5th eval query, "wait times at
> Dining Hall West during lunch," retrieves only tangential food/dining posts
> (top score ≈21). The corpus simply contains no discussion of dining-hall wait
> times, so retrieval surfaces the nearest neighbors rather than a true answer.
> This is a content-coverage gap, not a retrieval bug — see Failure Case Analysis.

---

## Grounded Generation

Generation uses **Groq `llama-3.3-70b-versatile`** ([generate.py](generate.py)) at
`temperature=0`. The end-to-end flow is in [query.py](query.py):
`retrieve(question)` → `generate_answer(question, chunks)` → attach sources.

**How grounding is enforced (not just suggested).** The retrieved chunks are
formatted into a numbered, source-labeled context block, and the **system prompt**
hard-constrains the model to that block:

> 1. Use ONLY information stated in the provided context documents. Never use your
>    own general or prior knowledge, even if you are confident.
> 2. If the context does not contain enough information to answer, reply with
>    EXACTLY this sentence and nothing else: *"I don't have enough information on that."*
> 3. When you do answer, cite the document(s) you used inline, e.g.
>    "(source: RMP Reviews combined.txt)". Only cite documents that actually support your answer.
> 4. Reflect what students/sources actually said — do not invent names, numbers,
>    professors, or policies that are not in the context.

Three structural choices reinforce this: (a) `temperature=0` minimizes drift from
the context; (b) the fixed refusal string is matched programmatically so the UI
can suppress sources on a refusal; (c) if retrieval returns nothing, `generate.py`
returns the refusal **without even calling the model**.

**How source attribution is surfaced.** Attribution is **programmatic, not left to
the LLM**: [query.py](query.py) collects the unique source documents of the
retrieved chunks and returns them as a `sources` list that the UI always displays
("Retrieved from"). The model is *also* asked to cite inline, but the displayed
source list is guaranteed by the pipeline. On a refusal, the source list is empty.

### Example responses (real output, `temperature=0`)

**Example 1 — in-scope, grounded + cited**
> **Q:** How do UTD pre-med students get clinical hours in Dallas?
> **A:** UTD pre-med students can get clinical hours in Dallas through various
> opportunities such as the Parkland Hospital Clinical Experience
> (source: Pre-health_merged.pdf), volunteering with student organizations like
> Molding Doctors, which offers clinical events (source: Pre-health_merged.pdf), or
> by finding summer programs/internships listed in resources like the AAMC Database
> or ExploreHealthCareers Database (source: Pre-health_merged.pdf).
> **Retrieved from:** `subreddit_posts-…csv#1sio3uz`, `Pre-health_merged.pdf`, `UTD-Pre-Health-Starter-Kit-2024-DB.pdf`

**Example 2 — in-scope, grounded + cited**
> **Q:** Can I park on campus without buying a permit?
> **A:** It seems that parking on campus without a permit can be problematic.
> According to a student's experience, they were told they could park in the
> visitor's section but were still ticketed (source: subreddit_posts-…csv#1s85mxq).
> Another student is considering parking in neighborhood streets but is unsure if
> they would get ticketed or towed (source: subreddit_posts-…csv#1s8d7me). There is
> also a pay-by-space meter parking option (source: subreddit_posts-…csv#1sumjnd).
> **Retrieved from:** five `subreddit_posts-…csv` parking posts (#1s8d7me, #1s85mxq, #1sumjnd, #1tfyplk, #1tqlh6l)

**Example 3 — out-of-scope, refusal**
> **Q:** Who won the 2022 FIFA World Cup?
> **A:** I don't have enough information on that.
> **Retrieved from:** (none — the system declined to answer)

The refusal fires even though the corpus is "about UTD" — for a question the
documents don't cover, the model returns the exact refusal rather than answering
from training knowledge.

---

## Query Interface

A **Gradio** web UI ([app.py](app.py)), launched with `python app.py` at
http://localhost:7860. It is intentionally minimal so a viewer can use it without
narration.

**Input field**
- **Your question** — a textbox; submit by clicking **Ask** or pressing Enter.
- Four clickable **example questions** are provided to seed a query.

**Output fields**
- **Answer** — the grounded, source-cited response (or the refusal string).
- **Retrieved from** — a bulleted list of the source document(s) the answer drew
  from; shows "(none — the system declined to answer from the corpus)" on a refusal.

**Sample interaction transcript**
```
Your question:  Can I park on campus without buying a permit?

[Ask]

Answer:
It seems that parking on campus without a permit can be problematic. According to
a student's experience, they were told they could park in the visitor's section
but were still ticketed (source: subreddit_posts-…csv#1s85mxq). Another student is
considering parking in neighborhood streets but is unsure if they would get
ticketed or towed (source: subreddit_posts-…csv#1s8d7me). There is also a
pay-by-space meter parking option (source: subreddit_posts-…csv#1sumjnd).

Retrieved from:
• subreddit_posts-utdallas-2026-06-09.csv#1s8d7me
• subreddit_posts-utdallas-2026-06-09.csv#1s85mxq
• subreddit_posts-utdallas-2026-06-09.csv#1sumjnd
• subreddit_posts-utdallas-2026-06-09.csv#1tfyplk
• subreddit_posts-utdallas-2026-06-09.csv#1tqlh6l
```

---

## Evaluation Report

<!-- Milestone 6 — all 5 questions with expected answers, system responses,
     accuracy judgments. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

---

## Failure Case Analysis

Three distinct failure cases surfaced, each tied to a specific pipeline stage.
(Full evaluation-table results are in the Evaluation Report — Milestone 6.)

### Failure 1 — Content-coverage gap (ingestion / corpus): "Dining Hall West wait times"
**Question:** "What do students say about wait times at Dining Hall West during lunch?"
**What the system returned:** *"I don't have enough information on that."* (refusal).
**Root cause — ingestion/corpus coverage, not retrieval or generation.** The corpus
contains no posts discussing dining-hall wait times. Retrieval therefore returns
only tangential food/dining posts (a food-access survey, "eating in the library",
power-outage posts) with notably lower similarity (top dot-product ≈21 vs ≈27–29
for well-covered topics), and generation correctly declines rather than inventing
an answer. The failure is that the source documents never covered this subtopic.
**Fix:** collect dining-specific sources (a dining-hall megathread, Yelp/Google
reviews of campus dining) so the subtopic is represented before queries hit it.

### Failure 2 — Questions without answers (ingestion structure / retrieval): "BIO 2311 professor"
**Question:** "Which professor should I take for BIO 2311 at UTD?"
**What the system returned:** *"I don't have enough information on that."* (refusal),
despite *topically* relevant retrieval (scores 27–29).
**Root cause — corpus structure at the ingestion stage.** `subreddit_posts.csv`
stores each post's *body* (the question) but **not its comment replies**, and the
separately-ingested comment threads cover other topics. So for this query
retrieval surfaces students *asking* which professor to take, plus a tangential
BIOL2312 RMP review — but no chunk actually *names or recommends* a BIOL 2311
professor. Grounding then forces a refusal instead of a fabricated recommendation.
The retrieval is on-topic; the answer-bearing text was simply never ingested.
**Fix:** ingest the comment replies for these posts (the answers live there), raise
top-k, or build a professor-name → review index from the RMP file.

### Failure 3 — Garbled PDF extraction (ingestion / PDF parsing): Housing Megathread
**Affected source:** `Housing Megathread - Fall Semester _ r_utdallas.pdf`.
**What happens:** pdfplumber reads this reddit web-print's multi-column layout in
the wrong order, interleaving text character-by-character (e.g. "Skip to1 mmoarien
rceopnlytent" = "Skip to main content" merged with "1 more reply"). The resulting
~9 chunks are low-quality word salad that can pollute retrieval for housing queries.
**Root cause — PDF extraction stage.** We empirically confirmed this salad is
neither auto-detectable (stopword ratio, period density, and consonant-run metrics
all overlap clean prose) nor reconstructable (the infographic/columns have no
x-gaps to split on). ~1% of the 1196 chunks are affected.
**Fix:** layout-aware/column-aware extraction, or re-collect the thread as text
(via the Reddit API / `.json`) instead of a printed-to-PDF web page.

---

## Spec Reflection

<!-- Milestone 6 — one way the spec helped, one way the implementation diverged
     and why. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

---

## AI Usage

<!-- To be completed by the student. Describe at least 2 specific instances where
     you used an AI tool: what you directed it to do, and what you reviewed,
     revised, or overrode. -->

**Instance 1**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*
