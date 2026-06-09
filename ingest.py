"""Milestone 3 — Document ingestion + chunking for The Unofficial Guide.

Loads every document in documents/ (.txt, .md, .pdf, .csv), cleans out
boilerplate, and splits the result into token-based chunks sized for the
embedding model (multi-qa-mpnet-base-dot-v1, 512-token window).

Per planning.md:
  - chunk_text(text, chunk_size=500, overlap=75)  [tokens, mpnet tokenizer]

Run modes (built up iteratively while developing — see __main__):
  python ingest.py txt      # test the .txt loader/cleaner
  python ingest.py md       # test the .md loader
  python ingest.py pdf      # test the .pdf loader/cleaner
  python ingest.py csv      # test the .csv loader
  python ingest.py chunks   # full pipeline: per-type chunk counts + samples
"""

import csv
import html
import re
import sys
from functools import lru_cache
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Chunks below this many tokens are dropped as meaningless fragments
# (bare post titles, one-word replies). See chunk_text().
MIN_CHUNK_TOKENS = 12

# Allow large CSV fields (reddit selftext / comment bodies can be long).
csv.field_size_limit(10_000_000)


# =========================================================================
# Cleaning
# =========================================================================

def clean_common(text):
    """Generic cleanup applied to every document regardless of type:
    decode HTML entities, strip HTML/XML tags, normalize whitespace.
    """
    text = html.unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)      # markdown images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # markdown links -> link text
    text = re.sub(r"https?://\S+", " ", text)              # bare URLs
    text = re.sub(r"\\([-^*_~`#.>])", r"\1", text)         # reddit escaped markdown (\- \^ ...)
    text = re.sub(r"<[^>]+>", " ", text)          # strip any HTML tags
    # Normalize odd unicode spaces to plain spaces.
    text = text.replace(" ", " ").replace("​", "")
    # Collapse runs of spaces/tabs within a line.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ blank lines down to a single paragraph break.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# Boilerplate that shows up on web-print PDFs (page headers/footers, nav, ads).
_PDF_DROP_LINE = re.compile(
    r"""^\s*(
        \d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)? |   # date-time header
        https?://\S+ |                                          # bare URL (footer)
        \d+\s*/\s*\d+ |                                          # page "1/7"
        skip\ to\ main\ content |
        find\ anything\ ask\ log\ in | find\ anything | log\ in | sign\ in |
        people\ also\ ask\ about |
        related\ posts | learn\ more | more\ replies | reply\ share |
        rereddit.* | top\ posts\ of.* | reddit |
        .*\bpromoted\b.* | .*washington\ post.* |
        .*kybr\.dev.* | stuck\ on\ a\ bug\?.* | we\ fix\ it\.?.* |
        \d+\s+\d+ |                                  # reddit vote pair "0 69"
        \d+\s+(upvotes?|comments?|points?)\b.* |     # "1 upvote · 2 comments"
        dm\ me\b.* |
        enroll\ now! |
        ad | mod | op |
        \W*                                                     # punctuation-only line
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def clean_pdf_text(text):
    """Drop the running headers / footers / nav / ad lines left by printing a
    web page to PDF, then apply the common cleanup."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        # Drop the "<date> <time> <Page Title> | The University of Texas..." header.
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}", stripped):
            continue
        # Drop any line containing a URL — on these web-print PDFs URLs are
        # footers/nav, often trailed by a "N/M" page marker.
        if "http://" in stripped or "https://" in stripped:
            continue
        # Drop standalone page markers like "1/7" or "Page 3 of 10".
        if re.match(r"^(page\s+)?\d+\s*(/|of)\s*\d+$", stripped, re.IGNORECASE):
            continue
        if _PDF_DROP_LINE.match(stripped):
            continue
        kept.append(stripped)
    return clean_common("\n".join(kept))


# RMP scrape chrome: histogram rows, action buttons, sidebar, ad markers.
_RMP_DROP_LINE = re.compile(
    r"""^\s*(
        rate | compare | arrow\ icon | ad |
        i'?m\ professor\b.* |
        similar\ professors |
        rating\ distribution |
        (awesome|great|good|ok|awful)\ [1-5] |   # histogram labels
        would\ take\ again |
        level\ of\ difficulty |
        \d+% |                                    # would-take-again percent on its own line
        \d+(\.\d+)? |                             # bare rating number on its own line
        /\ ?5
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def clean_rmp_text(text):
    """Strip Rate My Professors UI scaffolding, keeping professor names,
    department/quality context, and the actual review text."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _RMP_DROP_LINE.match(stripped):
            continue
        kept.append(stripped)
    return clean_common("\n".join(kept))


# =========================================================================
# Per-type loaders  ->  each returns a list of {"source", "text"} records
# =========================================================================

def load_txt(path):
    """Plain-text. RMP file gets RMP-specific cleaning; others get common."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if "rmp" in path.name.lower() or "ratemyprofessor" in raw.lower()[:200]:
        text = clean_rmp_text(raw)
    else:
        text = clean_common(raw)
    return [{"source": path.name, "text": text}] if text.strip() else []


def load_md(path):
    """Markdown reddit threads — already readable; just common cleanup."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_common(raw)
    return [{"source": path.name, "text": text}] if text.strip() else []


def load_pdf(path):
    """PDF via pdfplumber. Concatenate all pages, then strip web-print chrome.
    (pdfplumber does not OCR; scanned PDFs yield empty text and are skipped.)"""
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                pages.append(extracted)
    text = clean_pdf_text("\n".join(pages))
    return [{"source": path.name, "text": text}] if text.strip() else []


def load_csv(path):
    """CSV reddit exports. Two shapes:
       - subreddit_posts (has 'title'+'num_comments'): one record per POST row.
       - comment threads (has 'body'+'parent_id'): one record per FILE,
         concatenating the root post and its comments to preserve thread context.
    """
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    cols = set(rows[0].keys())

    records = []
    if {"title", "num_comments"} <= cols:
        # Post list: each row is its own self-contained post record.
        for r in rows:
            title = (r.get("title") or "").strip()
            body = (r.get("selftext") or "").strip()
            score = (r.get("score") or "").strip()
            parts = [p for p in [title, body] if p]
            if not parts:
                continue
            text = clean_common("\n\n".join(parts))
            if text.strip():
                records.append({
                    "source": f"{path.name}#{r.get('id', '')}",
                    "text": text,
                })
    elif "body" in cols:
        # Comment thread: post (first row) + comments, as one threaded document.
        lines = []
        for r in rows:
            body = (r.get("body") or "").strip()
            if not body or body.lower() in ("[removed]", "[deleted]"):
                continue
            author = (r.get("author") or "anon").strip()
            score = (r.get("score") or "").strip()
            tag = "POST" if str(r.get("name", "")).startswith("t3_") else f"u/{author}"
            score_str = f" ({score} pts)" if score else ""
            lines.append(f"{tag}{score_str}: {body}")
        text = clean_common("\n\n".join(lines))
        if text.strip():
            records.append({"source": path.name, "text": text})
    return records


_LOADERS = {
    ".txt": load_txt,
    ".md": load_md,
    ".pdf": load_pdf,
    ".csv": load_csv,
}


def load_documents():
    """Load + clean every supported file under documents/ (recursively).
    Returns a list of {"source", "text"} records (CSV post-lists expand to
    one record per post)."""
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"documents/ not found at {DOCUMENTS_DIR}")

    records = []
    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue  # skip .gitkeep, .py, images, etc.
        try:
            records.extend(loader(path))
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"  [error] {path.name}: {e}")
    return records


# =========================================================================
# Chunking  (token-based, sentence-aware; sized for the mpnet 512 window)
# =========================================================================

@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def _ntok(tokenizer, s):
    return len(tokenizer.encode(s, add_special_tokens=False))


def _split_sentences(text):
    """Split into sentence-ish units, respecting paragraph breaks first."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # Split on sentence boundaries but keep list/line structure intact.
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return units


def chunk_text(text, chunk_size=500, overlap=75, tokenizer=None):
    """Split text into <=chunk_size-token chunks (sentence-aware) with ~overlap
    tokens carried between consecutive chunks. Returns a list of strings."""
    tokenizer = tokenizer or _get_tokenizer()
    units = _split_sentences(text)
    if not units:
        return []

    chunks = []
    cur, cur_tok = [], 0

    def flush():
        if cur:
            chunk = " ".join(cur).strip()
            if chunk:
                chunks.append(chunk)

    for unit in units:
        ut = _ntok(tokenizer, unit)
        if ut > chunk_size:
            # A single oversized unit: flush, then hard-split it by tokens.
            flush()
            cur, cur_tok = [], 0
            ids = tokenizer.encode(unit, add_special_tokens=False)
            for i in range(0, len(ids), chunk_size):
                piece = tokenizer.decode(ids[i:i + chunk_size]).strip()
                if piece:
                    chunks.append(piece)
            continue
        if cur_tok + ut > chunk_size:
            flush()
            # Seed next chunk with trailing units worth ~overlap tokens.
            keep, kept_tok = [], 0
            for u in reversed(cur):
                t = _ntok(tokenizer, u)
                if kept_tok + t > overlap:
                    break
                keep.insert(0, u)
                kept_tok += t
            cur, cur_tok = keep[:], kept_tok
        cur.append(unit)
        cur_tok += ut
    flush()
    # Drop fragments too small to carry standalone meaning (bare titles,
    # "Thanks!" comments, etc.) — they add noise to the vector store.
    return [c for c in chunks if c.strip() and _ntok(tokenizer, c) >= MIN_CHUNK_TOKENS]


def chunk_records():
    """Full corpus as chunk dicts ready for the vector store:
    [{"text", "source", "chunk_index"}], where chunk_index is the chunk's
    position within its source document. Used by build_index.py."""
    tokenizer = _get_tokenizer()
    out = []
    for rec in load_documents():
        for j, ch in enumerate(chunk_text(rec["text"], tokenizer=tokenizer)):
            out.append({"text": ch, "source": rec["source"], "chunk_index": j})
    return out


# =========================================================================
# Dev/verification harness
# =========================================================================

def _safe(s):
    return s.encode("ascii", "replace").decode()


def _test_loader(ext):
    """Load every file of one extension and print a cleaned preview."""
    paths = sorted(p for p in DOCUMENTS_DIR.rglob(f"*{ext}") if p.is_file())
    print(f"{ext}: {len(paths)} file(s)")
    recs_total = 0
    for path in paths:
        recs = _LOADERS[ext](path)
        recs_total += len(recs)
        print(f"  {path.name:55} -> {len(recs)} record(s)")
    print(f"total records from {ext}: {recs_total}\n")
    # Print one cleaned preview from the first file.
    if paths:
        recs = _LOADERS[ext](paths[0])
        if recs:
            print("=" * 70)
            print(f"CLEANED PREVIEW — {recs[0]['source']} ({len(recs[0]['text'])} chars)")
            print("=" * 70)
            print(_safe(recs[0]["text"][:1500]))


def _type_of(source):
    """Infer the file type of a record from its source filename."""
    for ext in (".pdf", ".csv", ".md", ".txt"):
        if ext in source.lower():
            return ext
    return "?"


def _run_full_verification():
    """Full pipeline: load -> chunk everything, then print per-type chunk
    counts, 3 sample chunks per type, the total, and quality checks
    (no empty chunks, none over the 600-token guardrail)."""
    print("Loading documents...")
    records = load_documents()
    print(f"Loaded {len(records)} record(s) from documents/\n")
    print("Loading tokenizer + chunking (first run downloads the tokenizer)...")
    tokenizer = _get_tokenizer()

    by_type = {}   # ext -> list of (source, chunk_text)
    for rec in records:
        ext = _type_of(rec["source"])
        for ch in chunk_text(rec["text"], tokenizer=tokenizer):
            by_type.setdefault(ext, []).append((rec["source"], ch))

    all_chunks = [c for chunks in by_type.values() for c in chunks]
    total = len(all_chunks)

    print("\n" + "=" * 70)
    print("PER-TYPE CHUNK COUNTS")
    print("=" * 70)
    for ext in (".txt", ".md", ".pdf", ".csv"):
        print(f"  {ext:5} -> {len(by_type.get(ext, [])):>5} chunks")
    print(f"  TOTAL -> {total} chunks")

    # First 3 chunks per type (the AI Tool Plan verification step).
    for ext in (".txt", ".md", ".pdf", ".csv"):
        chunks = by_type.get(ext, [])
        print("\n" + "=" * 70)
        print(f"FIRST 3 CHUNKS — {ext}  ({len(chunks)} total)")
        print("=" * 70)
        for src, ch in chunks[:3]:
            ntok = _ntok(tokenizer, ch)
            print(f"\n[{src} | {ntok} tokens | {len(ch)} chars]")
            print(_safe(ch[:600]))

    # Quality checks.
    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)
    empties = sum(1 for _, c in all_chunks if not c.strip())
    tok_counts = [_ntok(tokenizer, c) for _, c in all_chunks]
    over_600 = sum(1 for n in tok_counts if n > 600)
    print(f"  empty chunks:            {empties}   (want 0)")
    print(f"  chunks over 600 tokens:  {over_600}   (want 0)")
    if tok_counts:
        print(f"  token size  min/avg/max: "
              f"{min(tok_counts)} / {sum(tok_counts)//len(tok_counts)} / {max(tok_counts)}")
    if total < 50:
        print("  [note] <50 chunks — chunks may be too large.")
    elif total > 2000:
        print("  [note] >2000 chunks — chunks may be too small / corpus very large.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chunks"
    if mode in (".txt", "txt"):
        _test_loader(".txt")
    elif mode in (".md", "md"):
        _test_loader(".md")
    elif mode in (".pdf", "pdf"):
        _test_loader(".pdf")
    elif mode in (".csv", "csv"):
        _test_loader(".csv")
    else:
        _run_full_verification()
