"""
Week 1 - Step 1: Ingestion.

Reads every .txt and .pdf file in data/raw/, splits it into overlapping
chunks, embeds each chunk locally with sentence-transformers, and stores
the vectors + text in a persistent ChromaDB collection.

Run:
    python src/ingest.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    DATA_RAW_DIR, CHROMA_DIR, EMBED_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME,
)

# Matches the start of a numbered section/clause, e.g. "\n7. " or "\n19.  "
# This is how Acts, statutes, and most numbered legal/policy documents are
# structured, so splitting on this boundary keeps each clause intact instead
# of slicing it in half by raw character count.
SECTION_PATTERN = re.compile(r'(?:^|\n)\s*(\d{1,3})\.\s+')

# "CHAPTER I" only appears once per genuine pass through an Act — far more
# textually stable than individual section-number digits, which PDF text
# extraction frequently mangles (missing spaces, OCR noise, broken headers).
# We use this as the primary anchor for finding the one authoritative copy
# of the Act body, rather than trying to count numbers 1, 2, 3... which
# breaks the moment a single header gets garbled.
CHAPTER_ONE_PATTERN = re.compile(r'\bCHAPTER\s+I\b')
# Matches the real schedule HEADING ("THE FIRST SCHEDULE" / "THE SECOND
# SCHEDULE"), which the Gazette prints in full uppercase. Deliberately
# case-SENSITIVE: the Act's own body text repeatedly cross-references
# these same words inline in normal sentence case (e.g. "...according to
# the form set out... in the First Schedule.", "...organisations
# specified in the Second Schedule..."). Matching case-insensitively
# caught those inline mentions too and truncated the verified span far
# too early (as early as Section 5-6, well before the real appendix).
# Requiring exact uppercase avoids this — same reasoning applied to
# CHAPTER_ONE_PATTERN above, since running page headers print "Chapter I"
# in mixed case while real chapter headings print "CHAPTER I" in caps.
SCHEDULE_PATTERN = re.compile(r'\bTHE\s+(?:FIRST|SECOND)\s+SCHEDULE\b')


def _read_pdf(path) -> str:
    """Extract all text from a PDF, preferring `pdftotext` (poppler) over
    pypdf.

    Different PDF text-extraction engines can produce meaningfully
    different output from the exact same file — different line-break
    placement, spacing around numbers, and handling of multi-column
    layouts. For this project's Gazette-style PDF, pypdf was found to
    mangle several numbered section headers that pdftotext extracts
    cleanly, which broke downstream section detection. pdftotext
    (poppler-utils) consistently produced the more reliable output during
    testing, so we use it when available and fall back to pypdf only if
    pdftotext isn't installed on the machine.
    """
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout

    print(f"  NOTE: pdftotext unavailable or failed for {path.name}, "
          f"falling back to pypdf (install poppler-utils for better extraction quality: "
          f"sudo apt install poppler-utils)")
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text_sliding(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character chunker with overlap.

    Used as a fallback for text with no detectable section structure
    (e.g. free-form prose, essays, unstructured notes).
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _find_authoritative_span(text: str):
    """Find the character span [start, end) of the ONE authoritative copy of
    the Act body, anchored on chapter/schedule markers rather than fragile
    section-number counting.

    Logic: start at the first "CHAPTER I". End at whichever comes first —
    the next "THE FIRST/SECOND SCHEDULE" marker (appendix material) or a second
    "CHAPTER I" (a duplicate reprint of the whole Act starting over).
    Everything inside this span is trustworthy; everything outside it
    (table of contents before the span, schedules/guides/reprints after)
    is not, regardless of what digit-based labels it carries.

    Returns None if no "CHAPTER I" marker is found at all, signalling the
    caller should fall back to the monotonic-number heuristic instead.
    """
    chapter_matches = list(CHAPTER_ONE_PATTERN.finditer(text))
    if not chapter_matches:
        return None

    start = chapter_matches[0].start()

    end_candidates = []
    if len(chapter_matches) > 1:
        end_candidates.append(chapter_matches[1].start())
    schedule_matches = [m.start() for m in SCHEDULE_PATTERN.finditer(text) if m.start() > start]
    if schedule_matches:
        end_candidates.append(schedule_matches[0])

    end = min(end_candidates) if end_candidates else len(text)
    return start, end


def _find_authoritative_range(matches) -> int:
    """FALLBACK ONLY (used when no CHAPTER I marker is found in the
    document at all). Find how far the first clean, monotonically-
    increasing run of section numbers (1, 2, 3, ...) extends through
    `matches`.

    This is less robust than _find_authoritative_span because a single
    garbled/missed section header anywhere in the document breaks the
    count for everything after it — even genuine later sections. Prefer
    the chapter-boundary approach whenever chapter markers are present.

    Returns the index (into `matches`) of the last match that belongs to
    the authoritative run, or -1 if no run starting at 1 is found at all.
    """
    expected = 1
    authoritative_end_idx = -1
    for idx, m in enumerate(matches):
        num = int(m.group(1))
        if num == expected:
            authoritative_end_idx = idx
            expected += 1
        elif num == expected + 1:
            # Tolerate a single missed match (e.g. OCR noise dropping one
            # section header) without abandoning an otherwise-clean run.
            authoritative_end_idx = idx
            expected = num + 1
        else:
            break
    return authoritative_end_idx


def _looks_like_legal_document(text: str) -> bool:
    """Gate: only attempt section-aware legal chunking (and citation
    verification) if the document actually looks like a statute/Act,
    rather than any generic text that happens to contain a clean 1, 2, 3...
    numbered list -- e.g. a numbered plan in a chat transcript, a recipe, a
    tutorial's steps. Without this gate, the monotonic-range fallback (see
    _find_authoritative_range) would happily treat any well-formed numbered
    list as "verified Act sections", which produces meaningless
    "Section N" citations on documents that were never structured that way.

    Requires at least one of:
    - An explicit "CHAPTER I" marker (the real Gazette convention)
    - The document naming itself as an Act ("...ACT, 2005" style)
    - Multiple genuine "Section N" self-references in the document's own
      text (capitalized, matching how statutes actually cite themselves)
    """
    if CHAPTER_ONE_PATTERN.search(text):
        return True
    if re.search(r'\bACT,?\s+\d{4}\b', text):
        return True
    if len(re.findall(r'\bSection\s+\d{1,3}\b', text)) >= 3:
        return True
    return False


def chunk_text_by_section(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Section-aware chunker for numbered legal/policy documents.

    Returns a list of {"text": str, "section_num": str|None, "verified": bool|None}.

    Splits on numbered section boundaries so each chunk corresponds to one
    complete clause. Critically, it also validates each section number
    against the document's true structure — primarily via chapter-boundary
    anchoring (_find_authoritative_span), falling back to monotonic-number
    counting (_find_authoritative_range) only if no chapter markers exist —
    so mislabeled fragments (table-of-contents entries, schedules, duplicate
    reprints, FAQ guides that reuse the same numbers for unrelated content)
    are marked unverified instead of being silently presented as real Act
    sections.

    Falls back to the plain sliding-window chunker (verified=None -> "n/a")
    if fewer than 3 section boundaries are detected, OR if the document
    doesn't pass _looks_like_legal_document -- e.g. a chat transcript with
    a clean numbered list would otherwise get its numbers mistaken for
    genuine Act sections.
    """
    text = text.strip()
    if not text:
        return []

    matches = list(SECTION_PATTERN.finditer(text))
    if len(matches) < 3 or not _looks_like_legal_document(text):
        return [
            {"text": t, "section_num": None, "verified": None}
            for t in chunk_text_sliding(text, chunk_size, overlap)
        ]

    span = _find_authoritative_span(text)
    if span is not None:
        auth_start, auth_end = span
        verified_flags = [auth_start <= m.start() < auth_end for m in matches]
    else:
        # No CHAPTER I marker found anywhere — fall back to the less
        # robust monotonic-number heuristic.
        authoritative_end_idx = _find_authoritative_range(matches)
        verified_flags = [i <= authoritative_end_idx for i in range(len(matches))]

    chunks = []
    # Keep any preamble before the first numbered section as its own chunk
    # (e.g. title, enactment clause) rather than discarding it.
    preamble = text[:matches[0].start()].strip()
    if preamble:
        for t in chunk_text_sliding(preamble, chunk_size, overlap):
            chunks.append({"text": t, "section_num": None, "verified": None})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue

        section_num = match.group(1)
        verified = verified_flags[i]

        if verified:
            unverified_notice = None
        else:
            unverified_notice = (
                f"[UNVERIFIED NUMBERING: the label '{section_num}.' below appears outside "
                f"the main sequential Act text (Sections 1-31). It likely comes from a table "
                f"of contents, schedule, duplicate reprint, or supplementary guide that reuses "
                f"this number for unrelated content. Do NOT state or imply this is "
                f"'Section {section_num}' of the Act.]"
            )

        if len(section_text) <= chunk_size * 1.5:
            body = section_text if verified else f"{unverified_notice}\n{section_text}"
            chunks.append({"text": body, "section_num": section_num, "verified": verified})
        else:
            # Long sections get sub-chunked, but each sub-chunk keeps its
            # verification status so partial chunks aren't accidentally
            # upgraded to "trusted" just because they got split.
            sub_chunks = chunk_text_sliding(section_text, chunk_size, overlap)
            for j, sub in enumerate(sub_chunks):
                if j == 0:
                    body = sub if verified else f"{unverified_notice}\n{sub}"
                elif verified:
                    body = f"[Section {section_num}, continued] {sub}"
                else:
                    body = f"{unverified_notice}\n[continued] {sub}"
                chunks.append({"text": body, "section_num": section_num, "verified": verified})

    return chunks


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract text from an in-memory uploaded file (e.g. from Streamlit's
    file_uploader, which provides bytes rather than a path on disk).

    Supports .pdf and .txt. PDFs are written to a temp file so we can reuse
    _read_pdf's pdftotext-based extraction (which needs a real file path).
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            return _read_pdf(Path(tmp.name))
    elif suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Only .pdf and .txt are supported.")


def load_documents(raw_dir) -> list[dict]:
    """Load every .txt and .pdf file in raw_dir, returning [{source, text}, ...]."""
    docs = []
    txt_files = sorted(raw_dir.glob("*.txt"))
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not txt_files and not pdf_files:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {raw_dir}. "
            f"Drop your source documents there first."
        )

    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({"source": path.name, "text": text})

    for path in pdf_files:
        print(f"  Extracting text from PDF: {path.name} ...")
        text = _read_pdf(path)
        if not text.strip():
            print(f"  WARNING: no extractable text found in {path.name} "
                  f"(it may be a scanned/image-based PDF, which needs OCR).")
        docs.append({"source": path.name, "text": text})

    return docs


def build_chunks_and_metadata(text: str, source_name: str) -> tuple[list[str], list[str], list[dict]]:
    """Chunk one document's text and build the (documents, ids, metadatas)
    lists ready for a Chroma collection.add() call. Shared by both the CLI
    ingest flow and any other caller (e.g. a Streamlit file upload) that
    wants to index a document without going through the fixed
    data/raw/ -> persistent-store pipeline.
    """
    chunks = chunk_text_by_section(text, CHUNK_SIZE, CHUNK_OVERLAP)
    doc_texts, ids, metas = [], [], []
    for i, chunk in enumerate(chunks):
        if chunk["verified"] is True:
            status = "verified"
        elif chunk["verified"] is False:
            status = "unverified"
        else:
            status = "n/a"  # no section structure applies (plain prose chunk)

        doc_texts.append(chunk["text"])
        ids.append(f"{source_name}::chunk_{i}")
        metas.append({
            "source": source_name,
            "chunk_index": i,
            "section_status": status,
            # Chroma metadata can't store None, so use "" as the
            # "no section number applies" sentinel (e.g. preamble chunks).
            "section_num": chunk["section_num"] or "",
        })
    return doc_texts, ids, metas


def index_documents(docs: list[dict], collection, embedder) -> int:
    """Chunk, embed, and add a list of {source, text} docs into an already-
    created Chroma collection. Works with ANY collection -- a persistent
    one (CLI ingest) or an ephemeral in-memory one (e.g. a Streamlit
    session's uploaded document) -- since the caller controls what
    `collection` actually is. Returns the total number of chunks indexed.
    """
    all_chunks, all_ids, all_metas = [], [], []
    for doc in docs:
        doc_texts, ids, metas = build_chunks_and_metadata(doc["text"], doc["source"])
        all_chunks.extend(doc_texts)
        all_ids.extend(ids)
        all_metas.extend(metas)

    if not all_chunks:
        raise ValueError("No chunks produced — check the source document isn't empty.")

    embeddings = embedder.encode(all_chunks).tolist()
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metas, embeddings=embeddings)
    return len(all_chunks)


def build_index():
    print(f"Loading documents from {DATA_RAW_DIR} ...")
    docs = load_documents(DATA_RAW_DIR)
    print(f"Loaded {len(docs)} document(s).")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' (CPU, first run downloads it) ...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh start each ingest run, so re-running never duplicates data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding and indexing chunks ...")
    n_chunks = index_documents(docs, collection, embedder)

    print(f"Done. Indexed {n_chunks} chunks from {len(docs)} document(s) "
          f"into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
