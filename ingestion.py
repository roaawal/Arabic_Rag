"""
Ingestion pipeline: PDFs -> cleaned pages -> chunks -> embeddings -> ChromaDB.

Run once, and again any time the PDFs in data/ change:

    python ingestion.py

Stages (all in this file):
  1. Extract   -- Gemini vision OCR of each page image (ocr_backends.py),
                  cached in ocr_cache/.
  2. Clean     -- strip running headers/footers repeated across a document's
                  pages, and warn if the source text looks corrupted.
  3. Sections  -- detect headings from each document's table of contents and
                  tag every page with the section it falls under.
  4. Chunk     -- Arabic-aware sentence packing with overlap; table-of-contents
                  chunks are skipped.
  5. Index     -- embed with BAAI/bge-m3 and rebuild the Chroma collection.
"""
import glob
import os
import re
from collections import Counter, defaultdict

import config

# ---------------------------------------------------------------------------
# 1. Extraction
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: str):
    """Yields (page_number, text) for one PDF; page_number is 1-indexed."""
    import ocr_backends
    yield from ocr_backends.extract_pages_cached(pdf_path)


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------


def strip_repeated_boilerplate(records: list, min_repeat_ratio: float = None) -> list:
    """
    Removes lines that repeat across many pages of ONE document (version
    tables, logos, page footers) so they don't pollute every chunk's
    embedding and BM25 tokens.

    `records` must be the page records of a single source document. Matching
    is digit-agnostic, so a header that only changes its page number
    ("15 /2") is still caught.
    """
    if min_repeat_ratio is None:
        min_repeat_ratio = config.HEADER_FOOTER_MIN_REPEAT_RATIO
    if len(records) < 3:
        # Too few pages to tell a repeated header from coincidence.
        return records

    def normalize(line: str) -> str:
        line = re.sub(r"\d+", "#", line)
        return re.sub(r"\s+", " ", line).strip()

    def too_generic(norm: str) -> bool:
        # A bare page number normalizes to "#"; nearly every page has one, but
        # it isn't shared boilerplate -- and stripping it would also delete
        # unrelated numbers (e.g. the per-entry page numbers on the TOC page).
        # Require at least 3 real characters.
        return len(re.sub(r"[#\s/\-.]", "", norm)) < 3

    per_page_lines = [rec["text"].split("\n") for rec in records]
    line_page_counts = Counter()
    for lines in per_page_lines:
        line_page_counts.update({normalize(l) for l in lines if normalize(l)})

    n_pages = len(records)
    boilerplate = {
        norm for norm, count in line_page_counts.items()
        if count / n_pages >= min_repeat_ratio and not too_generic(norm)
    }

    cleaned = []
    for rec, lines in zip(records, per_page_lines):
        new_rec = dict(rec)
        new_rec["text"] = "\n".join(l for l in lines if normalize(l) not in boilerplate)
        cleaned.append(new_rec)

    if boilerplate:
        print(f"  Stripped {len(boilerplate)} repeated header/footer line(s) "
              f"from {records[0]['source']} "
              f"(seen on >= {int(min_repeat_ratio * 100)}% of its pages).")
    return cleaned


# Common short Arabic function words paired with the "letters missing" form
# produced by broken font shaping. Space-padded to match whole words only.
_CORRUPTION_PROBE_PAIRS = [
    (" من ", " ن "),
    (" على ", " عل "),
    (" التي ", " التم "),
]


def estimate_text_corruption(full_text: str) -> float:
    """
    Rough 0-1 corruption score: the fraction of occurrences of a few common
    Arabic function words that came out malformed. It only flags a document
    worth a closer look -- it cannot tell an extraction bug from a defect
    baked into the PDF, and cannot fix anything.
    """
    padded = f" {full_text} "
    correct = sum(padded.count(c) for c, _ in _CORRUPTION_PROBE_PAIRS)
    broken = sum(padded.count(b) for _, b in _CORRUPTION_PROBE_PAIRS)
    total = correct + broken
    return broken / total if total else 0.0


def load_pdfs_from_folder(folder: str) -> list:
    """
    Loads every PDF in `folder` and returns page records:
    {"source": filename, "page": page_num, "text": text}.
    Boilerplate is stripped per document and corrupted-looking sources are
    reported with a warning.
    """
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder}'. Put your PDFs there and re-run."
        )

    records = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        print(f"Reading {filename} ...")
        doc_records = [
            {"source": filename, "page": page_num, "text": text}
            for page_num, text in extract_pages(path)
        ]
        print(f"  -> extracted {len(doc_records)} pages with text")

        doc_records = strip_repeated_boilerplate(doc_records)

        score = estimate_text_corruption("\n".join(r["text"] for r in doc_records))
        if score >= config.TEXT_CORRUPTION_WARN_RATIO:
            print(f"  WARNING: {filename} looks like it has corrupted source text "
                  f"({score:.0%} of sampled function words came out malformed). "
                  f"This is usually baked into the PDF's glyphs, so no re-extraction "
                  f"fixes it. Domain nouns are normally unaffected, but quoted prose "
                  f"may contain garbled connective words. Get a cleaner export if possible.")

        records.extend(doc_records)
    return records


# ---------------------------------------------------------------------------
# 3. Section detection (table of contents)
# ---------------------------------------------------------------------------
# OCR output keeps LOGICAL order: "<item>. <heading> ....dots.... <page>"
# (or a Markdown table row "| <item>. <heading> | <page> |").
_TOC_ENTRY_RE_LOGICAL = re.compile(
    r"(?<!\d)\d{1,2}[)\]]?\.\s*([^\d|.]{3,150}?)\s*(?:\.{3,}|\u2026+|\|)\s*(\d{1,3})(?!\d)"
)


def extract_toc_sections(records: list) -> list:
    """
    Scans ONE document's page records for TOC-style entries and returns an
    ordered list of (start_page, heading).

    Extraction often splits a visual TOC line across several lines, so
    whitespace is collapsed to single spaces before matching.
    """
    entries = []
    for rec in records:
        normalized = re.sub(r"\s+", " ", rec["text"]).strip()
        # Fix a stray space between an item number and its period/bracket
        # (RTL extraction artifact): "4 .heading" -> "4.heading".
        normalized = re.sub(r"(\d)\s+([).\]])", r"\1\2", normalized)

        for m in _TOC_ENTRY_RE_LOGICAL.finditer(normalized):
            heading, page_str = m.group(1).strip(), m.group(2)
            if heading and int(page_str) > 0:
                entries.append((int(page_str), heading))

    # Keep the first entry per page, ordered by page.
    seen, unique = set(), []
    for page_num, heading in entries:
        if page_num not in seen:
            seen.add(page_num)
            unique.append((page_num, heading))
    unique.sort(key=lambda x: x[0])

    if not unique and records:
        print(f"  NOTE: no table-of-contents entries detected for {records[0]['source']}. "
              f"Section tagging is disabled for this document (section=None).")
    return unique


def section_for_page(toc_entries: list, page_num: int):
    """Heading of the last TOC entry starting on or before `page_num`, or None."""
    current = None
    for start_page, heading in toc_entries:
        if start_page > page_num:
            break
        current = heading
    return current


def tag_sections(page_records: list) -> list:
    """Returns page records with a "section" key added, detected per document."""
    by_source = defaultdict(list)
    for rec in page_records:
        by_source[rec["source"]].append(rec)

    tagged = []
    for source, recs in by_source.items():
        toc_entries = extract_toc_sections(recs)
        if toc_entries:
            print(f"  {source}: found {len(toc_entries)} section heading(s)")
            for pg, hd in toc_entries:
                print(f"      page {pg:>3} -> {hd[:70]}")
        for rec in recs:
            tagged.append({**rec, "section": section_for_page(toc_entries, rec["page"])})
    return tagged


# ---------------------------------------------------------------------------
# 4. Chunking
# ---------------------------------------------------------------------------
# Sentence enders: Latin . ! ? and Arabic ؟ ۔ , plus newlines.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n+")
# Dot-leaders ("......... 8") mark table-of-contents pages, not prose.
_DOT_LEADER_RE = re.compile(r"\.{4,}")

# Procedure tables have a recurring row-label column. PyMuPDF puts the label
# and its value on separate lines, so a chunk boundary could separate them
# and leave "المنفذ" with no indication of who. Kept as a short explicit list
# to avoid gluing unrelated short lines together.
_ROLE_LABEL_LINES = {
    "الجهة املنفذة", "الجهة المنفذة",  # "executing department" (with/without PDF artifact)
    "املنفذ", "المنفذ",                  # "executor"
}


def _glue_role_labels(text: str) -> str:
    """Joins a label-only line to the next non-empty line ("label: value")."""
    lines = text.split("\n")
    glued, i = [], 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in _ROLE_LABEL_LINES:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                glued.append(f"{stripped}: {lines[j].strip()}")
                i = j + 1
                continue
        glued.append(lines[i])
        i += 1
    return "\n".join(glued)


def _looks_like_toc(text: str, min_dot_runs: int = 2) -> bool:
    return len(_DOT_LEADER_RE.findall(text)) >= min_dot_runs


def split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_END_RE.split(text) if p and p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Packs sentences into ~`chunk_size`-character chunks, carrying `overlap`
    characters of trailing sentences into the next chunk.
    """
    sentences = split_sentences(_glue_role_labels(text))
    if not sentences:
        return []

    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))

            overlap_sentences, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current, current_len = overlap_sentences, overlap_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_records(records: list, chunk_size: int, overlap: int) -> list:
    """
    Page records (with optional "section") -> chunk records
    {"source", "page", "chunk_id", "section", "text"}.

    The section heading is prepended to each chunk's text so both the dense
    embedding and BM25 see the topic. TOC-like chunks are skipped so they
    never surface as false-positive hits.
    """
    chunked, skipped_toc = [], 0
    for rec in records:
        section = rec.get("section")
        for i, piece in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            if _looks_like_toc(piece):
                skipped_toc += 1
                continue
            chunked.append({
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": i,
                "section": section,
                "text": f"{section}\n{piece}" if section else piece,
            })
    if skipped_toc:
        print(f"Skipped {skipped_toc} table-of-contents-like chunk(s) during indexing.")
    return chunked


# ---------------------------------------------------------------------------
# 5. Embedding + indexing
# ---------------------------------------------------------------------------


def build_chunks(folder: str = None) -> list:
    """PDFs in `folder` -> chunk records (stages 1-4)."""
    folder = folder or config.DATA_DIR

    print(f"Loading PDFs from {folder} ...")
    pages = load_pdfs_from_folder(folder)
    print(f"Loaded {len(pages)} pages total.\n")

    print("Detecting section headings from each document's table of contents ...")
    pages = tag_sections(pages)
    print()

    print("Chunking text ...")
    chunks = chunk_records(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Produced {len(chunks)} chunks.\n")
    return chunks


def index_chunks(chunks: list):
    """Embeds chunks and (re)creates the Chroma collection (stage 5)."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    print(f"Loading embedding model '{config.EMBED_MODEL_NAME}' "
          f"(first run downloads it, ~2.2GB)...")
    model = SentenceTransformer(config.EMBED_MODEL_NAME)

    print("Embedding chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True
    )

    print("\nWriting to Chroma ...")
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    ids = [f"{c['source']}::p{c['page']}::c{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "section": c["section"] or ""}
        for c in chunks
    ]

    batch = 200
    for i in tqdm(range(0, len(ids), batch)):
        collection.add(
            ids=ids[i:i + batch],
            embeddings=embeddings[i:i + batch].tolist(),
            documents=texts[i:i + batch],
            metadatas=metadatas[i:i + batch],
        )

    print(f"\nDone. Indexed {len(ids)} chunks into '{config.COLLECTION_NAME}' "
          f"at {config.CHROMA_DIR}")


def main():
    chunks = build_chunks()
    if not chunks:
        print("No chunks produced — check that your PDFs contain extractable text "
              "(scanned/image-only PDFs need OCR first).")
        return
    index_chunks(chunks)
    print("You can now run: python chat.py")


if __name__ == "__main__":
    main()
