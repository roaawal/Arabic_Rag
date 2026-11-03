"""
Ingestion pipeline: PDFs -> cleaned pages -> chunks -> embeddings -> ChromaDB.

Run once, and again any time the PDFs in data/ change:

    python ingestion.py

Stages (all in this file):
  1. Extract   -- Gemini vision OCR of each page image (ocr_backends.py),
                  cached in ocr_cache/.
  2. Check     -- warn if the source text looks corrupted. The OCR prompt
                  already skips running headers/footers.
  3. Structure -- read procedure titles, executing departments, executors and
                  tables from the OCR Markdown.
  4. Chunk     -- pack lines within one procedure on one page, repeating the
                  title, department and executor so no chunk is orphaned.
  5. Index     -- embed with BAAI/bge-m3 and rebuild the Chroma collection.
"""
import glob
import os
import re
from collections import defaultdict

import config

# ---------------------------------------------------------------------------
# 1. Extraction
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: str):
    """Yields (page_number, text) for one PDF; page_number is 1-indexed."""
    import ocr_backends
    yield from ocr_backends.extract_pages_cached(pdf_path)


# ---------------------------------------------------------------------------
# 2. Source-text check
# ---------------------------------------------------------------------------


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
    Corrupted-looking sources are reported with a warning.
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
# 3. Structure (OCR Markdown -> lines tagged with their procedure context)
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#\s*(.*)$")
_SUBHEADING_RE = re.compile(r"^###\s*(.+)$")
_EXECUTOR_RE = re.compile(r"^##\s*المنفذ\s*:?\s*(.*)$")
_DEPARTMENT_RE = re.compile(r"^الجهة المنفذة\s*:?\s*(.*)$")
# The manual's own name, sometimes transcribed as a heading on continuation
# pages. It must not replace the procedure the page belongs to.
_DOC_TITLE_RE = re.compile(r"^(دليل|إجراءات عمل وحدة)")
# Some titles lose their "#" in OCR: a short numbered line opening the page.
_PLAIN_TITLE_RE = re.compile(r"^\d{1,2}\.\s*.{3,120}:$")
_GENERAL_HEADINGS = {
    "مقدمة", "القواعد والأحكام العامة", "الملاحق", "التقارير", "النماذج",
    "جدول التعديلات", "الموافقات",
}
_IGNORED_LINES = {"الإجراءات", "الاجراءات", "المحتويات"}
_TABLE_RULE_RE = re.compile(r"^\s*\|?[\s:|\-]+\|?\s*$")
# Dot-leaders ("......... 8") mark table-of-contents pages, not content.
_DOT_LEADER_RE = re.compile(r"\.{4,}")


def _looks_like_toc(text: str, min_dot_runs: int = 2) -> bool:
    return len(_DOT_LEADER_RE.findall(text)) >= min_dot_runs


def _clean_inline(line: str) -> str:
    line = re.sub(r"<br\s*/?>", " ", line, flags=re.I)
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"\*\*|__|`", "", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def _table_cells(line: str) -> list:
    return [_clean_inline(c) for c in line.strip().strip("|").split("|")]


def _table_row_text(headers: list, cells: list) -> str:
    """One table row as "column: value" pairs, so a row still says what each
    value means once it is separated from the table's header."""
    pairs = []
    for i, cell in enumerate(cells):
        if not cell:
            continue
        header = headers[i] if i < len(headers) else ""
        pairs.append(f"{header}: {cell}" if header else cell)
    return " | ".join(pairs)


def _new_context() -> dict:
    return {"procedure": None, "sub": None, "department": None, "executor": None}


def _title_of(ctx: dict):
    parts = [p for p in (ctx["procedure"], ctx["sub"]) if p]
    return " - ".join(parts) if parts else None


def _open_heading(ctx: dict, heading: str):
    """Applies a heading to the running context. A numbered title or a general
    heading ("مقدمة:") starts a new part of the manual; anything else, such as
    "ثانياً. ...", is a part of the current procedure."""
    if re.match(r"\d", heading) or heading.rstrip(":").strip() in _GENERAL_HEADINGS:
        ctx.update(procedure=heading, sub=None)
    else:
        ctx["sub"] = heading
    ctx.update(department=None, executor=None)


def parse_page_lines(text: str, ctx: dict) -> list:
    """
    One page of OCR Markdown -> content lines, each tagged with the title,
    department and executor in force at that line. `ctx` is carried from page
    to page because a procedure usually continues over several pages.
    Each item is {"text", "title", "department", "executor", "attached"};
    `attached` marks sub-items and notes that belong to the line above.
    """
    items = []
    lines = text.split("\n")
    first_content = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        is_first, first_content = first_content, False

        if line.startswith("|"):
            has_header = i < len(lines) and _TABLE_RULE_RE.match(lines[i]) and "-" in lines[i]
            headers = _table_cells(line) if has_header else []
            if has_header:
                i += 1
            rows = [] if has_header else [_table_cells(line)]
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not _TABLE_RULE_RE.match(lines[i]):
                    rows.append(_table_cells(lines[i]))
                i += 1
            if has_header and not rows:
                rows, headers = [headers], []
            for cells in rows:
                row = _table_row_text(headers, cells)
                if row:
                    items.append({"text": row, "title": _title_of(ctx),
                                  "department": ctx["department"],
                                  "executor": ctx["executor"], "attached": False})
            continue

        match = _SUBHEADING_RE.match(line)
        if match:
            ctx["sub"] = _clean_inline(match.group(1))
            continue
        match = _EXECUTOR_RE.match(line)
        if match:
            ctx["executor"] = _clean_inline(match.group(1)) or ctx["executor"]
            continue
        match = _HEADING_RE.match(line) if not line.startswith("##") else None
        if match:
            heading = _clean_inline(match.group(1))
            if heading and not _DOC_TITLE_RE.match(heading):
                _open_heading(ctx, heading)
            continue
        match = _DEPARTMENT_RE.match(line)
        if match:
            ctx["department"] = _clean_inline(match.group(1)) or ctx["department"]
            continue

        line = _clean_inline(re.sub(r"^#{1,6}\s*", "", line))
        bare = line.rstrip(":").strip()
        if bare in _IGNORED_LINES:
            continue
        if bare in _GENERAL_HEADINGS or (is_first and _PLAIN_TITLE_RE.match(line)):
            _open_heading(ctx, line)
            continue
        items.append({
            "text": line,
            "title": _title_of(ctx),
            "department": ctx["department"],
            "executor": ctx["executor"],
            "attached": line.startswith(("-", "•", "ملاحظة")),
        })
    return items


# ---------------------------------------------------------------------------
# 4. Chunking
# ---------------------------------------------------------------------------


def _render_chunk(group: list) -> str:
    """Context header, then the lines; a change of executor stays visible."""
    first = group[0]
    out = []
    if first["title"]:
        out.append(first["title"])
    if first["department"]:
        out.append(f"الجهة المنفذة: {first['department']}")
    executor = first["executor"]
    if executor:
        out.append(f"المنفذ: {executor}")
    for item in group:
        if item["executor"] != executor and item["executor"]:
            executor = item["executor"]
            out.append(f"المنفذ: {executor}")
        out.append(item["text"])
    return "\n".join(out)


def _pack_lines(items: list, chunk_size: int, overlap: int) -> list:
    """
    Packs lines into ~`chunk_size`-character groups with `overlap` characters of
    trailing lines repeated. An attached line may overflow by half a chunk
    rather than be separated from the step it belongs to.
    """
    groups, current, current_len = [], [], 0
    for item in items:
        size = len(item["text"])
        limit = chunk_size * 1.5 if item["attached"] else chunk_size
        if current and current_len + size > limit:
            groups.append(current)
            kept, kept_len = [], 0
            for prev in reversed(current):
                if kept_len + len(prev["text"]) > overlap:
                    break
                kept.insert(0, prev)
                kept_len += len(prev["text"])
            current, current_len = kept, kept_len
        current.append(item)
        current_len += size
    if current:
        groups.append(current)
    return groups


def chunk_records(records: list, chunk_size: int, overlap: int) -> list:
    """
    Page records -> chunk records {"source", "page", "chunk_id", "section", "text"}.

    Chunks never cross a procedure or a page, so a citation always points at the
    right page, and each chunk repeats its procedure context, so both the dense
    embedding and BM25 see the topic. Table-of-contents pages are skipped.
    """
    chunked, skipped_toc = [], 0
    ctx, ctx_source = _new_context(), None
    for rec in records:
        if rec["source"] != ctx_source:
            ctx, ctx_source = _new_context(), rec["source"]
        if _looks_like_toc(rec["text"]):
            skipped_toc += 1
            continue

        by_title, order = defaultdict(list), []
        for item in parse_page_lines(rec["text"], ctx):
            if item["title"] not in by_title:
                order.append(item["title"])
            by_title[item["title"]].append(item)

        chunk_id = 0
        for title in order:
            for group in _pack_lines(by_title[title], chunk_size, overlap):
                chunked.append({
                    "source": rec["source"],
                    "page": rec["page"],
                    "chunk_id": chunk_id,
                    "section": title,
                    "text": _render_chunk(group),
                })
                chunk_id += 1
    if skipped_toc:
        print(f"Skipped {skipped_toc} table-of-contents page(s) during indexing.")
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
