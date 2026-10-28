"""
PDF page -> text extraction for Arabic manuals with Gemini vision
(page image -> Markdown, OpenAI-compatible endpoint).

The PDFs' embedded text layers are unusable (broken glyph mappings), so every
page is rendered to an image and transcribed. Gemini Flash-Lite won the
bake-off against Docling (RapidOCR / EasyOCR) and PaddleOCR-VL on accuracy at
a fraction of a cent per page; see ocr_compare.py to re-test other models.

Used by:
  * ingestion.py     -> extract_pages_cached(pdf_path) with a disk cache
  * ocr_compare.py   -> extract(pdf_path, page_no, model) for model bake-offs
"""
import base64
import hashlib
import os
import re
import time

import config

OCR_DPI = int(os.environ.get("OCR_DPI") or config.OCR_DPI)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fitz():
    try:
        import pymupdf as fitz  # newer name
    except ImportError:
        import fitz
    return fitz


def render_page_png(pdf_path: str, page_no: int, dpi: int = None) -> bytes:
    fitz = _fitz()
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi or OCR_DPI)
        return pix.tobytes("png")
    finally:
        doc.close()


def page_count(pdf_path: str) -> int:
    fitz = _fitz()
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n|\n\s*```\s*$")


def strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def markdown_to_plain(md: str) -> str:
    """Flatten Markdown to the plain line-oriented text the existing chunker
    was written for: drop table rules and pipes, heading marks, bold, html."""
    out = []
    for line in md.splitlines():
        if re.fullmatch(r"\s*\|?[\s:|\-]+\|?\s*", line) and "-" in line:
            continue  # table separator row
        line = re.sub(r"<br\s*/?>", " ", line, flags=re.I)
        line = re.sub(r"<[^>]+>", " ", line)
        line = line.replace("|", " ")
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"\*\*|__|`", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Output sanity check (catches vision-model glitches such as Cyrillic look-alikes)
# ---------------------------------------------------------------------------

def _script(ch: str) -> str:
    import unicodedata
    if not ch.isalpha():
        return ""
    name = unicodedata.name(ch, "")
    if name.startswith("ARABIC"):
        return "arabic"
    if name.startswith("LATIN"):
        return "latin"
    return name.split(" ")[0].lower() or "other"


def suspicious_words(text: str):
    """Words that mix Arabic with another script, or that are written in a
    script that has no business in these manuals (Cyrillic, Greek, ...).
    A non-empty result means: look at that page."""
    flagged = []
    for w in re.findall(r"\w+", text):
        scripts = {_script(c) for c in w} - {""}
        if not scripts:
            continue
        if "arabic" in scripts and len(scripts) > 1:
            flagged.append(w)
        elif not scripts <= {"arabic", "latin"}:
            flagged.append(w)
    return flagged


# Look-alike Cyrillic letters the vision model sometimes writes inside Arabic
# words (seen: "المطالعات" returned as "المطалعات": а = ا, л = ل). Repaired
# only when EVERY foreign letter in the word is in this map, so a genuinely
# foreign word is never rewritten.
_HOMOGLYPHS = {"а": "ا", "л": "ل"}


def repair_homoglyphs(text: str) -> str:
    def fix(m):
        w = m.group(0)
        scripts = {_script(c) for c in w} - {""}
        if "arabic" not in scripts or scripts <= {"arabic", "latin"}:
            return w
        foreign = {c for c in w if _script(c) not in ("arabic", "latin", "")}
        if foreign and foreign <= set(_HOMOGLYPHS):
            return "".join(_HOMOGLYPHS.get(c, c) for c in w)
        return w

    return re.sub(r"\w+", fix, text)


# ---------------------------------------------------------------------------
# Gemini vision
# ---------------------------------------------------------------------------

# Prompt tuned to this manual family (header table, procedure title row,
# "الجهة المنفذة" row, "المنفذ" role rows, numbered steps). It avoids Markdown
# tables for procedure pages, so there is no column-order ambiguity, and it
# drops the repeated header/footer so they never reach the index.
GEMINI_OCR_PROMPT = """You are a precise OCR engine for Arabic procedure manuals.
Transcribe the attached page image to Markdown, keeping the manual's structure.

Skip the running page header (logo, document name, version/date table) and the page
footer (Internal Use, page number, department name). Transcribe everything else.

If the page is a procedure page (rows labelled "المنفذ" with numbered steps), use this
format exactly, in reading order from top to bottom:
# <procedure number and title>
الجهة المنفذة: <text>
## المنفذ: <role or person named in that row>
1- <step text, complete, on one line>
- <sub-item of the step above>
ملاحظة: <note text>
Start a new "## المنفذ:" line each time the executor changes.
Keep section lines such as "أولاً: ..." as "### أولاً: ...".

For any other page (forms, reports, accounting entries, lists) transcribe tables as
Markdown tables with the label or first column FIRST in every row, and keep other text
as plain lines.

Rules:
- Copy the text exactly as printed. Do not translate, summarize, correct spelling, add
  hamza, or add diacritics that are not printed.
- Keep list numbering exactly as printed (1-, 2-, ...), and keep bullets.
- Every Arabic word must be written with Arabic letters only. Never substitute
  look-alike Latin or Cyrillic letters.
- Keep Latin text, digits and dates exactly as printed.
- Output only the Markdown. No commentary, no code fences."""


def gemini_model() -> str:
    """OCR_GEMINI_MODEL env var wins; the value 'fast' means config.GEMINI_FAST_MODEL."""
    m = os.environ.get("OCR_GEMINI_MODEL") or config.OCR_GEMINI_MODEL
    return config.GEMINI_FAST_MODEL if m == "fast" else m


def extract(pdf_path: str, page_no: int, model: str = None) -> str:
    """Transcribes one page (1-indexed) with Gemini; returns Markdown."""
    import requests

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in this terminal.")
    model = model or gemini_model()
    timeout = int(os.environ.get("OCR_TIMEOUT", "120"))
    effort = os.environ.get("OCR_REASONING", "minimal")  # "" = do not send
    debug = bool(os.environ.get("OCR_DEBUG"))
    b64 = base64.b64encode(render_page_png(pdf_path, page_no)).decode("ascii")
    payload = {
        "model": model,
        "max_tokens": 16000,  # also covers thinking tokens on Gemini
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": GEMINI_OCR_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }
    if effort:
        payload["reasoning_effort"] = effort
    url = f"{config.GEMINI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.GEMINI_API_KEY}",
               "Content-Type": "application/json"}

    retries = config.MAX_RETRIES
    backoff = config.RETRY_BACKOFF_SECONDS
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last = f"{type(e).__name__} after {timeout}s (model={model})"
            print(f"      [gemini] attempt {attempt + 1} failed: {last}")
            time.sleep(backoff * (attempt + 1))
            continue
        if (r.status_code == 400 and "reasoning_effort" in payload
                and any(w in r.text.lower() for w in ("reasoning", "thinking"))):
            payload.pop("reasoning_effort")
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(backoff * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
        body = r.json()
        if debug:
            print(f"      [gemini] model={model} usage={body.get('usage')}")
        choice = body["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty text (thinking may have used "
                               "the whole token budget).")
        if choice.get("finish_reason") == "length":
            raise RuntimeError("Gemini output was truncated (finish_reason=length).")
        return repair_homoglyphs(strip_code_fences(text))
    raise RuntimeError(f"Gemini failed after retries: {last}")


# ---------------------------------------------------------------------------
# Cached page iterator for ingestion
# ---------------------------------------------------------------------------

def cache_key() -> str:
    """Cache folder name; includes the model so changing it re-runs OCR.
    The "-structured" suffix names the prompt; it is kept so pages OCR'd
    before the other prompt was removed are still found in the cache."""
    return "gemini-" + gemini_model() + "-structured"


def _file_sha(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def extract_pages_cached(pdf_path: str, pages=None):
    """Yields (page_no, text) like ingestion.extract_pages. Results are cached
    on disk per (pdf content, model), so re-running ingestion does not
    re-OCR unchanged PDFs. Delete ocr_cache/ to force a fresh run."""
    cache_dir = os.path.join(config.OCR_CACHE_DIR, _file_sha(pdf_path), cache_key())
    os.makedirs(cache_dir, exist_ok=True)
    wanted = list(pages) if pages else range(1, page_count(pdf_path) + 1)
    for page_no in wanted:
        cache_file = os.path.join(cache_dir, f"page_{page_no:03d}.md")
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"    [gemini] page {page_no} ...")
            text = extract(pdf_path, page_no)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
        text = repair_homoglyphs(text)   # also fixes pages cached before the repair existed
        bad = suspicious_words(text)
        if bad:
            print(f"    WARNING page {page_no}: suspicious words {bad[:8]} - "
                  f"check ocr_cache/.../page_{page_no:03d}.md against the page image")
        if config.OCR_STRIP_MARKDOWN:
            text = markdown_to_plain(text)
        if text and text.strip():
            yield page_no, text
