"""
Central configuration for the Arabic-friendly RAG pipeline.
Tweak these values instead of hunting through the other files.
"""
import datetime
import os

# --- OCR (Gemini vision; see ocr_backends.py) ---
OCR_GEMINI_MODEL = "fast"   # "fast" = GEMINI_FAST_MODEL below
OCR_DPI = 200
OCR_STRIP_MARKDOWN = True
OCR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "ocr_cache")
try:
    from dotenv import load_dotenv
    load_dotenv()          # reads .env from the project folder
except ImportError:
    pass

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

# --- Embedding model ---
EMBED_MODEL_NAME = "BAAI/bge-m3"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 4
COLLECTION_NAME = "arabic_pdf_docs"

# --- Hybrid retrieval (dense + BM25 keyword, fused via Reciprocal Rank
# Fusion) ---
# Procedure manuals lean heavily on exact terms (form names, codes,
# section headings) that pure semantic embeddings can under-weight --
# BM25 catches exact term overlap, dense catches paraphrases/synonyms.
# Combining both via RRF improves recall for this kind of document.
# No re-indexing required: BM25 is built from the same chunk text
# already stored in Chroma by ingestion.py.
USE_HYBRID_RETRIEVAL = True
HYBRID_CANDIDATE_POOL = 20   # candidates each method contributes before fusion
RRF_K = 60                   # standard reciprocal-rank-fusion constant

# --- PDF cleanup (added after inspecting real scanned manuals) ---
# Corporate Arabic manuals like the Housing Bank procedures manual repeat
# an identical version/logo table (رقم النسخة / تاريخ الإصدار / اسم
# الاجراء ...) as extracted TEXT on every single page. Left alone, that
# boilerplate gets embedded into every chunk and dilutes both BM25 and
# dense similarity (every chunk starts to look alike). ingestion.py
# detects any line that repeats on at least this fraction of a document's
# pages and strips it before chunking. 0.5 caught 14/15 pages of the
# sample manual's header without touching real content (which naturally
# varies page to page).
HEADER_FOOTER_MIN_REPEAT_RATIO = 0.5

# --- Source-text quality check ---
# Some PDFs (confirmed on the Housing Bank sample: cross-checked with
# PyMuPDF, poppler's pdftotext, AND Tesseract OCR on the rendered page
# image -- all three agree) have common short Arabic words (من، على،
# التي) missing letters directly in the document's rendered glyphs, not
# as an extraction-tool bug. No extraction method fixes this -- it's
# baked into the source file. ingestion.py can only flag it, not
# repair it: if you can get a cleaner re-export of the original file,
# do that; otherwise this is a known, accepted limitation (it mainly
# hits short connective words, not the domain nouns you actually search
# and retrieve on).
# If the ratio of "broken-looking" to "well-formed" instances of a set
# of common function words exceeds this, ingestion.py prints a
# warning naming the affected source file.
TEXT_CORRUPTION_WARN_RATIO = 0.3

# --- LLM backend: Google Gemini API (OpenAI-compatible endpoint) ---
# Every LLM call (router, helper stages, generation, judge) goes to the
# Gemini API. Set your key in the environment before running:
#   PowerShell:  $env:GEMINI_API_KEY = "your-key"
# Use a PAID-tier key for internal documents: on the free tier Google may
# use prompts and responses to improve its products; on the paid tier it
# does not (https://ai.google.dev/gemini-api/terms).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# --- Model choice per role (prices checked 2026-10-20, see MODEL_PRICING) ---
# Router + helper stages: cheap, fast Flash-Lite tier. These are short
# classification/JSON/scoring calls, so a bigger model buys little.
GEMINI_FAST_MODEL = "gemini-3.5-flash-lite"
# Final answer + judge: Flash tier. gemini-3.8-flash is Google's newest
# Flash and costs the same as gemini-3.6-flash, so there is no reason to
# pick the older one. Pro/Argon models cost 2.7x+ more per token and
# always think, which makes every question slower and pricier.
GEMINI_MAIN_MODEL = "gemini-3.8-flash"

GENERATION_MODEL = GEMINI_MAIN_MODEL     # final grounded/direct answer
ROUTER_MODEL = GEMINI_FAST_MODEL         # route classification
REWRITER_MODEL = GEMINI_FAST_MODEL       # query rewriting
MULTI_QUERY_MODEL = GEMINI_FAST_MODEL    # multi-query generation
DECOMPOSITION_MODEL = GEMINI_FAST_MODEL  # question decomposition
HYDE_MODEL = GEMINI_FAST_MODEL           # hypothetical-document generation
RERANKER_MODEL = GEMINI_FAST_MODEL       # LLM-based reranking
COMPRESSION_MODEL = GEMINI_FAST_MODEL    # contextual compression
CRAG_MODEL = GEMINI_FAST_MODEL           # CRAG relevance grading
JUDGE_MODEL = GEMINI_MAIN_MODEL          # LLM-as-judge evaluation

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.2

# --- Thinking control ---
# Gemini 3 models always think, and thinking tokens are billed as output
# tokens. Lower effort = cheaper and faster. Per-stage override by stage
# name (the names used in generation.call_completion); anything not listed
# uses DEFAULT_REASONING_EFFORT. Valid: "minimal", "low", "medium", "high".
DEFAULT_REASONING_EFFORT = "low"
REASONING_EFFORT = {
    "router": "minimal",
    "rewriter": "minimal",
    "multi_query": "minimal",
    "decomposition": "minimal",
    "hyde": "minimal",
    "reranking": "minimal",
    "contextual_compression": "minimal",
    "crag_evaluator": "minimal",
    "direct_generation": "low",
    "final_generation": "low",
    "evaluation_judge": "low",
}
# max_tokens on the Gemini API also covers thinking tokens. This headroom
# is ADDED to every call's max_tokens so thinking can't squeeze the visible
# answer to nothing. You only pay for tokens actually generated.
THINKING_TOKEN_HEADROOM = 1024

# Retries for transient API errors (rate limit 429, server 5xx).
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# --- Cost tracking ---
# Standard paid-tier prices in USD per 1M tokens, from
# https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-20).
# Each entry is a list of (last_day_inclusive_ISO_date_or_None, input,
# output) rows; the first row whose date has not passed applies, so
# announced price changes take effect automatically on the right day.
# Output price includes thinking tokens. Update this table when Google
# changes prices or you switch models.
MODEL_PRICING = {
    # Flash-Lite tier (flat price)
    "gemini-3.1-flash-lite": [(None, 0.25, 1.50)],
    "gemini-3.5-flash-lite": [(None, 0.30, 2.50)],
    # Flash tier: introductory price until 2026-12-31, doubles on 2027-01-01
    "gemini-3.6-flash": [("2026-12-31", 0.75, 3.75), (None, 1.50, 7.50)],
    "gemini-3.8-flash": [("2026-12-31", 0.75, 3.75), (None, 1.50, 7.50)],
    # Pro / flagship (prompts up to 200k tokens; this pipeline stays far below)
    "gemini-3.1-pro-preview": [(None, 2.00, 12.00)],
    "gemini-4-argon": [("2026-11-30", 2.00, 10.00), (None, 4.00, 20.00)],
}


def price_per_million(model: str, on: datetime.date = None):
    """(input_usd, output_usd) per 1M tokens for `model` on date `on`
    (default: today), or None if the model has no entry in MODEL_PRICING."""
    rows = MODEL_PRICING.get(model)
    if not rows:
        return None
    on = on or datetime.date.today()
    for last_day, price_in, price_out in rows:
        if last_day is None or on <= datetime.date.fromisoformat(last_day):
            return price_in, price_out
    return rows[-1][1], rows[-1][2]


# --- LLM-as-judge evaluation ---
# Which model answers the judge prompts: see JUDGE_MODEL above.
JUDGE_MAX_NEW_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0  # deterministic scoring, not creative generation

# --- Router ---
# Classifies each incoming question into a route BEFORE any retrieval
# happens. Model: see ROUTER_MODEL above.
ROUTER_MAX_NEW_TOKENS = 512
ROUTER_TEMPERATURE = 0.0
VALID_ROUTES = {"direct", "rag", "advanced_rag"}
# The router only ever SELECTS from these four query-understanding
# techniques. self_query is intentionally not in this list: it isn't a
# question of "should we try a different phrasing", it's a metadata
# filter, so it's driven directly by the router's `has_metadata_constraints`
# boolean instead (see router.py / advanced_rag.py). reranking,
# contextual_compression, and crag are mandatory stages of the advanced
# route, not techniques the router chooses -- they're recorded in
# MANDATORY_ADVANCED_STAGES for logging/`techniques_used` purposes only.
VALID_TECHNIQUES = {"rewriting", "multi_query", "decomposition", "hyde"}
MANDATORY_ADVANCED_STAGES = {"reranking", "contextual_compression", "crag"}
VALID_COMPLEXITY = {"simple", "complex"}

# --- Advanced RAG: query understanding & transformation ---
MULTI_QUERY_N = 3            # how many paraphrased query variants to generate
DECOMPOSITION_MAX_SUBQS = 4  # cap on sub-questions from decomposition
HYDE_MAX_NEW_TOKENS = 250    # hypothetical-answer length is meant to be short

# --- Advanced RAG: retrieval improvement ---
RERANK_CANDIDATE_POOL = 15   # how many fused hits go INTO the LLM reranker
RERANK_KEEP_TOP_K = 5        # how many the reranker keeps for generation
CRAG_MAX_CORRECTIONS = 1     # how many corrective re-retrieval attempts CRAG gets
CRAG_CORRECTED_POOL_MULTIPLIER = 2  # widen HYBRID_CANDIDATE_POOL by this factor on correction
