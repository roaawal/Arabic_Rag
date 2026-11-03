# Arabic Procedure Manuals RAG

Arabic document question-answering system using Retrieval-Augmented Generation (RAG).
The project reads internal procedure manuals with Gemini vision OCR, indexes them locally,
retrieves relevant passages, and generates answers grounded in those passages. It supports
both a command-line interface and a Streamlit web interface.

## What the Project Does

The system answers questions about PDF procedure manuals covering:

- Central alarm and security procedures
- Fixed assets and warehouse operations
- Central mail and files procedures

The answer pipeline is designed to reduce hallucination by retrieving document evidence
before generation, showing source pages and chunk IDs, and explicitly reporting when the
retrieved context is insufficient.

## Architecture

```text
PDF files
   |
   v
Gemini vision OCR of each page image (cached in ocr_cache/)
   |
   v
Procedure structure read from the OCR Markdown
   |
   v
Chunking by procedure and page
   |
   v
Embeddings stored in ChromaDB
   |
User question
   |
   v
Router: direct / rag / advanced_rag / clarify
   |
   +--> Direct reply (greetings; declines out-of-scope questions)
   |
   +--> Clarifying question listing the procedures found
   |
   +--> Hybrid retrieval: dense search + BM25 + RRF
   |
   +--> Advanced retrieval: rewriting, decomposition, reranking
   |
   v
Grounded answer generation

Separately: evaluate_testset.py runs a test set and the LLM judge
```

## Project Files

| File | Purpose |
|---|---|
| `chat.py` | Command-line and Streamlit user interface |
| `pipeline.py` | Main entry point and route orchestration |
| `config.py` | Central configuration for models, OCR, retrieval, and evaluation |
| `ingestion.py` | Full ingestion pipeline: OCR, procedure structure, chunking, embedding, and ChromaDB indexing |
| `ocr_backends.py` | Gemini vision OCR of PDF pages, with an on-disk cache |
| `ocr_compare.py` | Benchmark for comparing OCR models against hand-typed ground truth |
| `rag.py` | Dense retrieval, BM25 retrieval, and Reciprocal Rank Fusion |
| `router.py` | Classifies questions and selects the processing route |
| `advanced_rag.py` | Runs the advanced RAG pipeline |
| `query_transform.py` | Rewriting, multi-query, decomposition, HyDE, and metadata filters |
| `retrieval_improve.py` | Reranking, contextual compression, and CRAG |
| `generation.py` | Gemini API calls, prompts, token/cost tracking, and latency tracking |
| `evaluate.py` | LLM-as-judge evaluation |
| `evaluate_testset.py` | Runs a test set: page recall, routes, and judged correctness |
| `data/` | Source PDF manuals |
| `ground_truth/` | Git-ignored: hand-typed pages for `ocr_compare.py`, the RAG test set, and its results |
| `ocr_cache/` | Cached OCR output per PDF and model |
| `chroma_store/` | Persistent local vector database |

## Requirements

- Python 3.10 or newer
- A project virtual environment
- A Google Gemini API key (paid tier recommended for internal documents)
- Sufficient disk space for the embedding model

Python packages are listed in `requirements.txt`:

- PyMuPDF for rendering PDF pages to images
- sentence-transformers for embeddings
- ChromaDB for vector storage
- rank-bm25 for keyword retrieval
- requests for Gemini API calls
- Streamlit for the web interface
- python-dotenv for loading the API key from `.env`

## Installation

From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Gemini API

Every LLM call (OCR, router, advanced-RAG helper stages, answer generation, evaluation judge)
goes to the Gemini API through its OpenAI-compatible endpoint. Embeddings and retrieval run
locally.

Set your key in the terminal before running anything (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Or put `GEMINI_API_KEY=your-key` in a `.env` file in the project folder (it is git-ignored).

Use a paid-tier key for internal documents. On the free tier Google may use prompts and
responses to improve its products; on the paid tier it does not.

### Models per role

| Role | Model | Why |
| --- | --- | --- |
| OCR of PDF pages | `gemini-3.5-flash-lite` | Near-perfect Arabic transcription in testing at a fraction of a cent per page |
| Router and helper stages (rewriting, multi-query, decomposition, HyDE, reranking, compression, CRAG) | `gemini-3.5-flash-lite` | Short classification, JSON and scoring calls; cheapest tier |
| Answer generation | `gemini-3.8-flash` | Newest Flash model; same price as `gemini-3.6-flash`. Falls back to `gemini-3.5-flash-lite` when rate-limited, overloaded or timing out |
| Evaluation judge | `gemini-3.8-flash` | Flash-tier judgement at low thinking effort |

Change them in `config.py` (`GEMINI_FAST_MODEL`, `GEMINI_MAIN_MODEL`, `OCR_GEMINI_MODEL`, or
any per-stage `*_MODEL`, or `GENERATION_FALLBACK_MODEL`).

Rate-limited calls wait as long as the API asks, up to `MAX_RETRY_WAIT_SECONDS`. If the wait
would be longer, such as an exhausted daily quota, the call fails at once.

### Cost and time per question

Every question prints (and the web UI shows) the cost and wall-clock time, plus every LLM
call with its tokens, thinking tokens, cost and latency. When the judge runs, its cost and
time are shown separately. Cost is computed from `MODEL_PRICING` in `config.py`, which holds the
standard paid-tier prices checked on 2026-10-20 and applies announced price changes on their
start date (for example the Flash price doubling on 2027-01-01). Thinking tokens are billed
as output tokens and are included. Update the table when Google changes prices or you switch
models: https://ai.google.dev/gemini-api/docs/pricing

## Build the Index

Place PDF files in `data/`, then run:

```powershell
python ingestion.py
```

Index construction performs these steps:

1. Render every PDF page to an image and transcribe it with Gemini vision (cached). The OCR
   prompt leaves out running headers and footers.
2. Warn about suspicious OCR output (mixed-script words, malformed common words).
3. Skip table-of-contents pages.
4. Read procedure titles, executing departments, executors and tables from the OCR Markdown.
   A procedure's context carries over from page to page.
5. Split each procedure on each page into overlapping chunks. Every chunk starts with its
   title, department and executor, and each table row keeps its column names.
6. Generate normalized `BAAI/bge-m3` embeddings.
7. Store text, embeddings, and metadata in ChromaDB.

The current chunk configuration is:

```text
Chunk size: 800 characters
Overlap: 150 characters
Collection: arabic_pdf_docs
```

Run the index build again whenever the PDFs change. The existing collection is replaced
when the index is rebuilt.

## PDF Extraction (OCR)

The PDFs in `data/` have a broken embedded text layer (missing letters, broken lam-alef
pairs, presentation-form glyphs), so text cannot be copied out of them directly. Instead,
each page is rendered to an image and transcribed by Gemini vision with a prompt tuned to
the manuals' layout: procedure title, `الجهة المنفذة:`, `## المنفذ:` role rows and numbered
steps. Running page headers and footers are dropped.

| Setting (`config.py`) | Value | Meaning |
|---|---|---|
| `OCR_GEMINI_MODEL` | `"fast"` | `"fast"` = `GEMINI_FAST_MODEL`, or a concrete model id |
| `OCR_DPI` | `200` | Resolution of the page image sent to Gemini |
| `OCR_STRIP_MARKDOWN` | `False` | Keep the OCR Markdown, which the chunker reads the structure from |

- OCR results are cached in `ocr_cache/`, per PDF content and model. Rebuilding the index
  does not re-OCR unchanged PDFs. Delete the folder (or a single page file in it) to force
  a fresh OCR.
- The OCR model can also be overridden for one run with the `OCR_GEMINI_MODEL` environment
  variable. `OCR_DEBUG=1` prints each call's token usage.
- After changing any OCR setting or the PDFs, run `python ingestion.py` to rebuild the index.

### Why Gemini Flash-Lite

Four pages of the Assets manual were typed by hand into `ground_truth/` and used to compare
extractors (lower character error rate is better):

| Extractor | Character error rate |
|---|---|
| Gemini Flash-Lite | 0.3% |
| Docling + RapidOCR | 22% (word order reversed inside table cells) |
| Docling + EasyOCR | 58% |
| PaddleOCR-VL | Did not run (out of memory on the test machine) |

OCR is a one-time, cached cost of roughly $0.002 per page (an estimate, not a measured
figure).

### Comparing OCR models

`ocr_compare.py` scores Gemini models (or DPI changes) against the ground-truth pages:

```powershell
python ocr_compare.py "data/Assets and wearhouse operation Tasks and Procedures Manual.pdf" --models gemini-3.5-flash-lite gemini-3.1-flash-lite
```

It prints character and word error rates, word F1 and warnings per model, and saves the raw
outputs in `ocr_out/`. Ground truth must be typed from the page image, not corrected from
one model's output, or the comparison favours that model.

## Run the Application

### Interactive command line

```powershell
python chat.py
```

Then enter a question, for example:

```text
ما هي طريقة تدوير المخزون؟
```

Exit with `exit`, `quit`, `q`, or `Ctrl+C`.

### One question from the command line

```powershell
python chat.py "ما هي طريقة تدوير المخزون؟"
```

For reliable Arabic input in PowerShell, save the command or script as UTF-8. If Arabic
appears as question marks, the terminal encoding or the pasted command is not being
handled as UTF-8; the application itself configures standard output for UTF-8 where
supported.

### Streamlit interface

```powershell
streamlit run chat.py
```

The interface displays the answer, source passages, page numbers, route details, model
path, token usage, latency, cost, and evaluation values.

## Query Routes

### Direct route

Used for greetings, questions about the assistant, and questions outside the manuals. It
replies to the first two and declines the rest, even when the model knows the answer. No
document retrieval occurs.

### Clarify route

Used when several different procedures could be meant and the question does not say which.
The reply lists the numbered procedures that retrieval actually found, with no LLM call. If
retrieval finds fewer than two, the question is answered through basic RAG instead.

### Basic RAG route

Used for a clear document question with one information need:

```text
router -> hybrid retrieval -> grounded generation
```

### Advanced RAG route

Used for vague, multi-part, conceptual, or metadata-constrained questions:

```text
router
  -> optional metadata filtering
  -> optional query rewriting
  -> decomposition or multi-query expansion
  -> optional HyDE query
  -> hybrid retrieval
  -> LLM reranking
  -> grounded generation
```

Contextual compression and CRAG are available but off (`USE_CONTEXTUAL_COMPRESSION` and
`USE_CRAG` in `config.py`). Turn them on only if `evaluate_testset.py` shows they improve
answers.

Generated rewrites, hypothetical passages, and search queries are used only to improve
retrieval. They are not passed to the final answer as document evidence.

## Retrieval Details

The normal retrieval mode combines two methods.

### Dense retrieval

The question and document chunks are represented as embeddings using `BAAI/bge-m3`.
This helps match paraphrases, synonyms, and semantically similar Arabic wording.

### BM25 keyword retrieval

BM25 finds exact word overlap. It is useful for names, codes, form titles, section headings,
dates, and exact Arabic terms.

### Reciprocal Rank Fusion

Dense and BM25 ranked lists are combined using Reciprocal Rank Fusion (RRF):

```text
RRF score = sum(1 / (k + rank + 1))
```

The current RRF constant is `60`. A result receives a stronger combined rank when it
appears near the top in both retrieval methods.

## Advanced Techniques

- **Query rewriting:** turns vague wording into a standalone search question.
- **Multi-query:** creates several phrasings of the same information need.
- **Decomposition:** splits a multi-part question into independent sub-questions.
- **HyDE:** creates a hypothetical passage to improve embedding-based search. It is never
  treated as evidence.
- **Self-query:** converts page, source, or section constraints into Chroma metadata filters.
  A manual can be named in Arabic ("في دليل الإنذار المركزي").
- **Reranking:** asks the helper model to score retrieved passages for direct relevance.
- **Contextual compression (off by default):** keeps only relevant original sentences from
  each passage.
- **CRAG (off by default):** grades retrieved context and retries retrieval once with a wider
  candidate pool when the evidence is ambiguous or incorrect.

## Grounding and Answer Behavior

The final prompt instructs the model to:

- Use only the supplied retrieved context for document claims.
- Answer the actual user question.
- Avoid inventing unsupported facts.
- State clearly when the context is insufficient.
- Preserve all important procedure steps.
- Include responsible roles, forms, records, and timing when available.
- Answer Arabic questions in Modern Standard Arabic, keeping system names and technical
  abbreviations such as BPM and RRN as they appear in the context.
- Ignore repeated administrative headers and footers.

Sources are shown in a separate Sources section with chunk IDs and page numbers. Any source
lines the model writes into the answer text anyway are removed
(`generation.strip_inline_sources`).

This reduces hallucination but cannot guarantee perfect answers. The model can still
misread an OCR error or make an incorrect interpretation.

## Evaluation

The LLM judge does not run while a user waits. It runs when a reference answer is passed to
`pipeline.answer(..., reference=...)`, which `evaluate_testset.py` does, or for every answer
if `EVALUATE_EVERY_ANSWER` is `True` in `config.py`.

```powershell
python evaluate_testset.py --retrieval-only
python evaluate_testset.py --delay 15
```

The first command checks locally, with no API calls, that the expected pages are retrieved.
The second runs the full pipeline and saves results to `ground_truth/eval_results.json` after
every question. It stops after two failures in a row. The test set is
`ground_truth/rag_testset.json`, git-ignored because it quotes the manuals.

Answers are evaluated using four scores from 0 to 5:

| Metric | Meaning |
|---|---|
| Context relevance | Whether the retrieved passages contain useful evidence |
| Faithfulness | Whether answer claims are supported by the retrieved context |
| Answer relevance | Whether the answer directly addresses the question |
| Correctness | Whether the answer matches a trusted reference answer |

Correctness is scored only when a reference answer is passed explicitly to
`pipeline.answer(..., reference=...)`. Otherwise it is `None` because no trusted answer is
available. Context relevance and faithfulness are `None` on the direct route, which has no
retrieved context.

The evaluator is also an LLM, so scores are useful for comparison and regression
detection but should not be treated as absolute ground truth.

## Troubleshooting

### Index collection not found

Run:

```powershell
python ingestion.py
```

### Cannot reach the Gemini API

Check that:

1. `GEMINI_API_KEY` is set in the same terminal you run the app from, or in `.env`.
2. The key is valid and your network or proxy allows `generativelanguage.googleapis.com`.
3. The model IDs in `config.py` match current Gemini model names.
4. A cost of `$0` with a "no price" warning means the model is missing from `MODEL_PRICING`.

### No PDFs found

Place at least one `.pdf` file inside `data/` and rebuild the index.

### OCR warnings during ingestion

`WARNING page N: suspicious words [...]` means a word mixes Arabic with another script.
This is often legitimate (an English term such as `MEPS` attached to an Arabic prefix), but
it can also be an OCR glitch. Open `ocr_cache/<pdf-hash>/<model>/page_NNN.md` and compare it
with the page in the PDF. To re-OCR one page, delete its file and run `python ingestion.py`.

### Answers are irrelevant

Check the following in order:

1. The OCR output in `ocr_cache/` for the relevant pages is correct.
2. The index was rebuilt after changing the PDFs.
3. The retrieved source pages are relevant.
4. The question is specific enough.
5. The router selected the correct route.
6. `python evaluate_testset.py --retrieval-only` still finds the expected pages.
7. `config.py` points at the intended Gemini models.

## Quick Presentation Summary

This project is an Arabic RAG system for querying internal procedure manuals. It transcribes
PDF pages with Gemini vision OCR (the PDFs' own text layer is corrupted), chunks the text by
procedure, embeds the chunks locally with `BAAI/bge-m3`, and stores them in ChromaDB. A router
chooses between a direct reply, basic hybrid RAG, advanced RAG, and asking a clarifying
question. Hybrid retrieval combines semantic embeddings with BM25 keyword search through
Reciprocal Rank Fusion. The advanced pipeline can rewrite, decompose, and rerank. The final
model answers only from retrieved document passages and returns source information. Every
request reports token, cost, and latency data, and a separate script evaluates quality on a
test set.
